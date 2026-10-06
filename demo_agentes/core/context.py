"""Paso 4: ensambla el contexto exacto que reciben los nodos del agente.

Mismas claves que devuelve tu retrieve_context_for_sql: dialect,
authorized_tables (bloques de texto con tu formato), schema_context,
business_context y join_rules.
"""
from __future__ import annotations

from core.models import ContextBundle, KnowledgeResult, RAGResult
from services.agent.base import estimate_tokens


def assemble_context(rag: RAGResult, knowledge: KnowledgeResult, dialect: str) -> ContextBundle:
    authorized = list(rag.authorized_tables)
    for bridge in knowledge.bridge_tables:
        authorized.append(f"{bridge}:\n-Description:Tabla puente definida en las reglas de join.")
    business = list(rag.business_context) + [g.as_context() for g in knowledge.glossary]
    joins = list(rag.join_rules) + [j.as_context() for j in knowledge.joins]
    context = {
        "dialect": dialect,
        "authorized_tables": authorized,
        "schema_context": list(rag.schema_context),
        "business_context": business,
        "join_rules": joins,
    }
    field_count = sum(max(block.count("\n* ") + 1, 1) for block in authorized if isinstance(block, str))
    return ContextBundle(
        context=context,
        table_count=len(authorized),
        field_count=field_count,
        fragment_count=len(context["schema_context"]),
        join_count=len(joins),
        glossary_count=len(business),
        tokens_by_block={
            "Tablas autorizadas": estimate_tokens(authorized) if authorized else 0,
            "Fragmentos del RAG": estimate_tokens(context["schema_context"]) if context["schema_context"] else 0,
            "Glosario": estimate_tokens(business) if business else 0,
            "Reglas de join": estimate_tokens(joins) if joins else 0,
        },
    )
