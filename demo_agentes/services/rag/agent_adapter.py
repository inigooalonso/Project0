"""Adaptador REAL del RAG: tu tool inject_query_context → retrieve_context_for_sql.

Contrato propuesto para que la vista pinte el árbol con puntuaciones: cada
elemento de ``schema_context`` como
    {"entity_id", "owner", "table", "field", "description", "score"}.
Si tu RAG todavía no lo devuelve así (hoy devuelve el texto de
``authorized_tables``), el adaptador construye el árbol a partir de ese texto,
sin puntuaciones. El dialecto se fuerza al configurado (Athena).
"""
from __future__ import annotations

import json
import re
from typing import Any

from core.models import EntitySearch, FieldMatch, OwnerMatch, Pydantic_SemanticQueryIR, RAGResult, TableMatch
from services.agent.langgraph_adapter import load_agent_module
from services.errors import ServiceError, describe_exception
from services.rag.base import ir_items
from services.watchdog import run_with_timeout

UUAA = re.compile(r"(?:^|\.)t_([a-z0-9]{4})_", re.IGNORECASE)


def owner_code(table: str) -> str:
    match = UUAA.search(table)
    return match.group(1).lower() if match else table.split(".")[0]


def parse_table_block(block: str) -> dict[str, Any] | None:
    """Interpreta el formato de la variable `table` del agente.

    ho_master.t_xxx:
    -Description:...
    -Fields:campo: etiqueta, descripción
    * campo: etiqueta, descripción
    """
    lines = [line.strip() for line in str(block).strip().splitlines() if line.strip()]
    if not lines or not lines[0].endswith(":"):
        return None
    table = {"name": lines[0][:-1].strip(), "description": "", "fields": []}
    for line in lines[1:]:
        if line.startswith("-Description:"):
            table["description"] = line[len("-Description:"):].strip()
            continue
        if line.startswith("-Fields:"):
            line = line[len("-Fields:"):]
        elif line.startswith("* "):
            line = line[2:]
        else:
            continue
        name, _, rest = line.partition(":")
        label, _, description = rest.strip().partition(",")
        if name.strip():
            table["fields"].append({"name": name.strip(), "label": label.strip(), "description": description.strip()})
    return table


class AgentRAGAdapter:
    name = "Tu RAG · retrieve_context_for_sql"
    simulated = False

    def __init__(self, module_path: str, dialect: str, timeout_s: float = 30.0) -> None:
        self.module_path = module_path
        self.dialect = dialect
        self.timeout_s = timeout_s

    def search(self, ir: Pydantic_SemanticQueryIR) -> RAGResult:
        module = load_agent_module(self.module_path)
        try:
            raw = run_with_timeout(module.inject_query_context.invoke, {"semantic_ir_json": ir.model_dump_json()},
                                   timeout=self.timeout_s, service="rag", what="Tu RAG")
            context = json.loads(raw)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("rag", "Tu RAG no ha respondido",
                               "La búsqueda de contexto ha fallado.", describe_exception(exc)) from exc

        original_dialect = context.get("dialect")
        context["dialect"] = self.dialect
        blocks = [b for b in context.get("authorized_tables", []) if isinstance(b, str)]
        parsed = [t for t in (parse_table_block(b) for b in blocks) if t]
        schema_items = [i for i in context.get("schema_context", []) if isinstance(i, dict) and i.get("table")]
        has_scores = any(isinstance(i.get("score"), (int, float)) for i in schema_items)

        owners: dict[str, OwnerMatch] = {}

        def table_node(name: str, description: str = "") -> TableMatch:
            code = owner_code(name)
            owner = owners.setdefault(code, OwnerMatch(code=code, name=f"UUAA {code}", selected=True))
            for t in owner.tables:
                if t.name == name:
                    return t
            node = TableMatch(name=name, label=name.split(".")[-1], description=description, selected=True)
            owner.tables.append(node)
            return node

        for t in parsed:
            node = table_node(t["name"], t["description"])
            node.fields = [FieldMatch(name=f["name"], label=f["label"], description=f["description"]) for f in t["fields"]]
        for item in schema_items:
            node = table_node(str(item["table"]))
            if item.get("owner"):
                owners[owner_code(node.name)].name = str(item["owner"])
            score = item.get("score") if isinstance(item.get("score"), (int, float)) else None
            existing = next((f for f in node.fields if f.name == item.get("field")), None)
            if existing is None and item.get("field"):
                existing = FieldMatch(name=str(item["field"]), description=str(item.get("description", "")))
                node.fields.append(existing)
            if existing is not None:
                existing.selected = True
                existing.score = max(existing.score or 0, score) if score is not None else existing.score
                if item.get("entity_id"):
                    existing.entity_ids.append(str(item["entity_id"]))
            if score is not None:
                node.score = max(node.score or 0, score)
        for owner in owners.values():
            scores = [t.score for t in owner.tables if t.score is not None]
            owner.score = max(scores) if scores else None

        searches = []
        by_entity = {str(i.get("entity_id")): i for i in schema_items if i.get("entity_id")}
        for item in ir_items(ir):
            hit = by_entity.get(item.id)
            searches.append(EntitySearch(
                entity_id=item.id, kind=item.kind, surface_form=item.surface_form, entity=item.entity, concept=item.concept,
                owner=owner_code(str(hit["table"])) if hit else None, table=str(hit["table"]) if hit else None,
                field=str(hit.get("field")) if hit else None,
                field_score=hit.get("score") if hit and isinstance(hit.get("score"), (int, float)) else None,
            ))

        selected_tables = [t.name for o in owners.values() for t in o.tables]
        raw_context = dict(context, original_dialect=original_dialect)
        return RAGResult(
            source="agent", dialect=self.dialect, owners=list(owners.values()), searches=searches,
            selected_tables=selected_tables, authorized_tables=list(context.get("authorized_tables", [])),
            schema_context=list(context.get("schema_context", [])),
            business_context=list(context.get("business_context", [])),
            join_rules=list(context.get("join_rules", [])), has_scores=has_scores, raw_context=raw_context,
        )
