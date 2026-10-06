"""Adaptador del RAG: tu tool inject_query_context → retrieve_context_for_sql.

Lee del contexto que devuelve tu RAG:
- ``authorized_tables``: bloques de texto con tu formato (tabla, descripción, campos);
- ``rag_candidates`` y ``rag_unified``: tablas de similitud que se pintan en el
  paso 2 (formato en services/rag/tables.py).
El dialecto se fuerza al configurado (Athena).
"""
from __future__ import annotations

import json
from typing import Any

from core.models import FieldInfo, Pydantic_SemanticQueryIR, RAGResult, TableInfo
from services.agent.langgraph_adapter import load_agent_module
from services.errors import ServiceError, describe_exception
from services.rag.tables import derive_unified, parse_candidates, parse_unified
from services.watchdog import run_with_timeout


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
        if not isinstance(context, dict):
            raise ServiceError("rag", "Tu RAG no ha devuelto un contexto válido",
                               "retrieve_context_for_sql debe devolver un diccionario.", str(raw)[:300])
        return build_result(context, self.dialect, self.name)


def build_result(context: dict[str, Any], dialect: str, source: str) -> RAGResult:
    original_dialect = context.get("dialect")
    blocks = [b for b in context.get("authorized_tables", []) if isinstance(b, str)]
    tables = []
    for block in blocks:
        parsed = parse_table_block(block)
        if parsed:
            tables.append(TableInfo(name=parsed["name"], description=parsed["description"],
                                    fields=[FieldInfo(**f) for f in parsed["fields"]]))
    candidates = parse_candidates(context)
    unified = parse_unified(context)
    derived = unified is None and bool(candidates)
    if unified is None:
        unified = derive_unified(candidates) if candidates else []
    return RAGResult(
        source=source, dialect=dialect, candidates=candidates, unified=unified, unified_derived=derived,
        tables=tables, authorized_tables=blocks,
        schema_context=list(context.get("schema_context") or []),
        business_context=list(context.get("business_context") or []),
        join_rules=list(context.get("join_rules") or []),
        raw_context=dict(context, original_dialect=original_dialect, dialect=dialect),
    )
