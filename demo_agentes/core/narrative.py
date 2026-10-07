"""Textos en lenguaje llano: qué hace cada paso, qué ha entendido el agente y
el resumen de una línea que se ve en el stepper."""
from __future__ import annotations

from dataclasses import dataclass

from core.formatting import fmt_date, fmt_int, fmt_seconds, fmt_tokens
from core.models import (
    ContextBundle,
    ExecutionResult,
    KnowledgeResult,
    Pydantic_SemanticQueryIR,
    RAGResult,
    SQLResult,
    StepId,
)


@dataclass(frozen=True)
class StepInfo:
    number: int
    title: str
    stage_title: str
    icon: str
    what: str
    why: str


STEP_INFO: dict[StepId, StepInfo] = {
    StepId.PSEUDOCODE: StepInfo(
        1, "Pseudocódigo", "Lo que ha entendido el agente", ":material/psychology:",
        "Traduce la pregunta a una estructura formal (JSON) validada contra el esquema Pydantic_SemanticQueryIR.",
        "Antes de tocar un dato, el agente deja por escrito qué ha entendido.",
    ),
    StepId.RAG: StepInfo(
        2, "RAG multinivel", "Dónde están los datos", ":material/account_tree:",
        "Para cada entidad detectada busca en el catálogo en tres niveles (UUAA, tabla y campo) y comprueba el grain.",
        "Solo trabaja con datos catalogados y sabe con qué confianza encontró cada uno.",
    ),
    StepId.JOINS: StepInfo(
        3, "Joins y glosario", "Cómo se relacionan y qué significan", ":material/hub:",
        "Calcula cómo se unen las tablas y añade las definiciones oficiales del glosario de negocio.",
        "Las relaciones y las definiciones están gobernadas: el agente no se las inventa.",
    ),
    StepId.CONTEXT: StepInfo(
        4, "Contexto", "Qué recibe el LLM", ":material/inventory_2:",
        "Ensambla lo encontrado en el contexto exacto que recibe el LLM.",
        "El LLM solo ve tablas autorizadas y definiciones aprobadas.",
    ),
    StepId.CLARIFY: StepInfo(
        5, "Aclaraciones", "Dudas antes de seguir", ":material/forum:",
        "El LLM decide si la pregunta es ambigua. Si lo es, pregunta y espera la respuesta antes de seguir.",
        "Ante la duda, pregunta: evita respuestas plausibles pero equivocadas.",
    ),
    StepId.SQL: StepInfo(
        6, "SQL", "La consulta generada", ":material/code:",
        "Genera una única consulta SQL de solo lectura y la valida antes de ejecutarla.",
        "Solo lectura, solo tablas autorizadas y supuestos explícitos.",
    ),
    StepId.EXECUTE: StepInfo(
        7, "Ejecución", "Resultado", ":material/insights:",
        "Ejecuta la consulta en Athena y muestra el resultado tal como lo devuelve la base de datos.",
        "De la pregunta al dato, con la trazabilidad completa.",
    ),
}

INTENT_LABELS = {
    "aggregate_analysis": "Agregado", "detail_retrieval": "Detalle", "comparison": "Comparación",
    "trend_analysis": "Evolución", "ranking": "Ranking", "distribution": "Distribución",
    "distinct_values": "Valores distintos", "count": "Recuento", "existence": "Existencia", "unknown": "Sin clasificar",
}


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def _plural(n: int, singular: str, plural: str) -> str:
    return f"{fmt_int(n)} {singular if n == 1 else plural}"


def _quoted(items: list[str]) -> str:
    return _join([f"«{i}»" for i in items if i])


def describe_ir(ir: Pydantic_SemanticQueryIR) -> str:
    """«Lo que he entendido», en una frase. Cita los términos de negocio entre comillas."""
    metrics = _quoted([m.concept for m in ir.metrics]) or "los datos"
    dims = _quoted([d.surface_form for d in ir.dimensions if d.grain is None])
    time_dims = _join([d.surface_form for d in ir.dimensions if d.grain is not None])
    desc = any(o.direction == "desc" for o in ir.order_by)
    intent = ir.intent.value
    if intent == "ranking":
        lead = f"Ordenar {dims or 'los resultados'} según {metrics}" + (", de mayor a menor" if desc else "")
    elif intent == "trend_analysis":
        lead = f"Ver la evolución de {metrics}" + (f" {time_dims}" if time_dims else "")
    elif intent == "distribution":
        lead = f"Repartir {metrics} por {dims}" if dims else f"Ver la distribución de {metrics}"
    elif intent == "comparison":
        lead = f"Comparar {metrics}" + (f" entre {dims}" if dims else "")
    elif intent == "count":
        lead = f"Contar {metrics}" + (f" por {dims}" if dims else "")
    elif intent == "detail_retrieval":
        lead = f"Listar {_quoted([a.surface_form for a in ir.attributes]) or dims or metrics}"
    else:
        lead = f"Calcular {metrics}" + (f" por {dims}" if dims else "")
    parts = [lead]
    parts += [f"solo «{f.surface_form}»" for f in ir.filters]
    if ir.time_range is not None:
        parts.append(ir.time_range.surface_form)
    if ir.limit:
        parts.append(f"los {ir.limit} primeros")
    text = " · ".join(parts)
    return text[:1].upper() + text[1:] + "."


def period_text(ir: Pydantic_SemanticQueryIR) -> str | None:
    tr = ir.time_range
    if tr is None:
        return None
    if tr.start and tr.end:
        return f"{fmt_date(tr.start)} – {fmt_date(tr.end)}"
    return f"desde {fmt_date(tr.start)}" if tr.start else f"hasta {fmt_date(tr.end)}"


# ---------------------------------------------------------------------
# Resúmenes de una línea para el stepper
# ---------------------------------------------------------------------

def summary_pseudocode(ir: Pydantic_SemanticQueryIR) -> str:
    parts = [INTENT_LABELS.get(ir.intent.value, ir.intent.value)]
    parts.append(_plural(len(ir.metrics), "métrica", "métricas"))
    if ir.dimensions:
        parts.append(_plural(len(ir.dimensions), "dimensión", "dimensiones"))
    if ir.filters:
        parts.append(_plural(len(ir.filters), "filtro", "filtros"))
    if ir.time_range is not None:
        parts.append("periodo")
    return " · ".join(parts)


def summary_rag(rag: RAGResult) -> str:
    parts = [_plural(len(rag.tables), "tabla", "tablas")]
    if rag.candidates:
        parts.append(_plural(len(rag.candidates), "candidato", "candidatos"))
        off = sum(1 for c in rag.candidates if c.meets_grain is False)
        if off:
            parts.append(_plural(off, "sin grain", "sin grain"))
    return " · ".join(parts)


def summary_joins(knowledge: KnowledgeResult) -> str:
    parts = [_plural(len(knowledge.joins), "join", "joins"), _plural(len(knowledge.glossary), "término", "términos")]
    if knowledge.bridge_tables:
        parts.append(_plural(len(knowledge.bridge_tables), "tabla puente", "tablas puente"))
    return " · ".join(parts)


def summary_context(bundle: ContextBundle) -> str:
    return f"{_plural(bundle.table_count, 'tabla autorizada', 'tablas autorizadas')} · ≈ {fmt_tokens(bundle.total_tokens)} tokens"


def summary_clarify(run, waiting: bool) -> str:
    if waiting:
        n = len(run.rounds[-1].questions)
        return "Esperando " + ("tu respuesta" if n == 1 else f"{n} respuestas")
    answered = len(run.state.get("clarifications", []))
    if answered == 0:
        return "Sin dudas: no hace falta preguntar"
    return _plural(answered, "aclaración resuelta", "aclaraciones resueltas")


def summary_sql(result: SQLResult) -> str:
    joins = result.inspection.join_count
    parts = ["Solo lectura validada", _plural(joins, "join", "joins")]
    if result.assumptions:
        parts.append(_plural(len(result.assumptions), "supuesto", "supuestos"))
    return " · ".join(parts)


def summary_execute(result: ExecutionResult) -> str:
    return f"{_plural(result.row_count, 'fila', 'filas')} en {fmt_seconds(result.elapsed_s)}"
