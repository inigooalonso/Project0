"""Modelos de dominio que comparten orquestación, servicios e interfaz.

El IR es la clase original del agente (agents/ada_text2sql/semantic_ir.py).
ClarificationDecision y SQLDraft replican campo a campo los modelos de salida
de agents/ada_text2sql/agent.py, que no se puede importar sin las librerías de
AWS; así el modo simulado produce exactamente las mismas estructuras.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

import pandas as pd
from pydantic import BaseModel, Field

from agents.ada_text2sql.semantic_ir import Pydantic_SemanticQueryIR  # noqa: F401  (reexportado)

T = TypeVar("T")


# ---------------------------------------------------------------------
# Pasos del pipeline
# ---------------------------------------------------------------------

class StepId(str, Enum):
    PSEUDOCODE = "pseudocode"
    RAG = "rag"
    JOINS = "joins"
    CONTEXT = "context"
    CLARIFY = "clarify"
    SQL = "sql"
    EXECUTE = "execute"


STEP_ORDER: list[StepId] = list(StepId)


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"


# ---------------------------------------------------------------------
# Salidas del LLM (réplica de los modelos del agente)
# ---------------------------------------------------------------------

class ClarificationDecision(BaseModel):
    needs_clarification: bool
    question: str | None = None


class SQLDraft(BaseModel):
    sql: str
    assumptions: list[str] = Field(default_factory=list)


@dataclass
class AgentCall(Generic[T]):
    """Resultado de una llamada al agente, con la telemetría capturada."""

    output: T
    input_tokens: int | None = None
    output_tokens: int | None = None
    tokens_estimated: bool = False
    prompts: list[dict[str, str]] | None = None


class ClarificationOption(BaseModel):
    value: str
    label: str


# ---------------------------------------------------------------------
# Paso 2 · RAG multinivel
# ---------------------------------------------------------------------

class FieldMatch(BaseModel):
    name: str
    label: str = ""
    description: str = ""
    type: str = ""
    role: str | None = None
    values: list[str] = Field(default_factory=list)
    score: float | None = None
    selected: bool = False
    entity_ids: list[str] = Field(default_factory=list)


class TableMatch(BaseModel):
    name: str  # nombre completo: base.tabla
    label: str = ""
    description: str = ""
    score: float | None = None
    selected: bool = False
    fields: list[FieldMatch] = Field(default_factory=list)

    @property
    def short_name(self) -> str:
        return self.name.split(".")[-1]


class OwnerMatch(BaseModel):
    code: str
    name: str
    business_owner: str | None = None
    score: float | None = None
    selected: bool = False
    tables: list[TableMatch] = Field(default_factory=list)


class EntitySearch(BaseModel):
    """Traza de la búsqueda en tres niveles para una entidad del IR."""

    entity_id: str
    kind: str  # metric | dimension | attribute | filter | time_range
    surface_form: str
    entity: str
    concept: str
    owner: str | None = None
    owner_score: float | None = None
    table: str | None = None
    table_score: float | None = None
    field: str | None = None
    field_score: float | None = None
    candidates_seen: int = 0


class RAGResult(BaseModel):
    source: str
    dialect: str
    owners: list[OwnerMatch] = Field(default_factory=list)
    searches: list[EntitySearch] = Field(default_factory=list)
    selected_tables: list[str] = Field(default_factory=list)
    authorized_tables: list[str] = Field(default_factory=list)  # bloques de texto en tu formato
    schema_context: list[dict[str, Any]] = Field(default_factory=list)
    business_context: list[Any] = Field(default_factory=list)
    join_rules: list[Any] = Field(default_factory=list)
    has_scores: bool = True
    raw_context: dict[str, Any] | None = None

    @property
    def selected_field_count(self) -> int:
        return sum(1 for o in self.owners for t in o.tables for f in t.fields if f.selected)


# ---------------------------------------------------------------------
# Paso 3 · Joins y glosario
# ---------------------------------------------------------------------

class JoinRule(BaseModel):
    left_table: str
    left_field: str
    right_table: str
    right_field: str
    type: str = "INNER"
    cardinality: str = "N:1"
    description: str = ""

    @property
    def on_clause(self) -> str:
        return f"{self.left_table}.{self.left_field} = {self.right_table}.{self.right_field}"

    def as_context(self) -> dict[str, str]:
        return {
            "tables": [self.left_table, self.right_table],
            "on": self.on_clause,
            "type": self.type,
            "cardinality": self.cardinality,
            "description": self.description,
        }


class GlossaryTerm(BaseModel):
    term: str
    synonyms: list[str] = Field(default_factory=list)
    definition: str
    formula: str | None = None
    fields: list[str] = Field(default_factory=list)
    matched_by: str | None = None  # fragmento de la pregunta que lo activa
    score: float | None = None

    def as_context(self) -> dict[str, Any]:
        item: dict[str, Any] = {"term": self.term, "definition": self.definition, "fields": self.fields}
        if self.formula:
            item["formula"] = self.formula
        if self.synonyms:
            item["synonyms"] = self.synonyms
        return item


class KnowledgeResult(BaseModel):
    tables: list[str] = Field(default_factory=list)
    bridge_tables: list[str] = Field(default_factory=list)
    disconnected_tables: list[str] = Field(default_factory=list)
    joins: list[JoinRule] = Field(default_factory=list)
    glossary: list[GlossaryTerm] = Field(default_factory=list)
    ambiguous_terms: dict[str, list[str]] = Field(default_factory=dict)
    source: str = ""


# ---------------------------------------------------------------------
# Paso 4 · Contexto
# ---------------------------------------------------------------------

class ContextBundle(BaseModel):
    context: dict[str, Any]
    table_count: int = 0
    field_count: int = 0
    fragment_count: int = 0
    join_count: int = 0
    glossary_count: int = 0
    tokens_by_block: dict[str, int] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens_by_block.values())


# ---------------------------------------------------------------------
# Paso 6 · SQL
# ---------------------------------------------------------------------

class SQLInspection(BaseModel):
    parsed: bool = True
    parse_error: str | None = None
    read_only: bool = True
    single_statement: bool = True
    tables: list[str] = Field(default_factory=list)
    unauthorized_tables: list[str] = Field(default_factory=list)
    join_count: int = 0
    cte_count: int = 0
    has_limit: bool = False
    order_by: list[str] = Field(default_factory=list)


class SQLResult(BaseModel):
    sql: str
    assumptions: list[str] = Field(default_factory=list)
    inspection: SQLInspection = Field(default_factory=SQLInspection)


# ---------------------------------------------------------------------
# Paso 7 · Ejecución
# ---------------------------------------------------------------------

@dataclass
class ExecutionResult:
    df: pd.DataFrame
    engine: str
    elapsed_s: float
    row_count: int
    truncated: bool = False
    executed_sql: str = ""
    simulated: bool = True


@dataclass
class Kpi:
    label: str
    value: str
    caption: str = ""


@dataclass
class ResultProfile:
    """Lectura de la forma del resultado para elegir gráfico y KPIs."""

    kind: str  # kpi | line | bar | table
    x: str | None = None
    y: str | None = None
    series: str | None = None
    measures: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    times: list[str] = field(default_factory=list)
    measure_kinds: dict[str, str] = field(default_factory=dict)  # currency | percent | count | number
    descending: bool = True
