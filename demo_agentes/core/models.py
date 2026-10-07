"""Modelos de dominio que comparten orquestación, servicios e interfaz.

El IR es la clase original del agente (agents/ada_text2sql/semantic_ir.py).
SQLDraft replica el modelo de salida de agents/ada_text2sql/agent.py, que no
se puede importar sin las librerías de AWS.
"""
from __future__ import annotations

from dataclasses import dataclass
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
# Salidas del LLM
# ---------------------------------------------------------------------

class ClarificationBatch(BaseModel):
    """Aclaraciones de una ronda: el agente puede hacer varias preguntas a la vez."""

    needs_clarification: bool
    questions: list[str] = Field(default_factory=list)


class SQLDraft(BaseModel):
    """Réplica de SQLDraft de agents/ada_text2sql/agent.py."""

    sql: str
    assumptions: list[str] = Field(default_factory=list)


@dataclass
class AgentCall(Generic[T]):
    """Resultado de una llamada al agente, con la telemetría capturada."""

    output: T
    input_tokens: int | None = None
    output_tokens: int | None = None
    prompts: list[dict[str, str]] | None = None


# ---------------------------------------------------------------------
# Paso 2 · RAG multinivel
# ---------------------------------------------------------------------

class FieldInfo(BaseModel):
    name: str
    label: str = ""
    description: str = ""


class TableInfo(BaseModel):
    """Tabla autorizada, leída del bloque de texto de authorized_tables."""

    name: str  # nombre completo: base.tabla
    description: str = ""
    fields: list[FieldInfo] = Field(default_factory=list)

    @property
    def short_name(self) -> str:
        return self.name.split(".")[-1]


class RAGCandidate(BaseModel):
    """Una fila de la tabla de candidatos del RAG (una entidad del IR frente a una tabla)."""

    entity: str = ""
    type: str = ""
    table: str = ""
    meets_grain: bool | None = None
    sim_uuaa: float | None = None
    sim_table: float | None = None
    sim_field: float | None = None
    sim_weighted: float | None = None


class RAGResult(BaseModel):
    source: str
    dialect: str
    candidates: list[RAGCandidate] = Field(default_factory=list)
    unified: list[dict[str, Any]] = Field(default_factory=list)  # filas con los nombres de columna de pantalla
    unified_derived: bool = False  # True si tu RAG no la aporta y se calcula aquí
    tables: list[TableInfo] = Field(default_factory=list)
    authorized_tables: list[str] = Field(default_factory=list)  # bloques de texto en tu formato
    schema_context: list[Any] = Field(default_factory=list)
    business_context: list[Any] = Field(default_factory=list)
    join_rules: list[Any] = Field(default_factory=list)
    raw_context: dict[str, Any] | None = None
    # Avisos sobre los campos de schema_context:
    unauthorized_field_tables: list[str] = Field(default_factory=list)  # campos de tablas no autorizadas
    duplicated_fields: int = 0  # campos repetidos entre bloques

    @property
    def selected_tables(self) -> list[str]:
        return [t.name for t in self.tables]

    @property
    def field_count(self) -> int:
        return sum(len(t.fields) for t in self.tables)


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
