"""Interfaz del agente LLM (pasos 1, 5 y 6).

``state`` usa las mismas claves que SQLAgentState del agente: user_query,
semantic_ir, context, clarifications y pending_question.
"""
from __future__ import annotations

import json
import math
from typing import Any, Protocol

from core.models import AgentCall, ClarificationDecision, ClarificationOption, Pydantic_SemanticQueryIR, SQLDraft


class AgentService(Protocol):
    name: str
    simulated: bool

    def parse(self, question: str) -> AgentCall[Pydantic_SemanticQueryIR]:
        """Paso 1: pregunta → IR semántico (parse_semantic_query)."""
        ...

    def decide_clarification(self, state: dict[str, Any]) -> AgentCall[ClarificationDecision]:
        """Paso 5: ¿hace falta preguntar? (decide_if_clarification_is_needed)."""
        ...

    def generate_sql(self, state: dict[str, Any]) -> AgentCall[SQLDraft]:
        """Paso 6: SQL de solo lectura (generate_sql + validate_read_only_sql)."""
        ...

    def clarification_options(self, state: dict[str, Any]) -> list[ClarificationOption]:
        """Respuestas rápidas para la aclaración pendiente (vacío si no hay)."""
        ...

    def explanation(self, state: dict[str, Any]) -> str | None:
        """Cómo se ha calculado, en lenguaje llano (None si el agente no la aporta)."""
        ...


def estimate_tokens(value: Any) -> int:
    """Estimación de tokens (~3,6 caracteres por token en español con JSON)."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return max(1, math.ceil(len(text) / 3.6))


def clarification_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Mismo JSON que envía decide_if_clarification_is_needed como mensaje de usuario."""
    return {
        "semantic_ir": state["semantic_ir"].model_dump(),
        "context": state["context"],
        "prior_answers": state.get("clarifications", []),
    }


def sql_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Mismo JSON que envía generate_sql como mensaje de usuario."""
    return {
        "semantic_ir": state["semantic_ir"].model_dump(),
        "context": state["context"],
        "clarifications": state.get("clarifications", []),
    }
