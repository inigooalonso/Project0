"""Interfaz del agente LLM (pasos 1, 5 y 6).

``state`` usa las mismas claves que SQLAgentState del agente: user_query,
semantic_ir, context, clarifications y pending_question.
"""
from __future__ import annotations

from typing import Any, Protocol

from core.models import AgentCall, ClarificationBatch, Pydantic_SemanticQueryIR, SQLDraft


class AgentService(Protocol):
    name: str

    def parse(self, question: str) -> AgentCall[Pydantic_SemanticQueryIR]:
        """Paso 1: pregunta → IR semántico (parse_semantic_query)."""
        ...

    def decide_clarifications(self, state: dict[str, Any], max_questions: int) -> AgentCall[ClarificationBatch]:
        """Paso 5: todas las preguntas que hacen falta en esta ronda (lista vacía si ninguna)."""
        ...

    def generate_sql(self, state: dict[str, Any]) -> AgentCall[SQLDraft]:
        """Paso 6: SQL de solo lectura (generate_sql + validate_read_only_sql)."""
        ...


def clarification_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Mismo JSON que envía decide_if_clarification_is_needed como mensaje de usuario."""
    return {
        "semantic_ir": state["semantic_ir"].model_dump(),
        "context": state["context"],
        "prior_answers": state.get("clarifications", []),
    }


def estimate_tokens(value: Any) -> int:
    """Estimación de tokens (~3,6 caracteres por token en español con JSON)."""
    import json
    import math

    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return max(1, math.ceil(len(text) / 3.6))
