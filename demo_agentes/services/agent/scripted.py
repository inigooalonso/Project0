"""Agente SIMULADO: devuelve las respuestas guionizadas de scenarios.yaml.

Produce exactamente las estructuras del agente real (IR validado con
Pydantic_SemanticQueryIR, ClarificationDecision y SQLDraft) y estima los tokens
a partir de los mismos mensajes que construyen tus nodos.
"""
from __future__ import annotations

import json
from typing import Any

from core.models import AgentCall, ClarificationDecision, ClarificationOption, Pydantic_SemanticQueryIR, SQLDraft
from services.agent.base import clarification_payload, estimate_tokens, sql_payload
from services.errors import UnsupportedQuestion
from services.scenarios import Scenario, load_scenarios, match_scenario
from services.sql_guard import validate_read_only

# Tamaño aproximado de las instrucciones de sistema de cada nodo (prompt + formato JSON).
SYSTEM_TOKENS = {"parse": 1650, "clarify": 420, "sql": 380}


class ScriptedAgent:
    name = "LLM simulado (guion)"
    simulated = True

    def __init__(self, scenarios: tuple[Scenario, ...] | None = None, max_clarifications: int = 3) -> None:
        self.scenarios = scenarios if scenarios is not None else load_scenarios()
        self.max_clarifications = max_clarifications

    def scenario(self, question: str) -> Scenario:
        found = match_scenario(question, self.scenarios)
        if found is None:
            raise UnsupportedQuestion(question)
        return found

    def _option(self, scenario: Scenario, state: dict[str, Any]) -> str | None:
        answers = state.get("clarifications") or []
        return scenario.option_for(answers[-1]["answer"]) if answers else None

    # ------------------------------------------------------------------
    def parse(self, question: str) -> AgentCall[Pydantic_SemanticQueryIR]:
        ir = self.scenario(question).ir()
        return AgentCall(
            output=ir,
            input_tokens=SYSTEM_TOKENS["parse"] + estimate_tokens(question),
            output_tokens=estimate_tokens(ir.model_dump_json()),
            tokens_estimated=True,
            prompts=[{"role": "system", "content": "prompt_semantic_ir_outbound + instrucciones de formato JSON"},
                     {"role": "human", "content": question}],
        )

    def decide_clarification(self, state: dict[str, Any]) -> AgentCall[ClarificationDecision]:
        scenario = self.scenario(state["user_query"])
        prior = state.get("clarifications") or []
        ask = scenario.has_clarification and not prior and len(prior) < self.max_clarifications
        decision = ClarificationDecision(
            needs_clarification=ask, question=scenario.clarification_question if ask else None
        )
        payload = json.dumps(clarification_payload(state), ensure_ascii=False, default=str)
        return AgentCall(
            output=decision,
            input_tokens=SYSTEM_TOKENS["clarify"] + estimate_tokens(payload),
            output_tokens=estimate_tokens(decision.model_dump_json()),
            tokens_estimated=True,
            prompts=[{"role": "system", "content": "Instrucciones de decide_if_clarification_is_needed"},
                     {"role": "human", "content": payload}],
        )

    def generate_sql(self, state: dict[str, Any]) -> AgentCall[SQLDraft]:
        scenario = self.scenario(state["user_query"])
        option = self._option(scenario, state)
        draft = SQLDraft(sql=validate_read_only(scenario.sql_for(option)), assumptions=scenario.assumptions_for(option))
        payload = json.dumps(sql_payload(state), ensure_ascii=False, default=str)
        return AgentCall(
            output=draft,
            input_tokens=SYSTEM_TOKENS["sql"] + estimate_tokens(payload),
            output_tokens=estimate_tokens(draft.model_dump_json()),
            tokens_estimated=True,
            prompts=[{"role": "system", "content": "Instrucciones de generate_sql"}, {"role": "human", "content": payload}],
        )

    def clarification_options(self, state: dict[str, Any]) -> list[ClarificationOption]:
        return list(self.scenario(state["user_query"]).options)

    def explanation(self, state: dict[str, Any]) -> str | None:
        scenario = self.scenario(state["user_query"])
        return scenario.explanation_for(self._option(scenario, state)) or None
