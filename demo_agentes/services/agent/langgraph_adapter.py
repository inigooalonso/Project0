"""Adaptador: llama uno a uno a los nodos de tu agente LangGraph.

No se invoca sql_agent_graph de una vez para poder intercalar los pasos que no
están en el grafo (joins y glosario, contexto, ejecución), medir cada paso y
pausar en st.session_state. La pausa de ask_user (interrupt) la reproduce la
máquina de estados con el mismo contrato: clarifications = [{question, answer}].

Aclaraciones (agent.clarification_mode en config/settings.toml):
- "node" (por defecto): se llama a tu nodo decide_if_clarification_is_needed
  y su pending_question es la pregunta de la ronda;
- "batch": el paso 5 usa tu invoke_pydantic y tu llm con el mismo mensaje de
  usuario y un prompt que pide todas las preguntas a la vez.
Tu código no se modifica.
"""
from __future__ import annotations

import importlib
import json
import re
from typing import Any

from core.models import AgentCall, ClarificationBatch, Pydantic_SemanticQueryIR, SQLDraft
from services.agent.base import clarification_payload
from services.errors import ServiceError, ServiceUnavailable, describe_exception, friendly_aws_error
from services.watchdog import run_with_timeout

READ_ONLY_ERRORS = ("Only read-only", "non-read-only")


def unique_questions(questions: list[str]) -> list[str]:
    """Quita vacías y duplicadas (sin distinguir mayúsculas, espacios ni puntuación)."""
    seen, result = set(), []
    for question in questions:
        text = str(question or "").strip()
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def load_agent_module(module_path: str):
    """Importa tu módulo; cualquier fallo se traduce a un error presentable."""
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        raise ServiceUnavailable(
            "agent", "No se puede cargar tu agente",
            "Faltan librerías de AWS o LangChain (pip install -r requirements.txt).",
            describe_exception(exc),
        ) from exc
    except Exception as exc:  # p. ej. error al crear el cliente de Bedrock
        raise ServiceUnavailable("agent", "No se puede inicializar tu agente",
                                 "El módulo del agente ha fallado al cargarse.", describe_exception(exc)) from exc


def translate_agent_error(exc: Exception) -> ServiceError:
    if isinstance(exc, ServiceError):
        return exc
    module = type(exc).__module__
    if module.startswith(("botocore", "boto3")):
        return friendly_aws_error("agent", exc, "Amazon Bedrock")
    name = type(exc).__name__
    if name in {"OutputParserException", "ValidationError", "JSONDecodeError"}:
        return ServiceError("agent", "La respuesta del modelo no tiene el formato esperado",
                            "El LLM ha devuelto un JSON que no cumple el esquema pedido.", describe_exception(exc))
    if isinstance(exc, ValueError) and any(m in str(exc) for m in READ_ONLY_ERRORS):
        return ServiceError("agent", "La SQL generada no ha superado la validación de solo lectura",
                            "El agente ha bloqueado una consulta que no era de solo lectura.", describe_exception(exc))
    return ServiceError("agent", "El agente no ha podido completar el paso",
                        "Se ha producido un error inesperado al llamar al LLM.", describe_exception(exc))


CLARIFICATION_PROMPT = """You are a data analyst preparing SQL.

Use the semantic query, retrieved context, and prior answers.
Ask questions only if they are necessary to generate correct SQL:
for example, an ambiguous metric definition, grain, time period,
peer-group definition, or target dialect.

Do not ask for information already present in the context or in prior answers.
Ask ALL the questions you need in this single turn, as a list, at most {max_questions}.
Each question must cover a different point: never repeat or rephrase the same question.
Keep each question concise and in the language of the user's query.
If nothing is needed, return needs_clarification=false and an empty list."""


class LangGraphAgentAdapter:
    name = "Bedrock · tu agente LangGraph"

    def __init__(self, module_path: str, timeout_s: float = 60.0, clarification_mode: str = "node") -> None:
        self.module_path = module_path
        self.timeout_s = timeout_s
        self.clarification_mode = clarification_mode
        self._module = None

    @property
    def module(self):
        if self._module is None:
            self._module = load_agent_module(self.module_path)
        return self._module

    def _call(self, fn, *args):
        from services.agent.capture import capture_llm_calls

        with capture_llm_calls() as capture:
            try:
                result = run_with_timeout(fn, *args, timeout=self.timeout_s, service="agent", what="Amazon Bedrock")
            except Exception as exc:
                raise translate_agent_error(exc) from exc
        return result, capture

    @staticmethod
    def _telemetry(call: AgentCall, capture) -> AgentCall:
        call.prompts = capture.prompts or None
        if capture.has_usage:
            call.input_tokens, call.output_tokens = capture.input_tokens, capture.output_tokens
        return call

    # ------------------------------------------------------------------
    def parse(self, question: str) -> AgentCall[Pydantic_SemanticQueryIR]:
        module = self.module
        result, capture = self._call(module.parse_semantic_query, {"user_query": question})
        ir = result["semantic_ir"]
        if not isinstance(ir, Pydantic_SemanticQueryIR):  # tu módulo puede definir su propia clase
            ir = Pydantic_SemanticQueryIR.model_validate(ir.model_dump())
        return self._telemetry(AgentCall(output=ir), capture)

    def decide_clarifications(self, state: dict[str, Any], max_questions: int) -> AgentCall[ClarificationBatch]:
        module = self.module
        if self.clarification_mode == "node" and hasattr(module, "decide_if_clarification_is_needed"):
            # Tu nodo, tal cual: decide si pregunta y devuelve {"pending_question": ...}.
            result, capture = self._call(module.decide_if_clarification_is_needed, state)
            question = str((result or {}).get("pending_question") or "").strip()
            batch = ClarificationBatch(needs_clarification=bool(question), questions=[question] if question else [])
            return self._telemetry(AgentCall(output=batch), capture)
        payload = json.dumps(clarification_payload(state), ensure_ascii=False, default=str)
        prompt = CLARIFICATION_PROMPT.format(max_questions=max_questions)
        batch, capture = self._call(module.invoke_pydantic, ClarificationBatch, prompt, payload)
        questions = unique_questions(batch.questions)[:max_questions]
        batch = ClarificationBatch(needs_clarification=bool(questions), questions=questions)
        return self._telemetry(AgentCall(output=batch), capture)

    def generate_sql(self, state: dict[str, Any]) -> AgentCall[SQLDraft]:
        module = self.module
        result, capture = self._call(module.generate_sql, state)
        draft = SQLDraft(sql=result["sql"], assumptions=list(result.get("assumptions") or []))
        return self._telemetry(AgentCall(output=draft), capture)
