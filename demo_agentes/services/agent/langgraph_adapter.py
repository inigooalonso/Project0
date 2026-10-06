"""Adaptador REAL: llama uno a uno a los nodos de tu agente LangGraph.

No se invoca sql_agent_graph de una vez para poder intercalar los pasos que no
están en el grafo (joins y glosario, contexto, ejecución), medir cada paso y
pausar en st.session_state. La pausa de ask_user (interrupt) la reproduce la
máquina de estados con el mismo contrato: clarifications = [{question, answer}].
"""
from __future__ import annotations

import importlib
from typing import Any

from core.models import AgentCall, ClarificationDecision, ClarificationOption, Pydantic_SemanticQueryIR, SQLDraft
from services.errors import ServiceError, ServiceUnavailable, describe_exception, friendly_aws_error
from services.watchdog import run_with_timeout

READ_ONLY_ERRORS = ("Only read-only", "non-read-only")


def load_agent_module(module_path: str):
    """Importa tu módulo; cualquier fallo se traduce a un error presentable."""
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        raise ServiceUnavailable(
            "agent", "No se puede cargar tu agente",
            "Faltan las librerías del modo real (pip install -r requirements-aws.txt).",
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


class LangGraphAgentAdapter:
    name = "Bedrock · tu agente LangGraph"
    simulated = False

    def __init__(self, module_path: str, timeout_s: float = 60.0) -> None:
        self.module_path = module_path
        self.timeout_s = timeout_s
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

    def decide_clarification(self, state: dict[str, Any]) -> AgentCall[ClarificationDecision]:
        module = self.module
        result, capture = self._call(module.decide_if_clarification_is_needed, state)
        question = result.get("pending_question") or None
        decision = ClarificationDecision(needs_clarification=bool(question), question=question)
        return self._telemetry(AgentCall(output=decision), capture)

    def generate_sql(self, state: dict[str, Any]) -> AgentCall[SQLDraft]:
        module = self.module
        result, capture = self._call(module.generate_sql, state)
        draft = SQLDraft(sql=result["sql"], assumptions=list(result.get("assumptions") or []))
        return self._telemetry(AgentCall(output=draft), capture)

    def clarification_options(self, state: dict[str, Any]) -> list[ClarificationOption]:
        return []

    def explanation(self, state: dict[str, Any]) -> str | None:
        return None
