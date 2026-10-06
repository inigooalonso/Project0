"""Máquina de estados del pipeline, pensada para vivir en st.session_state.

Cada ejecución del script de Streamlit avanza como mucho un paso, así que los
reruns (cambiar de vista, pulsar un paso del stepper…) nunca reinician el
pipeline, y la pausa por aclaraciones es simplemente un estado más.

    RUNNING ──paso ok──▶ RUNNING (automático) / WAITING_NEXT (modo presentador)
       │                        │ «Siguiente paso»
       │◀───────────────────────┘
       ├──aclaración──▶ WAITING_USER ──respuesta──▶ RUNNING (repite el paso 5)
       ├──error──▶ ERROR ──reintentar / continuar con mock──▶ RUNNING
       └──último paso──▶ DONE
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.models import (
    STEP_ORDER,
    AgentCall,
    ClarificationOption,
    ContextBundle,
    ExecutionResult,
    KnowledgeResult,
    RAGResult,
    SQLResult,
    StepId,
    StepStatus,
)

SERVICE_OF_STEP = {
    StepId.PSEUDOCODE: "agent",
    StepId.RAG: "rag",
    StepId.JOINS: "knowledge",
    StepId.CONTEXT: "context",
    StepId.CLARIFY: "agent",
    StepId.SQL: "agent",
    StepId.EXECUTE: "executor",
}


class Phase(str, Enum):
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    WAITING_NEXT = "waiting_next"
    ERROR = "error"
    DONE = "done"


@dataclass
class StepRecord:
    status: StepStatus = StepStatus.PENDING
    elapsed_s: float = 0.0
    calls: int = 0
    simulated: bool | None = None
    source: str = ""
    summary: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    tokens_estimated: bool = False
    prompts: list[dict[str, str]] | None = None

    def add_call(self, call: AgentCall) -> None:
        self.calls += 1
        if call.input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + call.input_tokens
        if call.output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + call.output_tokens
        self.tokens_estimated = self.tokens_estimated or call.tokens_estimated
        if call.prompts:
            self.prompts = call.prompts


@dataclass
class ErrorInfo:
    step: StepId
    service: str
    title: str
    message: str
    detail: str = ""
    can_fallback: bool = True


@dataclass
class ChatMessage:
    role: str  # assistant | user
    content: str


@dataclass
class PipelineRun:
    question: str
    autoplay: bool = True
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started_at: float = field(default_factory=time.time)
    phase: Phase = Phase.RUNNING
    current: StepId = StepId.PSEUDOCODE
    steps: dict[StepId, StepRecord] = field(default_factory=lambda: {s: StepRecord() for s in STEP_ORDER})
    state: dict[str, Any] = field(default_factory=dict)
    rag: RAGResult | None = None
    knowledge: KnowledgeResult | None = None
    context: ContextBundle | None = None
    clarification_options: list[ClarificationOption] = field(default_factory=list)
    chat: list[ChatMessage] = field(default_factory=list)
    sql: SQLResult | None = None
    execution: ExecutionResult | None = None
    explanation: str | None = None
    error: ErrorInfo | None = None
    focus: StepId | None = None
    modes: dict[str, str] = field(default_factory=dict)  # servicios fijados al lanzar la pregunta
    overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Mismas claves que SQLAgentState del agente.
        self.state.setdefault("user_query", self.question)
        self.state.setdefault("clarifications", [])
        self.state.setdefault("pending_question", "")

    # ------------------------------------------------------------------
    @property
    def needs_execution(self) -> bool:
        return self.phase == Phase.RUNNING

    @property
    def last_completed(self) -> StepId | None:
        done = [s for s in STEP_ORDER if self.steps[s].status == StepStatus.DONE]
        return done[-1] if done else None

    @property
    def total_elapsed(self) -> float:
        return sum(r.elapsed_s for r in self.steps.values())

    @property
    def progress(self) -> tuple[int, int]:
        return sum(1 for r in self.steps.values() if r.status == StepStatus.DONE), len(STEP_ORDER)

    def visible_step(self) -> StepId:
        """Paso que se muestra en el escenario principal."""
        if self.focus is not None and self.steps[self.focus].status != StepStatus.PENDING:
            return self.focus
        if self.phase == Phase.DONE:
            return StepId.EXECUTE
        if self.phase in (Phase.WAITING_USER, Phase.ERROR):
            return self.current
        return self.last_completed or self.current

    # ------------------------------------------------------------------
    def start_step(self) -> StepRecord:
        record = self.steps[self.current]
        record.status = StepStatus.RUNNING
        return record

    def complete_step(self) -> None:
        self.steps[self.current].status = StepStatus.DONE
        index = STEP_ORDER.index(self.current)
        if index == len(STEP_ORDER) - 1:
            self.phase = Phase.DONE
            return
        self.current = STEP_ORDER[index + 1]
        self.phase = Phase.RUNNING if self.autoplay else Phase.WAITING_NEXT

    def wait_for_user(self, question: str, options: list[ClarificationOption]) -> None:
        self.state["pending_question"] = question
        self.clarification_options = list(options)
        self.chat.append(ChatMessage("assistant", question))
        self.steps[self.current].status = StepStatus.WAITING
        self.phase = Phase.WAITING_USER
        self.focus = None

    def answer(self, text: str) -> None:
        """Equivalente a Command(resume=answer) en tu grafo: registra la respuesta y repite el paso 5."""
        if self.phase != Phase.WAITING_USER or not text.strip():
            return
        question = self.state.get("pending_question", "")
        self.state["clarifications"] = [*self.state["clarifications"], {"question": question, "answer": text.strip()}]
        self.state["pending_question"] = ""
        self.chat.append(ChatMessage("user", text.strip()))
        self.clarification_options = []
        self.phase = Phase.RUNNING

    def next_step(self) -> None:
        if self.phase == Phase.WAITING_NEXT:
            self.phase = Phase.RUNNING
            self.focus = None

    def fail(self, step: StepId, service: str, title: str, message: str, detail: str, can_fallback: bool) -> None:
        self.steps[step].status = StepStatus.ERROR
        self.error = ErrorInfo(step, service, title, message, detail, can_fallback)
        self.phase = Phase.ERROR
        self.focus = None

    def retry(self) -> None:
        if self.phase != Phase.ERROR:
            return
        self.steps[self.current].status = StepStatus.PENDING
        self.error = None
        self.phase = Phase.RUNNING

    def fallback_to_mock(self, service: str) -> None:
        """«Continuar con datos simulados»: ese servicio pasa a mock solo en esta ejecución."""
        self.overrides[service] = "mock"
        self.retry()
