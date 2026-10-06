"""Máquina de estados del pipeline, pensada para vivir en st.session_state.

Cada ejecución del script de Streamlit avanza como mucho un paso, así que los
reruns (cambiar de vista, pulsar un paso del stepper…) nunca reinician el
pipeline, y la pausa por aclaraciones es simplemente un estado más.

    RUNNING ──paso ok──▶ RUNNING (automático) / WAITING_NEXT (modo presentador)
       │                        │ «Siguiente paso»
       │◀───────────────────────┘
       ├──aclaraciones──▶ WAITING_USER ──respuestas──▶ RUNNING (repite el paso 5)
       ├──error──▶ ERROR ──reintentar──▶ RUNNING
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


NO_ANSWER = "Sin respuesta: usa el supuesto más razonable y decláralo."


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
    source: str = ""
    summary: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    prompts: list[dict[str, str]] | None = None

    def add_call(self, call: AgentCall) -> None:
        self.calls += 1
        if call.input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + call.input_tokens
        if call.output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + call.output_tokens
        if call.prompts:
            self.prompts = call.prompts


@dataclass
class ErrorInfo:
    step: StepId
    service: str
    title: str
    message: str
    detail: str = ""


@dataclass
class ClarificationRound:
    """Una ronda de aclaraciones: varias preguntas a la vez y sus respuestas."""

    questions: list[str]
    answers: list[str] | None = None


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
    rounds: list[ClarificationRound] = field(default_factory=list)
    sql: SQLResult | None = None
    execution: ExecutionResult | None = None
    explanation: str | None = None
    error: ErrorInfo | None = None
    focus: StepId | None = None

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

    @property
    def pending_questions(self) -> list[str]:
        if self.phase == Phase.WAITING_USER and self.rounds and self.rounds[-1].answers is None:
            return list(self.rounds[-1].questions)
        return []

    def wait_for_user(self, questions: list[str]) -> None:
        self.state["pending_question"] = "\n\n".join(questions)
        self.rounds.append(ClarificationRound(questions=list(questions)))
        self.steps[self.current].status = StepStatus.WAITING
        self.phase = Phase.WAITING_USER
        self.focus = None

    def answer(self, answers: list[str]) -> bool:
        """Equivale a Command(resume=...) en tu grafo, para todas las preguntas de la ronda a la vez.

        Registra un {question, answer} por pregunta (mismo contrato que ask_user) y repite el paso 5.
        Hace falta al menos una respuesta; las que queden vacías se envían como «Sin respuesta».
        """
        questions = self.pending_questions
        cleaned = [(a or "").strip() for a in answers][: len(questions)]
        cleaned += [""] * (len(questions) - len(cleaned))
        if not questions or not any(cleaned):
            return False
        cleaned = [a or NO_ANSWER for a in cleaned]
        self.rounds[-1].answers = cleaned
        self.state["clarifications"] = [
            *self.state["clarifications"], *({"question": q, "answer": a} for q, a in zip(questions, cleaned))
        ]
        self.state["pending_question"] = ""
        self.phase = Phase.RUNNING
        return True

    def next_step(self) -> None:
        if self.phase == Phase.WAITING_NEXT:
            self.phase = Phase.RUNNING
            self.focus = None

    def fail(self, step: StepId, service: str, title: str, message: str, detail: str) -> None:
        self.steps[step].status = StepStatus.ERROR
        self.error = ErrorInfo(step, service, title, message, detail)
        self.phase = Phase.ERROR
        self.focus = None

    def retry(self) -> None:
        if self.phase != Phase.ERROR:
            return
        self.steps[self.current].status = StepStatus.PENDING
        self.error = None
        self.phase = Phase.RUNNING
