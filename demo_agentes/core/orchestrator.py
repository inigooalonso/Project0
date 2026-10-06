"""Ejecuta los pasos del pipeline contra los servicios configurados.

Cada llamada a ``run_current_step`` ejecuta un único paso: mide su tiempo,
guarda sus artefactos en la ejecución (PipelineRun) y, si un servicio falla,
deja la ejecución en estado de error con un mensaje presentable en lugar de
propagar la excepción.
"""
from __future__ import annotations

import hashlib
import time
from typing import Callable

from core import narrative
from core.context import assemble_context
from core.models import SQLResult, StepId, StepStatus
from core.pipeline import SERVICE_OF_STEP, Phase, PipelineRun, StepRecord
from core.settings import Settings
from services.errors import ServiceError, describe_exception
from services.factory import Services
from services.sql_guard import inspect_sql

# Latencia simulada de referencia por paso (segundos), solo para servicios simulados.
BASE_LATENCY = {
    StepId.PSEUDOCODE: 2.3,
    StepId.RAG: 2.0,
    StepId.JOINS: 1.1,
    StepId.CONTEXT: 0.8,
    StepId.CLARIFY: 1.5,
    StepId.SQL: 2.6,
    StepId.EXECUTE: 1.3,
}

ProgressCallback = Callable[[list[str]], None]


def simulated_latency(step: StepId, salt: str) -> float:
    """Latencia determinista con ±12 % de variación según la pregunta."""
    h = hashlib.sha1(f"{step.value}|{salt}".encode("utf-8")).digest()[0] / 255.0
    return BASE_LATENCY[step] * (0.88 + 0.24 * h)


class Orchestrator:
    def __init__(self, services: Services, settings: Settings, sleep: Callable[[float], None] = time.sleep) -> None:
        self.services = services
        self.settings = settings
        self._sleep = sleep

    def _pause(self, seconds: float) -> None:
        if seconds > 0 and self.settings.speed > 0:
            self._sleep(seconds * self.settings.speed)

    def _pace(self, step: StepId, run: PipelineRun, simulated: bool, already: float = 0.0) -> None:
        if simulated:
            self._pause(max(0.0, simulated_latency(step, run.question) - already / max(self.settings.speed, 1e-9)))

    # ------------------------------------------------------------------
    def run_current_step(self, run: PipelineRun, progress: ProgressCallback | None = None) -> None:
        if run.phase != Phase.RUNNING:
            return
        step = run.current
        record = run.start_step()
        started = time.perf_counter()
        try:
            handler = {
                StepId.PSEUDOCODE: self._pseudocode,
                StepId.RAG: self._rag,
                StepId.JOINS: self._joins,
                StepId.CONTEXT: self._context,
                StepId.CLARIFY: self._clarify,
                StepId.SQL: self._sql,
                StepId.EXECUTE: self._execute,
            }[step]
            handler(run, record, progress)
        except ServiceError as exc:
            record.elapsed_s += time.perf_counter() - started
            run.fail(step, exc.service, exc.title, exc.message, exc.detail, exc.can_fallback)
            return
        except Exception as exc:  # nunca debe llegar una traza a la pantalla
            record.elapsed_s += time.perf_counter() - started
            run.fail(step, SERVICE_OF_STEP[step], "Algo no ha ido como se esperaba",
                     "El paso no se ha podido completar.", describe_exception(exc), can_fallback=True)
            return
        record.elapsed_s += time.perf_counter() - started
        if record.status == StepStatus.RUNNING:
            run.complete_step()

    # ------------------------------------------------------------------
    def _pseudocode(self, run: PipelineRun, record: StepRecord, progress) -> None:
        agent = self.services.agent
        call = agent.parse(run.question)
        self._pace(StepId.PSEUDOCODE, run, agent.simulated)
        run.state["semantic_ir"] = call.output
        record.add_call(call)
        record.source, record.simulated = agent.name, agent.simulated
        record.summary = narrative.summary_pseudocode(call.output)

    def _rag(self, run: PipelineRun, record: StepRecord, progress) -> None:
        rag = self.services.rag
        result = rag.search(run.state["semantic_ir"])
        if rag.simulated:
            self._replay_search(run, result, progress)
        run.rag = result
        record.source, record.simulated = rag.name, rag.simulated
        record.summary = narrative.summary_rag(result)

    def _replay_search(self, run: PipelineRun, result, progress) -> None:
        """Reproduce la búsqueda nivel a nivel para que la animación tenga ritmo."""
        total = simulated_latency(StepId.RAG, run.question)
        steps = max(1, len(result.searches) * 3)
        lines: list[str] = []
        for search in result.searches:
            table = (search.table or "").split(".")[-1]
            for level, text in (
                ("propietario", f"{search.owner} ({search.owner_score:.2f})".replace(".", ",")),
                ("tabla", f"{table} ({search.table_score:.2f})".replace(".", ",")),
                ("campo", f"{search.field} ({search.field_score:.2f})".replace(".", ",")),
            ):
                current = f"«{search.surface_form}» → {level}: {text}"
                if progress:
                    progress(lines + [current])
                self._pause(total / steps)
            lines.append(f"«{search.surface_form}» → {search.owner} › {table} › {search.field}")
        if progress:
            progress(lines)

    def _joins(self, run: PipelineRun, record: StepRecord, progress) -> None:
        knowledge = self.services.knowledge.resolve(run.state["semantic_ir"], run.rag)
        # Paso de maqueta: solo se le da ritmo cuando el resto del pipeline es simulado.
        simulated = self.services.rag.simulated
        self._pace(StepId.JOINS, run, simulated)
        run.knowledge = knowledge
        record.source, record.simulated = knowledge.source, simulated
        record.summary = narrative.summary_joins(knowledge)

    def _context(self, run: PipelineRun, record: StepRecord, progress) -> None:
        bundle = assemble_context(run.rag, run.knowledge, self.services.catalog, self.settings.dialect)
        simulated = self.services.rag.simulated
        self._pace(StepId.CONTEXT, run, simulated)
        run.context = bundle
        run.state["context"] = bundle.context
        record.source, record.simulated = "Ensamblado local", simulated
        record.summary = narrative.summary_context(bundle)

    def _clarify(self, run: PipelineRun, record: StepRecord, progress) -> None:
        agent = self.services.agent
        answered = len(run.state["clarifications"])
        record.source, record.simulated = agent.name, agent.simulated
        question = None
        if answered < self.settings.max_clarifications:  # misma guarda que tu nodo
            call = agent.decide_clarification(run.state)
            self._pace(StepId.CLARIFY, run, agent.simulated)
            record.add_call(call)
            question = call.output.question if call.output.question else None
        if question:
            run.wait_for_user(question, agent.clarification_options(run.state))
            record.summary = narrative.summary_clarify(answered, waiting=True)
            return
        run.state["pending_question"] = ""
        record.summary = narrative.summary_clarify(answered, waiting=False)

    def _sql(self, run: PipelineRun, record: StepRecord, progress) -> None:
        agent = self.services.agent
        call = agent.generate_sql(run.state)
        self._pace(StepId.SQL, run, agent.simulated)
        record.add_call(call)
        record.source, record.simulated = agent.name, agent.simulated
        draft = call.output
        authorized = run.knowledge.tables if run.knowledge else []
        inspection = inspect_sql(draft.sql, authorized, self.settings.sqlglot_dialect)
        if not inspection.parsed:
            raise ServiceError("agent", "La SQL generada no se puede interpretar",
                               "La consulta no es SQL válida para Athena.", inspection.parse_error or "")
        if not (inspection.read_only and inspection.single_statement):
            raise ServiceError("agent", "La SQL generada no es de solo lectura",
                               "El agente solo puede ejecutar una consulta SELECT.", draft.sql[:300])
        if inspection.unauthorized_tables:
            raise ServiceError("agent", "La SQL usa tablas no autorizadas",
                               "La consulta referencia tablas que no estaban en el contexto.",
                               ", ".join(inspection.unauthorized_tables))
        run.state["sql"] = draft.sql
        run.state["assumptions"] = list(draft.assumptions)
        run.sql = SQLResult(sql=draft.sql, assumptions=list(draft.assumptions), inspection=inspection)
        run.explanation = agent.explanation(run.state) or narrative.explanation_from_ir(
            run.state["semantic_ir"], draft.assumptions
        )
        record.summary = narrative.summary_sql(run.sql)

    def _execute(self, run: PipelineRun, record: StepRecord, progress) -> None:
        executor = self.services.executor
        result = executor.execute(run.state["sql"])
        self._pace(StepId.EXECUTE, run, executor.simulated, already=result.elapsed_s)
        run.execution = result
        record.source, record.simulated = executor.name, executor.simulated
        record.summary = narrative.summary_execute(result)
