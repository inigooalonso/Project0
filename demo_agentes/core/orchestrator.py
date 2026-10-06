"""Ejecuta los pasos del pipeline contra los servicios configurados.

Cada llamada a ``run_current_step`` ejecuta un único paso: mide su tiempo,
guarda sus artefactos en la ejecución (PipelineRun) y, si un servicio falla,
deja la ejecución en estado de error con un mensaje presentable en lugar de
propagar la excepción.
"""
from __future__ import annotations

import time

from core import narrative
from core.context import assemble_context
from core.models import SQLResult, StepId, StepStatus
from core.pipeline import SERVICE_OF_STEP, Phase, PipelineRun, StepRecord
from core.settings import Settings
from services.errors import ServiceError, describe_exception
from services.factory import Services
from services.sql_guard import inspect_sql


class Orchestrator:
    def __init__(self, services: Services, settings: Settings) -> None:
        self.services = services
        self.settings = settings

    # ------------------------------------------------------------------
    def run_current_step(self, run: PipelineRun) -> None:
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
            handler(run, record)
        except ServiceError as exc:
            record.elapsed_s += time.perf_counter() - started
            run.fail(step, exc.service, exc.title, exc.message, exc.detail)
            return
        except Exception as exc:  # nunca debe llegar una traza a la pantalla
            record.elapsed_s += time.perf_counter() - started
            run.fail(step, SERVICE_OF_STEP[step], "Algo no ha ido como se esperaba",
                     "El paso no se ha podido completar.", describe_exception(exc))
            return
        record.elapsed_s += time.perf_counter() - started
        if record.status == StepStatus.RUNNING:
            run.complete_step()

    # ------------------------------------------------------------------
    def _pseudocode(self, run: PipelineRun, record: StepRecord) -> None:
        agent = self.services.agent
        call = agent.parse(run.question)
        run.state["semantic_ir"] = call.output
        record.add_call(call)
        record.source = agent.name
        record.summary = narrative.summary_pseudocode(call.output)

    def _rag(self, run: PipelineRun, record: StepRecord) -> None:
        rag = self.services.rag
        run.rag = rag.search(run.state["semantic_ir"])
        record.source = rag.name
        record.summary = narrative.summary_rag(run.rag)

    def _joins(self, run: PipelineRun, record: StepRecord) -> None:
        knowledge = self.services.knowledge
        run.knowledge = knowledge.resolve(run.state["semantic_ir"], run.rag)
        record.source = run.knowledge.source
        record.summary = narrative.summary_joins(run.knowledge)

    def _context(self, run: PipelineRun, record: StepRecord) -> None:
        bundle = assemble_context(run.rag, run.knowledge, self.settings.dialect)
        run.context = bundle
        run.state["context"] = bundle.context
        record.source = "Ensamblado local"
        record.summary = narrative.summary_context(bundle)

    def _clarify(self, run: PipelineRun, record: StepRecord) -> None:
        agent = self.services.agent
        answered_rounds = sum(1 for r in run.rounds if r.answers is not None)
        record.source = agent.name
        questions: list[str] = []
        if answered_rounds < self.settings.max_clarification_rounds:
            call = agent.decide_clarifications(run.state, self.settings.max_questions_per_round)
            record.add_call(call)
            questions = list(call.output.questions)
        if questions:
            run.wait_for_user(questions)
            record.summary = narrative.summary_clarify(run, waiting=True)
            return
        run.state["pending_question"] = ""
        record.summary = narrative.summary_clarify(run, waiting=False)

    def _sql(self, run: PipelineRun, record: StepRecord) -> None:
        agent = self.services.agent
        call = agent.generate_sql(run.state)
        record.add_call(call)
        record.source = agent.name
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
        record.summary = narrative.summary_sql(run.sql)

    def _execute(self, run: PipelineRun, record: StepRecord) -> None:
        executor = self.services.executor
        run.execution = executor.execute(run.state["sql"])
        record.source = executor.name
        record.summary = narrative.summary_execute(run.execution)
