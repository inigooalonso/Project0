"""Máquina de estados: recorrido completo, pausa por aclaración, modo presentador y fallback."""
from dataclasses import replace

import pytest

from core.models import STEP_ORDER, ExecutionResult, StepId, StepStatus
from core.orchestrator import Orchestrator
from core.pipeline import Phase, PipelineRun
from core.result_profile import build_kpis, headline, profile_result
from services.errors import ServiceUnavailable
from services.factory import build_services


def drive(orchestrator, run, answers=(), limit=30):
    answers = list(answers)
    for _ in range(limit):
        if run.phase == Phase.WAITING_USER:
            run.answer(answers.pop(0))
        elif run.phase == Phase.WAITING_NEXT:
            run.next_step()
        elif run.phase in (Phase.DONE, Phase.ERROR):
            return run
        orchestrator.run_current_step(run)
    raise AssertionError("el pipeline no termina")


@pytest.mark.parametrize("scenario_id,chart", [
    ("hipotecas_oficina", "bar"), ("saldo_vista", "line"), ("gasto_tarjeta", "bar"),
])
def test_scenarios_run_end_to_end(scenario_id, chart, scenarios, services, settings):
    scenario = next(s for s in scenarios if s.id == scenario_id)
    run = drive(Orchestrator(services, settings), PipelineRun(scenario.question))
    assert run.phase == Phase.DONE, run.error
    assert all(run.steps[s].status == StepStatus.DONE for s in STEP_ORDER)
    assert run.state["clarifications"] == []
    profile = profile_result(run.execution.df, run.sql.inspection.order_by)
    assert profile.kind == chart
    assert len(build_kpis(run.execution.df, profile)) == 3
    assert headline(run.execution.df, profile)


def test_clarification_pauses_and_resumes_from_step_5(scenarios, services, settings):
    scenario = next(s for s in scenarios if s.id == "morosidad_oficinas")
    orchestrator = Orchestrator(services, settings)
    run = PipelineRun(scenario.question)
    for _ in range(10):
        orchestrator.run_current_step(run)
        if run.phase == Phase.WAITING_USER:
            break
    assert run.current == StepId.CLARIFY and run.steps[StepId.CLARIFY].status == StepStatus.WAITING
    assert [o.label for o in run.clarification_options] == ["Tasa de mora (> 90 días)", "Ratio de impagados"]
    orchestrator.run_current_step(run)  # sin respuesta no avanza
    assert run.phase == Phase.WAITING_USER

    run.answer("Ratio de impagados")
    drive(orchestrator, run)
    assert run.phase == Phase.DONE
    assert run.state["clarifications"] == [{"question": scenario.clarification_question, "answer": "Ratio de impagados"}]
    assert "ratio_de_impagados_pct" in run.state["sql"]
    assert run.execution.df.iloc[0]["oficina"] == "Sevilla · Triana"


def test_presenter_mode_waits_for_next_step(scenarios, services, settings):
    run = PipelineRun(scenarios[0].question, autoplay=False)
    orchestrator = Orchestrator(services, settings)
    orchestrator.run_current_step(run)
    assert run.phase == Phase.WAITING_NEXT and run.current == StepId.RAG
    orchestrator.run_current_step(run)  # no hace nada hasta «Siguiente paso»
    assert run.steps[StepId.RAG].status == StepStatus.PENDING
    run.next_step()
    orchestrator.run_current_step(run)
    assert run.steps[StepId.RAG].status == StepStatus.DONE


class BrokenExecutor:
    name = "Amazon Athena"
    simulated = False

    def execute(self, sql):
        raise ServiceUnavailable("executor", "Sin credenciales para Amazon Athena", "Sin credenciales.", "NoCredentialsError")


def test_real_service_failure_offers_fallback_to_mock(scenarios, settings):
    services = build_services(settings)
    broken = replace(services, executor=BrokenExecutor())
    run = drive(Orchestrator(broken, settings), PipelineRun(scenarios[0].question))
    assert run.phase == Phase.ERROR and run.current == StepId.EXECUTE
    assert run.error.service == "executor" and run.error.can_fallback
    assert run.steps[StepId.EXECUTE].status == StepStatus.ERROR

    run.fallback_to_mock("executor")
    assert run.overrides == {"executor": "mock"} and run.phase == Phase.RUNNING
    drive(Orchestrator(services, settings), run)  # mismos pasos ya hechos; solo se repite el 7
    assert run.phase == Phase.DONE and isinstance(run.execution, ExecutionResult)


def test_unexpected_exceptions_never_escape(scenarios, settings):
    class Exploding:
        name = "LLM"
        simulated = True

        def parse(self, question):
            raise RuntimeError("boom")

    services = replace(build_services(settings), agent=Exploding())
    run = PipelineRun(scenarios[0].question)
    Orchestrator(services, settings).run_current_step(run)
    assert run.phase == Phase.ERROR and "RuntimeError" in run.error.detail


def test_free_question_in_full_mock_mode_is_rejected_gracefully(services, settings):
    run = PipelineRun("¿Cuántos clientes tenemos en Bilbao?")
    Orchestrator(services, settings).run_current_step(run)
    assert run.phase == Phase.ERROR and not run.error.can_fallback
    assert isinstance(run.error.title, str) and run.error.service == "agent"


def test_unauthorized_sql_is_blocked_before_execution(scenarios, settings):
    class SneakyAgent:
        def __init__(self, inner):
            self.inner = inner
            self.name, self.simulated = inner.name, inner.simulated

        def __getattr__(self, item):
            return getattr(self.inner, item)

        def generate_sql(self, state):
            call = self.inner.generate_sql(state)
            call.output.sql = "SELECT * FROM ho_master.t_pcli_customers;"
            return call

    services = build_services(settings)
    run = drive(Orchestrator(replace(services, agent=SneakyAgent(services.agent)), settings), PipelineRun(scenarios[0].question))
    assert run.phase == Phase.ERROR and run.current == StepId.SQL
    assert "t_pcli_customers" in run.error.detail
