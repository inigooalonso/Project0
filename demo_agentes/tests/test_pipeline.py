"""Máquina de estados: recorrido completo, aclaraciones en lote, modo presentador y errores."""
import pytest

from core.models import STEP_ORDER, StepId, StepStatus
from core.orchestrator import Orchestrator
from core.pipeline import NO_ANSWER, Phase, PipelineRun
from services.errors import ServiceUnavailable
from tests.fakes import QUESTION, QUESTIONS, FakeAgent, fake_services


def drive(orchestrator, run, answers=(), limit=30):
    answers = list(answers)
    for _ in range(limit):
        if run.phase == Phase.WAITING_USER:
            assert run.answer(answers.pop(0))
        elif run.phase == Phase.WAITING_NEXT:
            run.next_step()
        elif run.phase in (Phase.DONE, Phase.ERROR):
            return run
        orchestrator.run_current_step(run)
    raise AssertionError("el pipeline no termina")


def test_question_without_doubts_runs_all_steps(settings):
    services = fake_services(FakeAgent(questions=[]))
    run = drive(Orchestrator(services, settings), PipelineRun(QUESTION))
    assert run.phase == Phase.DONE, run.error
    assert all(run.steps[s].status == StepStatus.DONE for s in STEP_ORDER)
    assert run.state["clarifications"] == [] and run.rounds == []
    assert run.execution.df.iloc[0]["mesa"] == "Rates"
    assert run.steps[StepId.PSEUDOCODE].input_tokens == 1200


def test_all_questions_of_a_round_are_asked_and_answered_at_once(settings):
    orchestrator = Orchestrator(fake_services(), settings)
    run = PipelineRun(QUESTION)
    for _ in range(10):
        orchestrator.run_current_step(run)
        if run.phase == Phase.WAITING_USER:
            break
    assert run.current == StepId.CLARIFY and run.steps[StepId.CLARIFY].status == StepStatus.WAITING
    assert run.pending_questions == QUESTIONS
    assert run.state["pending_question"] == "\n\n".join(QUESTIONS)
    orchestrator.run_current_step(run)  # sin respuestas no avanza
    assert run.phase == Phase.WAITING_USER
    assert not run.answer(["", "  "])  # hace falta al menos una respuesta

    assert run.answer(["Margen (gf_franch_oper_rslt_amount)", ""])
    drive(orchestrator, run)
    assert run.phase == Phase.DONE
    assert run.state["clarifications"] == [
        {"question": QUESTIONS[0], "answer": "Margen (gf_franch_oper_rslt_amount)"},
        {"question": QUESTIONS[1], "answer": NO_ANSWER},
    ]
    assert run.steps[StepId.CLARIFY].calls == 2  # pregunta y, tras responder, vuelve a decidir


def test_clarification_rounds_are_limited(settings):
    agent = FakeAgent(rounds_with_questions=99)
    run = drive(Orchestrator(fake_services(agent), settings.with_overrides(max_clarification_rounds=2)),
                PipelineRun(QUESTION), answers=["a", "b"])
    assert run.phase == Phase.DONE and len(run.rounds) == 2 and agent.decisions == 2


def test_questions_per_round_are_capped(settings):
    agent = FakeAgent(questions=[f"¿Pregunta {i}?" for i in range(8)])
    orchestrator = Orchestrator(fake_services(agent), settings.with_overrides(max_questions_per_round=3))
    run = PipelineRun(QUESTION)
    for _ in range(10):
        orchestrator.run_current_step(run)
    assert len(run.pending_questions) == 3


def test_presenter_mode_waits_for_next_step(settings):
    run = PipelineRun(QUESTION, autoplay=False)
    orchestrator = Orchestrator(fake_services(), settings)
    orchestrator.run_current_step(run)
    assert run.phase == Phase.WAITING_NEXT and run.current == StepId.RAG
    orchestrator.run_current_step(run)  # no hace nada hasta «Siguiente paso»
    assert run.steps[StepId.RAG].status == StepStatus.PENDING
    run.next_step()
    orchestrator.run_current_step(run)
    assert run.steps[StepId.RAG].status == StepStatus.DONE


class BrokenExecutor:
    name = "Amazon Athena"

    def execute(self, sql):
        raise ServiceUnavailable("executor", "Sin credenciales para Amazon Athena", "Sin credenciales.", "NoCredentialsError")


def test_service_failure_stops_at_its_step_and_can_be_retried(settings):
    agent = FakeAgent(questions=[])
    run = drive(Orchestrator(fake_services(agent, executor=BrokenExecutor()), settings), PipelineRun(QUESTION))
    assert run.phase == Phase.ERROR and run.current == StepId.EXECUTE
    assert run.error.service == "executor" and run.steps[StepId.EXECUTE].status == StepStatus.ERROR
    run.retry()
    drive(Orchestrator(fake_services(agent), settings), run)  # solo se repite el paso 7
    assert run.phase == Phase.DONE


def test_unexpected_exceptions_never_escape(settings):
    class Exploding(FakeAgent):
        def parse(self, question):
            raise RuntimeError("boom")

    run = PipelineRun(QUESTION)
    Orchestrator(fake_services(Exploding()), settings).run_current_step(run)
    assert run.phase == Phase.ERROR and "RuntimeError" in run.error.detail


@pytest.mark.parametrize("sql,fragment", [
    ("SELECT * FROM ho_master.t_pcli_customers;", "t_pcli_customers"),
    ("DELETE FROM ho_master.t_o1dm_franchise_gm_daily;", "solo lectura"),
])
def test_unsafe_sql_is_blocked_before_execution(sql, fragment, settings):
    run = drive(Orchestrator(fake_services(FakeAgent(questions=[], sql=sql)), settings), PipelineRun(QUESTION))
    assert run.phase == Phase.ERROR and run.current == StepId.SQL
    assert fragment in (run.error.detail + run.error.title)
