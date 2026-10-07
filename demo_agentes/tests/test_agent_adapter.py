"""Tu código a través de los adaptadores.

El LLM de Bedrock se sustituye por un modelo falso de LangChain que devuelve
JSON fijo, de modo que se ejecuta TU código (prompts, PydanticOutputParser,
invoke_pydantic, validate_read_only_sql, retrieve_context_for_sql) sin red
ni credenciales.
"""
import json

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_aws")

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from core.orchestrator import Orchestrator  # noqa: E402
from core.pipeline import Phase, PipelineRun  # noqa: E402
from services.agent.langgraph_adapter import LangGraphAgentAdapter, translate_agent_error, unique_questions  # noqa: E402
from services.errors import ServiceUnavailable  # noqa: E402
from services.rag.agent_adapter import AgentRAGAdapter, parse_table_block  # noqa: E402
from tests.fakes import CONTEXT, QUESTION, QUESTIONS, SAMPLE_IR, FakeExecutor, fake_services  # noqa: E402

MODULE = "agents.ada_text2sql.agent"
SQL = ("SELECT gf_crm_group_id, SUM(gf_franch_oper_rslt_amount) AS negocio "
       "FROM ho_master.t_o1dm_franchise_gm_daily GROUP BY 1 ORDER BY 2 DESC LIMIT 1")


@pytest.fixture
def agent_module(monkeypatch):
    import importlib

    module = importlib.import_module(MODULE)
    responses = [
        json.dumps(SAMPLE_IR, ensure_ascii=False),
        json.dumps({"needs_clarification": True, "questions": [*QUESTIONS, QUESTIONS[0].upper()]}, ensure_ascii=False),
        json.dumps({"needs_clarification": False, "questions": []}),
        json.dumps({"sql": SQL, "assumptions": ["supuesto de prueba"]}, ensure_ascii=False),
    ]
    monkeypatch.setattr(module, "llm", FakeListChatModel(responses=responses))
    return module


def test_adapter_runs_the_team_nodes_end_to_end(agent_module, settings):
    from dataclasses import replace

    services = replace(fake_services(), agent=LangGraphAgentAdapter(MODULE), rag=AgentRAGAdapter(MODULE, settings.dialect),
                       executor=FakeExecutor())
    orchestrator = Orchestrator(services, settings)
    run = PipelineRun(QUESTION)
    for _ in range(20):
        if run.phase == Phase.WAITING_USER:
            # Varias preguntas a la vez; la repetida (mismo texto en mayúsculas) se descarta.
            assert run.pending_questions == QUESTIONS
            run.answer(["Margen", "Campo de franquicia"])
        if run.phase in (Phase.DONE, Phase.ERROR):
            break
        orchestrator.run_current_step(run)
    assert run.phase == Phase.DONE, run.error
    assert run.state["sql"].endswith(";")  # lo añade tu validate_read_only_sql
    assert run.state["assumptions"] == ["supuesto de prueba"]
    assert [c["answer"] for c in run.state["clarifications"]] == ["Margen", "Campo de franquicia"]
    prompts = run.steps[list(run.steps)[0]].prompts
    assert prompts and any("JSON" in p["content"] for p in prompts)


def test_rag_adapter_reads_your_stub_and_forces_athena(agent_module):
    from core.models import Pydantic_SemanticQueryIR

    result = AgentRAGAdapter(MODULE, "AWS Athena (Trino SQL)").search(Pydantic_SemanticQueryIR.model_validate(SAMPLE_IR))
    assert result.dialect == "AWS Athena (Trino SQL)"
    assert result.raw_context["original_dialect"] == "AWS Athena"
    assert result.selected_tables == ["ho_master.t_o1dm_franchise_gm_daily", "ho_master.t_o1dm_franchise_gm_monthly"]
    # Los campos llegan en schema_context; table_fields2 repite 3 campos de la tabla diaria.
    assert [len(t.fields) for t in result.tables] == [90, 0] and result.duplicated_fields == 3
    assert result.candidates == []


def test_rag_adapter_reads_candidate_tables(agent_module, monkeypatch):
    from core.models import Pydantic_SemanticQueryIR

    monkeypatch.setattr(agent_module, "retrieve_context_for_sql", lambda ir: CONTEXT)
    result = AgentRAGAdapter(MODULE, "AWS Athena (Trino SQL)").search(Pydantic_SemanticQueryIR.model_validate(SAMPLE_IR))
    assert len(result.candidates) == 3 and result.unified_derived


def test_duplicate_questions_are_removed():
    assert unique_questions(["¿A?", " ¿a? ", "", "¿B?"]) == ["¿A?", "¿B?"]


def test_table_block_parser():
    block = parse_table_block("ho_master.t_abcd_x:\n-Description:desc\n-Fields:f1: etiqueta uno, descripción\n* f2: dos, d2")
    assert block["name"] == "ho_master.t_abcd_x"
    assert [f["name"] for f in block["fields"]] == ["f1", "f2"]


def test_aws_errors_become_presentable_messages():
    from botocore.exceptions import NoCredentialsError

    error = translate_agent_error(NoCredentialsError())
    assert isinstance(error, ServiceUnavailable)
    assert "credenciales" in error.message.lower()


def test_slow_service_times_out_with_a_presentable_error():
    import time

    from services.errors import ServiceError
    from services.watchdog import run_with_timeout

    with pytest.raises(ServiceError) as info:
        run_with_timeout(time.sleep, 2, timeout=0.2, service="agent", what="Amazon Bedrock")
    assert "tardando" in info.value.title
