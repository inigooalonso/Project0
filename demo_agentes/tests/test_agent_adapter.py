"""Modo real con tu código: los nodos de LangGraph se llaman a través del adaptador.

El LLM de Bedrock se sustituye por un modelo falso de LangChain que devuelve
JSON fijo, de modo que se ejecuta TU código (prompts, PydanticOutputParser,
validate_read_only_sql, retrieve_context_for_sql) sin red ni credenciales.
"""
import json
from dataclasses import replace

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_aws")

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from core.orchestrator import Orchestrator  # noqa: E402
from core.pipeline import Phase, PipelineRun  # noqa: E402
from services.agent.langgraph_adapter import LangGraphAgentAdapter, translate_agent_error  # noqa: E402
from services.errors import ServiceUnavailable  # noqa: E402
from services.rag.agent_adapter import AgentRAGAdapter, parse_table_block  # noqa: E402

MODULE = "agents.ada_text2sql.agent"


@pytest.fixture
def agent_module(monkeypatch, scenarios):
    import importlib

    module = importlib.import_module(MODULE)
    scenario = next(s for s in scenarios if s.id == "hipotecas_oficina")
    responses = [
        json.dumps(scenario.semantic_ir, default=str, ensure_ascii=False),
        json.dumps({"needs_clarification": False, "question": None}),
        json.dumps({"sql": scenario.sql.strip(), "assumptions": ["supuesto de prueba"]}, ensure_ascii=False),
    ]
    monkeypatch.setattr(module, "llm", FakeListChatModel(responses=responses))
    return module


def test_real_adapter_runs_the_team_nodes_end_to_end(agent_module, scenarios, services, settings):
    scenario = next(s for s in scenarios if s.id == "hipotecas_oficina")
    real = replace(services, agent=LangGraphAgentAdapter(MODULE))
    run = PipelineRun(scenario.question)
    orchestrator = Orchestrator(real, settings.with_overrides(agent_mode="bedrock"))
    for _ in range(20):
        if run.phase in (Phase.DONE, Phase.ERROR):
            break
        orchestrator.run_current_step(run)
    assert run.phase == Phase.DONE, run.error
    assert run.state["sql"].endswith(";")  # lo añade tu validate_read_only_sql
    assert run.state["assumptions"] == ["supuesto de prueba"]
    prompts = run.steps[run.current].prompts or run.steps[list(run.steps)[0]].prompts
    assert prompts and any("JSON" in p["content"] for p in prompts)
    assert run.execution.row_count == 10


def test_real_rag_adapter_parses_the_current_stub_and_forces_athena(agent_module, scenarios):
    adapter = AgentRAGAdapter(MODULE, "AWS Athena (Trino SQL)")
    result = adapter.search(scenarios[0].ir())
    assert result.dialect == "AWS Athena (Trino SQL)"
    assert result.raw_context["original_dialect"] == "snowflake"
    assert result.selected_tables == ["ho_master.t_o1dm_franchise_gm_daily"]
    assert result.owners[0].code == "o1dm" and not result.has_scores
    assert len(result.owners[0].tables[0].fields) == 5


def test_table_block_parser():
    block = parse_table_block("ho_master.t_abcd_x:\n-Description:desc\n-Fields:f1: etiqueta uno, descripción\n* f2: dos, d2")
    assert block["name"] == "ho_master.t_abcd_x"
    assert [f["name"] for f in block["fields"]] == ["f1", "f2"]


def test_aws_errors_become_presentable_messages():
    from botocore.exceptions import NoCredentialsError

    error = translate_agent_error(NoCredentialsError())
    assert isinstance(error, ServiceUnavailable) and error.can_fallback
    assert "credenciales" in error.message.lower()


def test_slow_real_service_times_out_with_a_presentable_error():
    import time

    from services.errors import ServiceError
    from services.watchdog import run_with_timeout

    with pytest.raises(ServiceError) as info:
        run_with_timeout(time.sleep, 2, timeout=0.2, service="agent", what="Amazon Bedrock")
    assert info.value.can_fallback and "tardando" in info.value.title
