"""Business Understanding: tu código (rag_save, rag_retrieve, agentic_rag) a través del adaptador.

Sin red: Qdrant en memoria, embeddings deterministas y un LLM guionizado (tests/bu_fakes.py).
"""
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")

from langchain_core.messages import AIMessage  # noqa: E402

from services.bu.models import LLMEvent, ToolEvent  # noqa: E402
from tests.bu_fakes import call, make_service  # noqa: E402

QUESTION = "Explícame qué significa Franquicia de Distribución y qué mide"


def test_rag_save_chunks_by_headings_and_topics():
    service = make_service()
    catalog = service.catalog()
    assert set(catalog) == {"glosario", "procesos"}  # front matter «tema:» y subcarpeta
    hits = service.document("procesos/analitica_clientes_europa.md")
    assert [h.position for h in hits] == list(range(len(hits)))
    assert any(h.path.endswith("Fases del proceso > Validación") for h in hits)


def test_search_ranks_the_relevant_fragment_first():
    hits = make_service().search("qué es la Franquicia de Distribución y qué mide", 3)
    assert hits[0].chunk_id == "glosario/terminos.md#0" and hits[0].score > hits[-1].score


def test_agent_run_is_recorded_step_by_step():
    seen_live = []
    turn = make_service().ask(QUESTION, on_event=seen_live.append)
    assert turn.error is None and turn.steps == 3
    assert seen_live == turn.events  # cada evento se notifica al momento
    llm = [e for e in turn.events if isinstance(e, LLMEvent)]
    tools = [e for e in turn.events if isinstance(e, ToolEvent)]
    assert [e.step for e in llm] == [1, 2, 3] and llm[-1].is_final
    assert [t.name for t in tools] == ["search", "search", "document_outline", "read_section", "read_context"]
    assert tools[0].hits and tools[1].args["topic"] == "procesos"
    assert tools[2].outline and tools[3].hits
    assert tools[4].status == "error" and "no-existe" in tools[4].error  # el agente recibe el error y sigue
    assert turn.input_tokens and turn.output_tokens


def test_only_retrieved_fragments_count_as_sources():
    turn = make_service().ask(QUESTION)
    assert "inventado.md#7" in turn.cited and "inventado.md#7" not in turn.sources
    assert set(turn.sources) <= set(turn.seen)
    assert turn.sources == ["glosario/terminos.md#0", "procesos/analitica_clientes_europa.md#3"]


def test_follow_up_questions_reuse_the_conversation():
    service = make_service()
    first = service.ask(QUESTION)
    second = service.ask("¿Puedes dar más detalle?", history=first.messages)
    assert second.followup and len(second.messages) > len(first.messages)


def test_llm_failures_become_presentable_errors():
    from botocore.exceptions import NoCredentialsError

    def boom(messages):
        raise NoCredentialsError()

    turn = make_service(script=[boom]).ask(QUESTION)
    assert turn.error is not None and "credenciales" in turn.error.message.lower()


def test_max_steps_from_settings_is_respected():
    from core.settings import Settings

    loop = [lambda m: AIMessage(content="", tool_calls=[call("list_catalog", {}, "x")])]
    turn = make_service(script=loop, settings=Settings(bu_max_steps=3)).ask(QUESTION)
    assert turn.steps == 3 and len(turn.llm_events) == 3
    assert turn.tool_events[0].catalog


def test_missing_collection_is_detected():
    assert not make_service(indexed=False).exists()


def test_indexing_script_skips_hidden_folders(tmp_path):
    import importlib.util

    script = Path(__file__).resolve().parent.parent / "scripts" / "indexar_bu.py"
    spec = importlib.util.spec_from_file_location("indexar_bu", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / ".ipynb_checkpoints").mkdir()
    (tmp_path / "sub").mkdir()
    for name in ("a.md", "sub/b.md", ".ipynb_checkpoints/a-checkpoint.md"):
        (tmp_path / name).write_text("# T\n\ntexto", encoding="utf-8")
    assert [f.relative_to(tmp_path).as_posix() for f in module.markdown_files(tmp_path)] == ["a.md", "sub/b.md"]
