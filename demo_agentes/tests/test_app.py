"""La aplicación completa, de extremo a extremo, con el ejecutor de tests de Streamlit."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.pipeline import Phase

APP = Path(__file__).resolve().parent.parent / "app.py"
ADA = "app_pages/ada_text2sql.py"


@pytest.fixture
def app():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    return at


def html_text(at) -> str:
    return " ".join(str(h.proto.body) for h in at.get("html"))


def button(at, key):
    return next(b for b in at.button if b.key == key)


def test_home_and_business_understanding_render(app):
    assert "Agentes de IA que responden" in html_text(app)
    app.switch_page("app_pages/business_understanding.py").run()
    assert not app.exception
    assert "Próximamente" in html_text(app)


def test_example_question_runs_all_steps_and_switches_views(app):
    app.switch_page(ADA).run()
    button(app, "ex_hipotecas_oficina").click().run()
    assert not app.exception
    run = app.session_state["ada_run"]
    assert run.phase == Phase.DONE
    assert "encabeza el resultado" in html_text(app)
    app.session_state["ada_view"] = "Técnica"
    app.run()
    assert not app.exception
    for key in ("stepbtn_pseudocode", "stepbtn_rag", "stepbtn_joins", "stepbtn_context", "stepbtn_clarify", "stepbtn_sql"):
        button(app, key).click().run()
        assert not app.exception, key


def test_clarification_pauses_until_the_user_answers(app):
    app.switch_page(ADA).run()
    button(app, "ex_morosidad_oficinas").click().run()
    run = app.session_state["ada_run"]
    assert run.phase == Phase.WAITING_USER
    assert "¿Cuál quieres usar" in " ".join(m.markdown[0].value for m in app.chat_message if m.markdown)
    button(app, "ada_reply_0").click().run()
    assert not app.exception
    assert app.session_state["ada_run"].phase == Phase.DONE


def test_free_question_in_mock_mode_shows_a_friendly_notice(app):
    app.switch_page(ADA).run()
    app.text_input(key="ada_question_input").input("¿Cuántos clientes tenemos en Bilbao?").run()
    assert not app.exception
    assert "ada_run" not in app.session_state
    assert "fuera del guion de la demo" in html_text(app)


def test_services_are_fixed_when_the_question_starts(app):
    from ui import runtime  # noqa: F401  (mismo módulo que usa la app)

    app.switch_page(ADA).run()
    button(app, "ex_hipotecas_oficina").click().run()
    run = app.session_state["ada_run"]
    assert run.modes == {"agent_mode": "mock", "rag_mode": "mock", "executor_mode": "mock"}
