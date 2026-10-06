"""La aplicación completa, de extremo a extremo, con el ejecutor de tests de Streamlit.

Los servicios se sustituyen por los dobles de tests/fakes.py (mismos contratos
que tu agente, tu RAG y Athena).
"""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from core.pipeline import Phase
from tests import fakes
from ui.components import esc

APP = Path(__file__).resolve().parent.parent / "app.py"
ADA = "app_pages/ada_text2sql.py"


def make_app(monkeypatch, agent=None):
    from ui import runtime

    st.cache_resource.clear()
    monkeypatch.setattr(runtime, "build_services", lambda settings: fakes.fake_services(agent))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    return at


@pytest.fixture
def app(monkeypatch):
    return make_app(monkeypatch)


def html_text(at) -> str:
    return " ".join(str(h.proto.body) for h in at.get("html"))


def button(at, key):
    return next(b for b in at.button if b.key == key)


def test_home_and_business_understanding_render(app):
    assert "Agentes de IA que responden" in html_text(app)
    app.switch_page("app_pages/business_understanding.py").run()
    assert not app.exception
    assert "Próximamente" in html_text(app)


def test_example_question_pauses_with_all_questions_at_once(app):
    app.switch_page(ADA).run()
    assert not app.get("radio") and not app.sidebar.get("radio")  # sin selector de modos ni de vista
    example = next(b for b in app.button if b.key and b.key.startswith("ex_"))
    example.click().run()
    run = app.session_state["ada_run"]
    assert run.phase == Phase.WAITING_USER
    assert all(esc(q) in html_text(app) for q in fakes.QUESTIONS)
    assert len(app.text_area) == len(fakes.QUESTIONS)

    app.text_area[0].input("Margen")
    submit = next(b for b in app.button if b.label == "Enviar respuestas")
    submit.click().run()
    assert not app.exception
    run = app.session_state["ada_run"]
    assert run.phase == Phase.DONE, run.error
    assert run.state["clarifications"][0] == {"question": fakes.QUESTIONS[0], "answer": "Margen"}


def test_every_step_renders_without_errors(monkeypatch):
    app = make_app(monkeypatch, fakes.FakeAgent(questions=[]))
    app.switch_page(ADA).run()
    app.text_input(key="ada_question_input").input(fakes.QUESTION).run()
    assert app.session_state["ada_run"].phase == Phase.DONE
    assert not app.get("plotly_chart")  # la ejecución solo muestra el DataFrame
    for key in ("stepbtn_pseudocode", "stepbtn_rag", "stepbtn_joins", "stepbtn_context", "stepbtn_clarify",
                "stepbtn_sql", "stepbtn_execute"):
        button(app, key).click().run()
        assert not app.exception, key
        assert "No se ha podido dibujar" not in str(app.get("alert")), key


def test_pseudocode_marks_unresolved_concepts_in_red(monkeypatch):
    app = make_app(monkeypatch, fakes.FakeAgent(questions=[]))
    app.switch_page(ADA).run()
    app.text_input(key="ada_question_input").input(fakes.QUESTION).run()
    button(app, "stepbtn_pseudocode").click().run()
    text = html_text(app)
    assert 'ada-pill warn">Duda detectada' in text
    assert 'ada-pill err">Concepto por resolver' in text
