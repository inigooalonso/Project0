"""Estado de sesión y acciones de la página Business Understanding."""
from __future__ import annotations

import streamlit as st

from core.settings import REAL_DIR
from services.bu.models import BUError, Turn
from services.bu.service import BUService, build_bu_service, translate_error
from services.examples import load_real_examples
from ui import runtime

TURNS_KEY = "bu_turns"
PENDING_KEY = "bu_pending"
QUESTION_KEY = "bu_question"
FOLLOW_KEY = "bu_follow"
EXAMPLES_PATH = REAL_DIR / "bu_examples.yaml"


@st.cache_resource(show_spinner="Abriendo la base de conocimiento…")
def _service() -> BUService:
    # Un único cliente de Qdrant por proceso (modo local). Si falla, no se cachea y se reintenta al recargar.
    return build_bu_service(runtime.settings())


def get_service() -> tuple[BUService | None, BUError | None]:
    try:
        return _service(), None
    except Exception as exc:
        return None, translate_error(exc)


def turns() -> list[Turn]:
    return st.session_state.setdefault(TURNS_KEY, [])


def examples():
    return load_real_examples(EXAMPLES_PATH)


def _ask(question: str) -> None:
    question = (question or "").strip()
    if not question:
        return
    follow = bool(st.session_state.get(FOLLOW_KEY)) and bool(turns()) and not turns()[-1].error
    st.session_state[PENDING_KEY] = (question, follow)


def submit() -> None:
    _ask(st.session_state.get(QUESTION_KEY, ""))


def ask_example(question: str) -> None:
    st.session_state[QUESTION_KEY] = question
    st.session_state[FOLLOW_KEY] = False
    _ask(question)


def pop_pending() -> tuple[str, bool] | None:
    return st.session_state.pop(PENDING_KEY, None)


def history_for(followup: bool):
    if followup and turns():
        return turns()[-1].messages or None
    return None


def new_conversation() -> None:
    st.session_state[TURNS_KEY] = []
    st.session_state[QUESTION_KEY] = ""
    st.session_state[FOLLOW_KEY] = False
