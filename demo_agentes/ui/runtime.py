"""Puente entre Streamlit y la orquestación: estado de sesión y acciones.

Las páginas solo llaman a estas funciones; aquí se avanza la máquina de
estados un paso por rerun.
"""
from __future__ import annotations

import streamlit as st

from core.models import StepId
from core.orchestrator import Orchestrator
from core.pipeline import PipelineRun
from core.settings import Settings, load_settings
from services.examples import load_real_examples
from services.factory import Services, build_services

RUN_KEY = "ada_run"
QUESTION_KEY = "ada_question_input"
ANSWER_KEY = "ada_answer_{run}_{round}_{index}"
ANSWER_ERROR_KEY = "ada_answer_error"


def settings() -> Settings:
    # Sin caché: se relee config/settings.toml en cada ejecución (es un fichero pequeño). Así, al
    # cambiar el fichero o core/settings.py con la app en marcha, nunca queda una configuración antigua.
    return load_settings()


@st.cache_resource(show_spinner=False)
def _services(key: str, _settings: Settings) -> Services:
    return build_services(_settings)


def services() -> Services:
    # Los servicios sí se guardan, pero se reconstruyen si cambia la configuración.
    current = settings()
    return _services(repr(current), current)


# ---------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------

def get_run() -> PipelineRun | None:
    return st.session_state.get(RUN_KEY)


def start_run(question: str) -> None:
    question = (question or "").strip()
    if not question:
        return
    st.session_state[ANSWER_ERROR_KEY] = False
    st.session_state[RUN_KEY] = PipelineRun(question=question, autoplay=settings().autoplay)


def start_example(question: str) -> None:
    st.session_state[QUESTION_KEY] = question
    start_run(question)


def submit_question() -> None:
    start_run(st.session_state.get(QUESTION_KEY, ""))


def reset_run() -> None:
    st.session_state.pop(RUN_KEY, None)
    st.session_state[QUESTION_KEY] = ""


def focus_step(step: StepId) -> None:
    run = get_run()
    if run is not None:
        run.focus = None if run.visible_step() == step and run.focus is not None else step


def follow_progress() -> None:
    run = get_run()
    if run is not None:
        run.focus = None


def answer_key(run: PipelineRun, index: int) -> str:
    return ANSWER_KEY.format(run=run.run_id, round=len(run.rounds), index=index)


def submit_answers() -> None:
    """Envía de una vez las respuestas a todas las preguntas de la ronda."""
    run = get_run()
    if run is None:
        return
    answers = [st.session_state.get(answer_key(run, i), "") for i in range(len(run.pending_questions))]
    st.session_state[ANSWER_ERROR_KEY] = not run.answer(answers)


def next_step() -> None:
    run = get_run()
    if run is not None:
        run.next_step()


def retry() -> None:
    run = get_run()
    if run is not None:
        run.retry()


def examples():
    return load_real_examples()


def advance(run: PipelineRun | None) -> None:
    """Ejecuta el paso pendiente (si lo hay) y vuelve a pintar la página."""
    if run is None or not run.needs_execution:
        return
    Orchestrator(services(), settings()).run_current_step(run)
    st.rerun()
