"""Puente entre Streamlit y la orquestación: estado de sesión y acciones.

Las páginas solo llaman a estas funciones; aquí se decide qué configuración
aplica (fichero + ajustes de la barra lateral + «continuar con mock» de la
ejecución en curso) y se avanza la máquina de estados un paso por rerun.
"""
from __future__ import annotations

import streamlit as st

from core.models import StepId
from core.orchestrator import Orchestrator, ProgressCallback
from core.pipeline import PipelineRun
from core.settings import Settings, load_settings
from services.factory import Services, build_services
from services.scenarios import load_scenarios, match_scenario

RUN_KEY = "ada_run"
VIEW_KEY = "ada_view"
NOTICE_KEY = "ada_notice"
QUESTION_KEY = "ada_question_input"
CHAT_KEY = "ada_chat_input"

VIEWS = ("Ejecutiva", "Técnica")
SERVICE_FIELDS = {"agent": "agent_mode", "rag": "rag_mode", "executor": "executor_mode"}


@st.cache_resource(show_spinner=False)
def base_settings() -> Settings:
    return load_settings()


def current_settings() -> Settings:
    ss = st.session_state
    return base_settings().with_overrides(
        agent_mode=ss.get("cfg_agent"),
        rag_mode=ss.get("cfg_rag"),
        executor_mode=ss.get("cfg_executor"),
        autoplay=ss.get("cfg_autoplay"),
        speed=ss.get("cfg_speed"),
    )


def run_settings(run: PipelineRun | None) -> Settings:
    """Configuración efectiva de una ejecución: los servicios con los que se lanzó
    la pregunta más sus «continuar con datos simulados». El ritmo y el avance
    automático sí se pueden cambiar en marcha."""
    settings = current_settings()
    if run is None:
        return settings
    overrides = {SERVICE_FIELDS[s]: "mock" for s in run.overrides if s in SERVICE_FIELDS}
    return settings.with_overrides(**run.modes).with_overrides(**overrides)


@st.cache_resource(show_spinner=False)
def _services(agent_mode: str, rag_mode: str, executor_mode: str, _settings: Settings) -> Services:
    return build_services(_settings)


def services_for(settings: Settings) -> Services:
    return _services(settings.agent_mode, settings.rag_mode, settings.executor_mode, settings)


# ---------------------------------------------------------------------
# Estado de la vista
# ---------------------------------------------------------------------

def view_mode() -> str:
    return st.session_state.get(VIEW_KEY, VIEWS[0])


def is_technical() -> bool:
    return view_mode() == VIEWS[1]


def _sync_view() -> None:
    value = st.session_state.get("ada_view_widget")
    if value:
        st.session_state[VIEW_KEY] = value


def render_view_toggle() -> None:
    st.segmented_control(
        "Vista", VIEWS, key="ada_view_widget", default=view_mode(), required=True,
        on_change=_sync_view, label_visibility="collapsed",
    )


# ---------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------

def get_run() -> PipelineRun | None:
    return st.session_state.get(RUN_KEY)


def start_run(question: str) -> None:
    question = (question or "").strip()
    if not question:
        return
    settings = current_settings()
    if settings.agent_mode == "mock" and match_scenario(question) is None:
        st.session_state[NOTICE_KEY] = question
        return
    st.session_state[NOTICE_KEY] = None
    modes = {field: getattr(settings, field) for field in SERVICE_FIELDS.values()}
    st.session_state[RUN_KEY] = PipelineRun(question=question, autoplay=settings.autoplay, modes=modes)


def start_example(question: str) -> None:
    st.session_state[QUESTION_KEY] = question
    start_run(question)


def submit_question() -> None:
    start_run(st.session_state.get(QUESTION_KEY, ""))


def reset_run() -> None:
    st.session_state.pop(RUN_KEY, None)
    st.session_state[NOTICE_KEY] = None
    st.session_state[QUESTION_KEY] = ""


def focus_step(step: StepId) -> None:
    run = get_run()
    if run is not None:
        run.focus = None if run.visible_step() == step and run.focus is not None else step


def follow_progress() -> None:
    run = get_run()
    if run is not None:
        run.focus = None


def answer(text: str) -> None:
    run = get_run()
    if run is not None:
        run.answer(text)


def answer_from_chat() -> None:
    answer(st.session_state.get(CHAT_KEY) or "")


def next_step() -> None:
    run = get_run()
    if run is not None:
        run.next_step()


def retry() -> None:
    run = get_run()
    if run is not None:
        run.retry()


def fallback_to_mock(service: str) -> None:
    run = get_run()
    if run is not None:
        run.fallback_to_mock(service)


def examples():
    return load_scenarios()


def advance(run: PipelineRun | None, progress: ProgressCallback | None) -> None:
    """Ejecuta el paso pendiente (si lo hay) y vuelve a pintar la página."""
    if run is None or not run.needs_execution:
        return
    settings = run_settings(run)
    Orchestrator(services_for(settings), settings).run_current_step(run, progress)
    st.rerun()
