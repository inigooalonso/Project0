"""Elementos comunes del escenario principal: cabecera, banners y errores."""
from __future__ import annotations

import streamlit as st

from core.formatting import fmt_seconds, fmt_tokens
from core.models import STEP_ORDER, StepId, StepStatus
from core.narrative import STEP_INFO
from core.pipeline import Phase, PipelineRun
from ui import runtime
from ui.components import esc, pill

SERVICE_LABELS = {"agent": "el LLM", "rag": "el RAG", "executor": "la ejecución", "knowledge": "joins y glosario",
                  "context": "el contexto"}


def stage_header(run: PipelineRun, step: StepId, technical: bool) -> None:
    info = STEP_INFO[step]
    record = run.steps[step]
    meta = []
    if record.status == StepStatus.DONE:
        meta.append(pill(fmt_seconds(record.elapsed_s), "info"))
    elif record.status == StepStatus.WAITING:
        meta.append(pill("En pausa · esperando respuesta", "warn"))
    if record.simulated:
        meta.append(pill("simulado", "mock", dot=True))
    if technical:
        if record.source:
            meta.append(pill(record.source))
        if record.input_tokens is not None:
            approx = "≈ " if record.tokens_estimated else ""
            meta.append(pill(f"{approx}{fmt_tokens(record.input_tokens)} → {fmt_tokens(record.output_tokens)} tokens"))
    st.html(
        f'<div class="ada-stage-head"><div><div class="num">Paso {info.number} de 7 · {esc(info.title)}</div>'
        f'<div class="title">{esc(info.stage_title)}</div><div class="what">{esc(info.what)}</div>'
        f'<div class="why">{esc(info.why)}</div></div><div class="meta">{"".join(meta)}</div></div>'
    )


def running_banner(run: PipelineRun):
    """Banner del paso en curso. Devuelve el callback de progreso del orquestador."""
    info = STEP_INFO[run.current]
    placeholder = st.empty()

    def draw(lines: list[str] | None = None) -> None:
        body = ""
        if lines:
            body = '<div class="lines">' + "".join(f"<div>{esc(line)}</div>" for line in lines[-6:]) + "</div>"
        placeholder.html(
            f'<div class="ada-running"><div class="title"><span class="ada-spinner"></span>'
            f'Paso {info.number} de 7 · {esc(info.title)}</div><div class="text">{esc(info.what)}</div>{body}'
            f'<div class="bar"></div></div>'
        )

    draw()
    return draw


def next_banner(run: PipelineRun) -> None:
    """Modo presentador: el paso ha terminado y se espera a «Siguiente paso»."""
    done = run.last_completed
    upcoming = STEP_INFO[run.current]
    cols = st.columns([0.72, 0.28], vertical_alignment="center")
    with cols[0]:
        title = STEP_INFO[done].title if done else ""
        st.html(f'<div class="ada-info"><b>Paso completado: {esc(title)}.</b> '
                f'A continuación: {esc(upcoming.title)} — {esc(upcoming.what)}</div>')
    with cols[1]:
        st.button("Siguiente paso", icon=":material/arrow_forward:", type="primary", width="stretch",
                  on_click=runtime.next_step, key="ada_next", shortcut="PageDown",
                  help="También con la tecla AvPág o el mando de presentaciones")


def error_card(run: PipelineRun, technical: bool) -> None:
    error = run.error
    if error is None:
        return
    st.html(f'<div class="ada-error"><div class="t">{esc(error.title)}</div><div class="m">{esc(error.message)}</div></div>')
    cols = st.columns([0.3, 0.38, 0.32])
    with cols[0]:
        st.button("Reintentar", icon=":material/refresh:", width="stretch", on_click=runtime.retry, key="ada_retry")
    with cols[1]:
        if error.can_fallback and error.service in ("agent", "rag", "executor"):
            st.button(f"Continuar con datos simulados ({SERVICE_LABELS[error.service]})", icon=":material/play_arrow:",
                      type="primary", width="stretch", on_click=runtime.fallback_to_mock, args=(error.service,), key="ada_fallback")
    with cols[2]:
        st.button("Empezar de nuevo", icon=":material/restart_alt:", width="stretch", on_click=runtime.reset_run, key="ada_reset_err")
    if technical and error.detail:
        with st.expander("Detalle técnico"):
            st.code(error.detail, language="text", wrap_lines=True)


def focus_notice(run: PipelineRun) -> None:
    if run.focus is None or run.phase == Phase.DONE:
        return
    info = STEP_INFO[run.focus]
    cols = st.columns([0.7, 0.3], vertical_alignment="center")
    cols[0].caption(f"Estás revisando el paso {info.number} · {info.title}. El pipeline sigue su curso.")
    cols[1].button("Volver al paso actual", icon=":material/my_location:", type="tertiary", on_click=runtime.follow_progress,
                   key="ada_follow", width="stretch")


def welcome() -> None:
    """Estado inicial: qué va a pasar, antes de lanzar ninguna pregunta."""
    st.html(
        '<div class="ada-stage-head"><div><div class="num">Cómo funciona</div>'
        '<div class="title">De la pregunta al dato, en 7 pasos visibles</div>'
        '<div class="what">Elige una pregunta de ejemplo o escribe la tuya. ADA mostrará cada paso que da hasta la respuesta: '
        'qué ha entendido, dónde están los datos, cómo se relacionan, qué recibe el LLM, si tiene dudas, '
        'la SQL que escribe y el resultado.</div></div></div>'
    )
    cards = []
    for step in STEP_ORDER:
        info = STEP_INFO[step]
        cards.append(f'<div class="ada-card"><h4>{info.number} · {esc(info.title)}</h4><p>{esc(info.why)}</p></div>')
    st.html('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:0.8rem">' + "".join(cards) + "</div>")


def first_step_placeholder(run: PipelineRun) -> None:
    st.html(f'<p class="ada-question">«{esc(run.question)}»</p>')
