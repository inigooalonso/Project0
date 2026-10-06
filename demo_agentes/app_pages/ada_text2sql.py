"""ADA · Text2SQL: el pipeline se despliega paso a paso ante la audiencia."""
from __future__ import annotations

import streamlit as st

from ui import runtime, stepper
from ui.components import esc, mode_pill, page_header
from ui.steps import render_stage

settings = runtime.current_settings()
run = runtime.get_run()
if run is not None:
    run.autoplay = settings.autoplay
technical = runtime.is_technical()

# --- Cabecera ---------------------------------------------------------
head_left, head_right = st.columns([0.66, 0.34], vertical_alignment="bottom")
with head_left:
    page_header(
        "Agente 1 · Disponible",
        "ADA · Text2SQL",
        "De la pregunta de negocio a la respuesta con datos: SQL validada, ejecutada y explicada.",
    )
with head_right:
    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", key="ada_head_tools"):
        st.html(mode_pill(runtime.run_settings(run).simulated_services))
        runtime.render_view_toggle()

# --- Pregunta y ejemplos -----------------------------------------------
with st.container(key="ada_question_bar"):
    q_col, b_col = st.columns([0.84, 0.16], vertical_alignment="bottom")
    with q_col:
        st.text_input(
            "Pregunta de negocio", key=runtime.QUESTION_KEY, placeholder="Escribe una pregunta de negocio…",
            label_visibility="collapsed", on_change=runtime.submit_question,
        )
    with b_col:
        st.button("Preguntar", icon=":material/send:", type="primary", width="stretch", on_click=runtime.submit_question)
    with st.container(horizontal=True, key="ada_examples", gap="small"):
        st.caption("Ejemplos:", width="content")
        for scenario in runtime.examples():
            st.button(scenario.label, key=f"ex_{scenario.id}", icon=scenario.icon,
                      on_click=runtime.start_example, args=(scenario.question,))

notice = st.session_state.get(runtime.NOTICE_KEY)
if notice:
    st.html(
        f'<div class="ada-info" style="margin-top:0.6rem"><b>Esta pregunta queda fuera del guion de la demo.</b> '
        f'Con el LLM simulado, ADA responde a las preguntas de ejemplo; para preguntas libres activa Bedrock en '
        f'«Ajustes de la demo» (barra lateral). Pregunta recibida: «{esc(notice)}».</div>'
    )

st.space("small")

# --- Stepper + escenario -------------------------------------------------
stepper_col, stage_col = st.columns([0.25, 0.75], gap="large")
with stepper_col:
    stepper.render(run)
with stage_col:
    progress = render_stage(run, technical)

# --- Avance de la máquina de estados (siempre al final del script) ------
runtime.advance(run, progress)
