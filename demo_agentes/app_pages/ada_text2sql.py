"""ADA · Text2SQL: el pipeline se despliega paso a paso ante la audiencia."""
from __future__ import annotations

import streamlit as st

from ui import runtime, stepper
from ui.components import page_header
from ui.steps import render_stage

run = runtime.get_run()
if run is not None:
    run.autoplay = runtime.settings().autoplay

page_header(
    "Agente 1 · Disponible",
    "ADA · Text2SQL",
    "De la pregunta de negocio a la respuesta con datos: SQL validada y ejecutada en Athena.",
)

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
    examples = runtime.examples()
    if examples:
        with st.container(horizontal=True, key="ada_examples", gap="small"):
            st.caption("Ejemplos:", width="content")
            for example in examples:
                st.button(example.label, key=f"ex_{example.id}", icon=example.icon,
                          on_click=runtime.start_example, args=(example.question,))

st.space("small")

# --- Stepper + escenario -------------------------------------------------
stepper_col, stage_col = st.columns([0.25, 0.75], gap="large")
with stepper_col:
    stepper.render(run)
with stage_col:
    render_stage(run)

# --- Avance de la máquina de estados (siempre al final del script) ------
runtime.advance(run)
