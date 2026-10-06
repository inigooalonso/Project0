"""Paso 5 · Aclaraciones: el pipeline se pausa, pregunta y continúa."""
from __future__ import annotations

import streamlit as st

from core.models import StepId
from core.pipeline import Phase, PipelineRun
from ui import runtime

AVATARS = {"assistant": ":material/smart_toy:", "user": ":material/person:"}


def render(run: PipelineRun, technical: bool) -> None:
    waiting = run.phase == Phase.WAITING_USER
    if not run.chat:
        st.html('<p class="ada-understood">Con el contexto disponible la pregunta no es ambigua: '
                'no ha hecho falta preguntar nada.</p>')
    with st.container(key="ada_chat"):
        for message in run.chat:
            with st.chat_message(message.role, avatar=AVATARS.get(message.role)):
                st.markdown(message.content)
        if waiting:
            st.caption("El pipeline está en pausa en el paso 5. Responde y continuará exactamente desde aquí.")
            if run.clarification_options:
                with st.container(horizontal=True, key="ada_quick_replies"):
                    for i, option in enumerate(run.clarification_options):
                        st.button(option.label, key=f"ada_reply_{i}", icon=":material/reply:", type="primary" if i == 0 else "secondary",
                                  on_click=runtime.answer, args=(option.label,))
            st.chat_input("Escribe tu respuesta…", key=runtime.CHAT_KEY, on_submit=runtime.answer_from_chat)
        elif run.chat:
            st.caption("Respuesta registrada: el agente ha continuado con la generación de la SQL.")

    if technical:
        st.html('<div class="ada-section">Estado del agente</div>')
        answered = run.state.get("clarifications", [])
        st.json({
            "pending_question": run.state.get("pending_question", ""),
            "clarifications": answered,
            "llamadas_a_decide_if_clarification_is_needed": run.steps[StepId.CLARIFY].calls,
        }, expanded=True)
        st.caption("Equivale a interrupt() + Command(resume=…) de tu grafo: la máquina de estados guarda la pregunta, "
                   "espera la respuesta y vuelve a decidir (máximo 3 aclaraciones, como en tu nodo).")
