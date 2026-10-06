"""Paso 5 · Aclaraciones: el agente hace todas sus preguntas a la vez y espera las respuestas."""
from __future__ import annotations

import streamlit as st

from core.pipeline import Phase, PipelineRun
from ui import runtime
from ui.components import esc


def _question(number: int, text: str) -> str:
    return f'<div class="ada-q"><span class="n">{number}</span><span class="t">{esc(text)}</span></div>'


def render(run: PipelineRun) -> None:
    waiting = run.phase == Phase.WAITING_USER and bool(run.pending_questions)
    answered = [r for r in run.rounds if r.answers is not None]
    if not run.rounds:
        st.html('<p class="ada-understood">Con el contexto disponible la pregunta no es ambigua: '
                'no ha hecho falta preguntar nada.</p>')

    for i, rnd in enumerate(answered, start=1):
        items = "".join(_question(n, q) + f'<div class="a">{esc(a)}</div>'
                        for n, (q, a) in enumerate(zip(rnd.questions, rnd.answers), start=1))
        st.html(f'<div class="ada-round"><div class="h">Ronda {i} · respondida</div>{items}</div>')

    if waiting:
        questions = run.pending_questions
        n = len(questions)
        st.html(f'<p class="ada-understood">El agente necesita {"aclarar un punto" if n == 1 else f"aclarar {n} puntos"} '
                'antes de escribir la SQL.</p>')
        with st.form(key=f"ada_answers_{run.run_id}_{len(run.rounds)}", border=True, enter_to_submit=False):
            for index, question in enumerate(questions):
                st.html(_question(index + 1, question))
                st.text_area(f"Respuesta {index + 1}", key=runtime.answer_key(run, index), height=68,
                             placeholder="Tu respuesta…", label_visibility="collapsed")
            st.form_submit_button("Enviar respuestas" if n > 1 else "Enviar respuesta", icon=":material/send:",
                                  type="primary", on_click=runtime.submit_answers)
        if st.session_state.get(runtime.ANSWER_ERROR_KEY):
            st.warning("Responde al menos a una pregunta para continuar.", icon=":material/edit_note:")
        st.caption("El pipeline está en pausa en el paso 5. Las preguntas que dejes en blanco se envían como "
                   "«sin respuesta» y el agente declarará el supuesto que use.")
    elif answered:
        st.caption("Respuestas registradas: el agente ha continuado con la generación de la SQL.")

    st.html('<div class="ada-section">Estado del agente</div>')
    st.json({
        "pending_question": run.state.get("pending_question", ""),
        "clarifications": run.state.get("clarifications", []),
        "rondas": len(run.rounds),
    }, expanded=False)
    st.caption("Equivale a interrupt() + Command(resume=…) de tu grafo: la máquina de estados guarda las preguntas, "
               "espera las respuestas y vuelve a decidir (máximo de rondas en config/settings.toml).")
