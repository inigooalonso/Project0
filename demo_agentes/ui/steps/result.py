"""Paso 7 · Ejecución: el resultado tal como lo devuelve la base de datos."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.formatting import fmt_int, fmt_seconds, fmt_tokens
from core.models import STEP_ORDER
from core.narrative import STEP_INFO
from core.pipeline import PipelineRun
from ui.components import esc, pill


def render(run: PipelineRun) -> None:
    execution = run.execution
    df = execution.df
    st.html(f'<p class="ada-question">«{esc(run.question)}»</p>')
    st.html('<div style="display:flex;gap:0.45rem;flex-wrap:wrap;margin-bottom:0.6rem">'
            + pill(f"Motor: {execution.engine}", "info") + pill(f"{fmt_int(execution.row_count)} filas")
            + pill(f"{len(df.columns)} columnas") + pill(f"Consulta: {fmt_seconds(execution.elapsed_s)}")
            + pill(f"Total: {fmt_seconds(run.total_elapsed)}") + "</div>")
    if df.empty:
        st.info("La consulta no ha devuelto filas.", icon=":material/search_off:")
    else:
        st.dataframe(df, hide_index=True, width="stretch")
    if execution.truncated:
        st.caption(f"Se muestran las primeras {fmt_int(len(df))} filas de {fmt_int(execution.row_count)}.")

    st.html('<div class="ada-section">Trazabilidad</div>')

    def tokens(record) -> str:
        if record.input_tokens is None:
            return "—"
        return f"{fmt_int(record.input_tokens)} → {fmt_int(record.output_tokens or 0)}"

    trace = pd.DataFrame([
        {
            "Paso": f"{STEP_INFO[s].number} · {STEP_INFO[s].title}",
            "Duración": fmt_seconds(run.steps[s].elapsed_s),
            "Origen": run.steps[s].source or "—",
            "Llamadas al LLM": str(run.steps[s].calls) if run.steps[s].calls else "—",
            "Tokens (entrada → salida)": tokens(run.steps[s]),
        }
        for s in STEP_ORDER
    ])
    st.dataframe(trace, hide_index=True, width="stretch")
    total_in = sum(r.input_tokens or 0 for r in run.steps.values())
    total_out = sum(r.output_tokens or 0 for r in run.steps.values())
    st.caption(f"Tokens totales: {fmt_tokens(total_in)} de entrada · {fmt_tokens(total_out)} de salida")
