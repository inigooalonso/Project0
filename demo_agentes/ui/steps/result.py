"""Paso 7 · Ejecución: resultado, indicadores y gráfico según la forma del dato."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.formatting import fmt_date, fmt_int, fmt_month, fmt_seconds, fmt_tokens, is_integral, pretty_column
from core.models import STEP_ORDER, ResultProfile
from core.narrative import STEP_INFO
from core.pipeline import PipelineRun
from core.result_profile import build_kpis, format_measure, headline, percent_decimals, profile_result
from ui.charts import CONFIG, latency_figure, result_figure
from ui.components import esc, kpi_row, pill


def _is_month_end(series: pd.Series) -> bool:
    dates = pd.to_datetime(series, errors="coerce").dropna()
    return bool(len(dates)) and bool((dates.dt.is_month_end).all())


def styled_table(df: pd.DataFrame, profile: ResultProfile):
    formats = {}
    for col in df.columns:
        label = pretty_column(col)
        if col in profile.measures:
            kind, integral = profile.measure_kinds.get(col, "number"), is_integral(df[col])
            decimals = percent_decimals(df[col]) if kind == "percent" else 2
            formats[label] = lambda v, k=kind, i=integral, d=decimals: format_measure(v, k, compact=False, integral=i, decimals=d)
        elif col in profile.times:
            formats[label] = fmt_month if _is_month_end(df[col]) else fmt_date
    shown = df.rename(columns=pretty_column)
    return shown.style.format(formats, na_rep="—")


def render(run: PipelineRun, technical: bool) -> None:
    execution = run.execution
    df = execution.df
    profile = profile_result(df, run.sql.inspection.order_by if run.sql else None)
    title = headline(df, profile)
    st.html(f'<p class="ada-question">«{esc(run.question)}»</p>'
            + (f'<p class="ada-headline">{esc(title)}</p>' if title else ""))
    if df.empty:
        st.info("La consulta no ha devuelto filas para este periodo o filtro.", icon=":material/search_off:")
        return
    kpi_row(build_kpis(df, profile))

    figure = result_figure(df, profile)
    if figure is not None:
        st.plotly_chart(figure, config=CONFIG, theme=None, key="ada_result_fig")
    st.dataframe(styled_table(df, profile), hide_index=True, width="stretch")
    if execution.truncated:
        st.caption(f"Se muestran las primeras {fmt_int(len(df))} filas de {fmt_int(execution.row_count)}.")

    st.html('<div class="ada-section">Cómo lo he calculado</div>')
    explanation = esc(run.explanation or "")
    answers = run.state.get("clarifications", [])
    if answers:
        explanation += "".join(f'<br/><span class="ada-muted">Aclaración: «{esc(a["answer"])}»</span>' for a in answers)
    st.html(f'<p style="color:#46505C;font-size:1.02rem;margin:0">{explanation}</p>')

    if technical:
        st.html('<div class="ada-section">Trazabilidad de la ejecución</div>')
        st.html('<div style="display:flex;gap:0.45rem;flex-wrap:wrap;margin-bottom:0.6rem">'
                + pill(f"Motor: {execution.engine}", "info") + pill(f"{fmt_int(execution.row_count)} filas")
                + pill(f"Consulta: {fmt_seconds(execution.elapsed_s)}") + pill(f"Total: {fmt_seconds(run.total_elapsed)}")
                + "</div>")
        rows = [(f"{STEP_INFO[s].number} · {STEP_INFO[s].title}", run.steps[s].elapsed_s, bool(run.steps[s].simulated))
                for s in STEP_ORDER]
        st.caption("Duración por paso (tono claro = simulado)")
        st.plotly_chart(latency_figure(rows), config=CONFIG, theme=None, key="ada_latency_fig")

        def tokens(record) -> str:
            if record.input_tokens is None:
                return "—"
            approx = "≈ " if record.tokens_estimated else ""
            return f"{approx}{fmt_int(record.input_tokens)} → {fmt_int(record.output_tokens or 0)}"

        trace = pd.DataFrame([
            {
                "Paso": f"{STEP_INFO[s].number} · {STEP_INFO[s].title}",
                "Duración": fmt_seconds(run.steps[s].elapsed_s),
                "Origen": run.steps[s].source or "—",
                "Simulado": "sí" if run.steps[s].simulated else "no",
                "Llamadas al LLM": str(run.steps[s].calls) if run.steps[s].calls else "—",
                "Tokens (entrada → salida)": tokens(run.steps[s]),
            }
            for s in STEP_ORDER
        ])
        st.dataframe(trace, hide_index=True, width="stretch")
        total_in = sum(r.input_tokens or 0 for r in run.steps.values())
        total_out = sum(r.output_tokens or 0 for r in run.steps.values())
        estimated = any(r.tokens_estimated for r in run.steps.values())
        st.caption(f"Tokens totales: {'≈ ' if estimated else ''}{fmt_tokens(total_in)} de entrada · "
                   f"{fmt_tokens(total_out)} de salida" + (" (estimados: LLM simulado)" if estimated else ""))
        if execution.simulated and execution.executed_sql:
            with st.expander("SQL ejecutada en el motor simulado"):
                st.code(execution.executed_sql, language="sql")
