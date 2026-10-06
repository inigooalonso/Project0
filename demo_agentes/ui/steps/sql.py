"""Paso 6 · SQL generada y validada."""
from __future__ import annotations

import streamlit as st

from core.pipeline import PipelineRun
from services.sql_guard import to_duckdb
from ui import runtime
from ui.components import esc, pill


def render(run: PipelineRun, technical: bool) -> None:
    result = run.sql
    ins = result.inspection
    settings = runtime.run_settings(run)
    authorized = len(ins.tables) - len(ins.unauthorized_tables)
    badges = [
        pill("Solo lectura", "ok"),
        pill("Una única sentencia", "ok"),
        pill(f"Tablas autorizadas {authorized}/{len(ins.tables)}", "ok"),
        pill(f"Dialecto {settings.dialect}", "info"),
        pill(f"{ins.join_count} {'join' if ins.join_count == 1 else 'joins'}"),
    ]
    st.html(f'<div style="display:flex;gap:0.45rem;flex-wrap:wrap;margin-bottom:0.8rem">{"".join(badges)}</div>')
    if run.explanation:
        st.html(f'<p style="font-size:1.08rem;color:#46505C;margin:0 0 0.8rem 0">{esc(run.explanation)}</p>')
    st.code(result.sql, language="sql", line_numbers=technical)
    if result.assumptions:
        items = "".join(f"<li>{esc(a)}</li>" for a in result.assumptions)
        st.html(f'<div class="ada-section">Supuestos explícitos</div><ul style="margin-top:0;color:#46505C">{items}</ul>')

    if technical:
        st.html('<div class="ada-section">Validación (sqlglot)</div>')
        st.json(ins.model_dump(), expanded=False)
        if settings.executor_mode == "mock":
            with st.expander("Traducción para el motor simulado (DuckDB)"):
                st.code(to_duckdb(result.sql, settings.sqlglot_dialect), language="sql")
                st.caption("Solo cambia la sintaxis: la lógica es la misma que se ejecutaría en Athena.")
