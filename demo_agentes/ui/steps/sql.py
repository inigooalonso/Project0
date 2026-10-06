"""Paso 6 · SQL generada y validada."""
from __future__ import annotations

import streamlit as st

from core.pipeline import PipelineRun
from ui import runtime
from ui.components import esc, pill


def render(run: PipelineRun) -> None:
    result = run.sql
    ins = result.inspection
    authorized = len(ins.tables) - len(ins.unauthorized_tables)
    badges = [
        pill("Solo lectura", "ok"),
        pill("Una única sentencia", "ok"),
        pill(f"Tablas autorizadas {authorized}/{len(ins.tables)}", "ok"),
        pill(f"Dialecto {runtime.settings().dialect}", "info"),
        pill(f"{ins.join_count} {'join' if ins.join_count == 1 else 'joins'}"),
    ]
    st.html(f'<div style="display:flex;gap:0.45rem;flex-wrap:wrap;margin-bottom:0.8rem">{"".join(badges)}</div>')
    st.code(result.sql, language="sql", line_numbers=True)
    if result.assumptions:
        items = "".join(f"<li>{esc(a)}</li>" for a in result.assumptions)
        st.html(f'<div class="ada-section">Supuestos explícitos</div><ul style="margin-top:0;color:#46505C">{items}</ul>')

    st.html('<div class="ada-section">Validación (sqlglot)</div>')
    st.json(ins.model_dump(), expanded=False)
