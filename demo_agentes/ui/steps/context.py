"""Paso 4 · Contexto: qué recibe exactamente el LLM."""
from __future__ import annotations

import json

import streamlit as st

from core.formatting import fmt_int, fmt_tokens
from core.models import Kpi
from core.pipeline import PipelineRun
from services.agent.base import clarification_payload
from ui.charts import CONFIG, context_figure
from ui.components import esc, kpi_row


def render(run: PipelineRun, technical: bool) -> None:
    b = run.context
    tables = [block.split(":", 1)[0] for block in b.context["authorized_tables"] if isinstance(block, str)]
    kpi_row([
        Kpi("Tablas autorizadas", fmt_int(b.table_count), "lo único que el LLM puede consultar"),
        Kpi("Campos descritos", fmt_int(b.field_count), "con su significado de negocio"),
        Kpi("Fragmentos del RAG", fmt_int(b.fragment_count), "una coincidencia por entidad"),
        Kpi("Glosario y joins", f"{fmt_int(b.glossary_count)} + {fmt_int(b.join_count)}", "definiciones + reglas de unión"),
        Kpi("Tamaño del contexto", f"≈ {fmt_tokens(b.total_tokens)}", "tokens (estimación)"),
    ])
    st.html('<div class="ada-section">Composición del contexto (tokens por bloque)</div>')
    st.plotly_chart(context_figure(b), config=CONFIG, theme=None, key="ada_context_fig")
    chips = "".join(f'<span class="code" style="font-size:0.9rem">{esc(t)}</span>' for t in tables)
    st.html(f'<div class="ada-info"><b>Perímetro de seguridad.</b> El LLM solo recibe estas tablas y las definiciones aprobadas; '
            f'si la SQL usa cualquier otra, se bloquea en el paso 6.<div style="margin-top:0.5rem" class="ada-card" '
            f'>{chips}</div></div>')

    if technical:
        st.html('<div class="ada-section">Lo que se envía al LLM</div>')
        tab_msg, tab_tables, tab_ctx = st.tabs(["Mensaje de usuario (decidir aclaración)", "Tablas autorizadas", "Contexto (JSON)"])
        with tab_msg:
            payload = clarification_payload(run.state)
            st.code(json.dumps(payload, ensure_ascii=False, indent=2, default=str), language="json", height=360)
            st.caption("Mismo JSON que construye decide_if_clarification_is_needed; generate_sql envía la misma "
                       "estructura con «clarifications».")
        with tab_tables:
            st.code("\n\n".join(b.context["authorized_tables"]), language="text", height=360, wrap_lines=True)
        with tab_ctx:
            st.json(b.context, expanded=False)
