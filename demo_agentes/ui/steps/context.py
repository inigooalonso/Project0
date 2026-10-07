"""Paso 4 · Contexto: qué recibe exactamente el LLM."""
from __future__ import annotations

import json

import streamlit as st

from core.formatting import fmt_int, fmt_tokens
from core.pipeline import PipelineRun
from services.agent.base import clarification_payload
from ui.components import Kpi, esc, kpi_row


def render(run: PipelineRun) -> None:
    b = run.context
    tables = [block.split(":", 1)[0] for block in b.context["authorized_tables"] if isinstance(block, str)]
    kpi_row([
        Kpi("Tablas autorizadas", fmt_int(b.table_count), "lo único que el LLM puede consultar"),
        Kpi("Campos descritos", fmt_int(b.field_count), "con su significado de negocio"),
        Kpi("Fragmentos del RAG", fmt_int(b.fragment_count), "schema_context"),
        Kpi("Glosario y joins", f"{fmt_int(b.glossary_count)} + {fmt_int(b.join_count)}", "definiciones + reglas de unión"),
        Kpi("Tamaño del contexto", f"≈ {fmt_tokens(b.total_tokens)}", "tokens (estimación)"),
    ])
    chips = "".join(f'<span class="code" style="font-size:0.9rem">{esc(t)}</span>' for t in tables)
    st.html(f'<div class="ada-info"><b>Perímetro de seguridad.</b> El LLM solo recibe estas tablas y las definiciones aprobadas; '
            f'si la SQL usa cualquier otra, se bloquea en el paso 6.<div style="margin-top:0.5rem" class="ada-card" '
            f'>{chips}</div></div>')

    st.html('<div class="ada-section">Lo que se envía al LLM</div>')
    tab_msg, tab_tables, tab_fields, tab_ctx = st.tabs(
        ["Mensaje de usuario (aclaraciones)", "Tablas autorizadas", "Campos (schema_context)", "Contexto (JSON)"])
    with tab_msg:
        payload = clarification_payload(run.state)
        st.code(json.dumps(payload, ensure_ascii=False, indent=2, default=str), language="json", height=360)
        st.caption("Mismo JSON que construye decide_if_clarification_is_needed; generate_sql envía la misma "
                   "estructura con «clarifications».")
    with tab_tables:
        st.code("\n\n".join(b.context["authorized_tables"]), language="text", height=360, wrap_lines=True)
    with tab_fields:
        blocks = [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
                  for x in b.context["schema_context"]]
        if blocks:
            st.code("\n\n".join(blocks), language="text", height=360, wrap_lines=True)
        else:
            st.caption("Tu RAG no ha devuelto schema_context.")
    with tab_ctx:
        st.json(b.context, expanded=False)
