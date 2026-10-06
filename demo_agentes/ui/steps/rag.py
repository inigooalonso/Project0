"""Paso 2 · RAG multinivel: propietario → tabla → campo, con similitud."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.pipeline import PipelineRun
from services.rag.base import KIND_LABELS
from ui.components import esc, tag
from ui.graphs import rag_tree_dot
from ui.theme import BORDER, CORE_BLUE, MEDIUM_BLUE, SKY, SURFACE_2


def _legend(technical: bool) -> str:
    swatches = [(CORE_BLUE, "≥ 0,90"), (MEDIUM_BLUE, "0,80 – 0,90"), ("#9FD3F8", "0,70 – 0,80"), (SKY, "< 0,70")]
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:0.35rem;margin-right:1rem">'
        f'<span style="width:0.9rem;height:0.9rem;border-radius:4px;background:{c}"></span>{esc(t)}</span>'
        for c, t in swatches
    )
    if technical:
        items += (f'<span style="display:inline-flex;align-items:center;gap:0.35rem"><span style="width:0.9rem;height:0.9rem;'
                  f'border-radius:4px;background:{SURFACE_2};border:1px solid {BORDER}"></span>candidato descartado</span>')
    return f'<div style="font-size:0.85rem;color:#46505C;margin:0.2rem 0 0.4rem 0"><b>Similitud</b> &nbsp; {items}</div>'


def render(run: PipelineRun, technical: bool) -> None:
    rag = run.rag
    owners = [o for o in rag.owners if o.selected]
    st.html(
        f'<p class="ada-understood">Ha buscado {len(rag.searches)} entidades del pseudocódigo en tres niveles '
        f'(propietario, tabla y campo) y se queda con {len(rag.selected_tables)} tablas de {len(owners)} '
        f'{"dominio" if len(owners) == 1 else "dominios"} de datos.</p>'
    )
    if not rag.has_scores:
        st.info("Tu RAG todavía no devuelve puntuaciones: el árbol se construye a partir de las tablas autorizadas. "
                "Ver el contrato propuesto en el README.", icon=":material/info:")
    else:
        st.html(_legend(technical))
    st.graphviz_chart(rag_tree_dot(rag, show_candidates=technical, show_scores=technical), width="stretch")

    if not technical:
        st.html('<div class="ada-section">Dónde ha encontrado cada concepto</div>')
        owner_names = {o.code: o.name for o in rag.owners}
        table_labels = {t.name: t.label for o in rag.owners for t in o.tables}
        field_labels = {(t.name, f.name): f.label for o in rag.owners for t in o.tables for f in t.fields}
        rows = []
        for s in rag.searches:
            if not s.table:
                continue
            path = f"{owner_names.get(s.owner, s.owner)} › {table_labels.get(s.table, s.table)} › {field_labels.get((s.table, s.field), s.field)}"
            rows.append(
                f'<div style="display:flex;align-items:center;gap:0.8rem;margin:0.35rem 0">'
                f'{tag(s.kind, KIND_LABELS.get(s.kind, s.kind), s.surface_form)}'
                f'<span style="color:#9AA5B1;font-size:1.3rem">→</span>'
                f'<span style="color:#072146;font-size:1.02rem">{esc(path)}</span></div>'
            )
        st.html("".join(rows))
        return

    st.html('<div class="ada-section">Puntuaciones por entidad</div>')
    data = pd.DataFrame(
        [
            {
                "Entidad": s.surface_form, "Tipo": KIND_LABELS.get(s.kind, s.kind),
                "Propietario": s.owner, "Sim. propietario": s.owner_score,
                "Tabla": (s.table or "").split(".")[-1], "Sim. tabla": s.table_score,
                "Campo": s.field, "Sim. campo": s.field_score, "Candidatos": s.candidates_seen,
            }
            for s in rag.searches
        ]
    )
    score_col = lambda label: st.column_config.ProgressColumn(label, min_value=0.0, max_value=1.0, format="%.2f")  # noqa: E731
    st.dataframe(
        data, hide_index=True, width="stretch",
        column_config={"Sim. propietario": score_col("Sim. propietario"), "Sim. tabla": score_col("Sim. tabla"),
                       "Sim. campo": score_col("Sim. campo")},
    )
    with st.expander("schema_context devuelto al agente"):
        st.json(rag.schema_context, expanded=False)
