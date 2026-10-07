"""Paso 2 · RAG multinivel: candidatos por entidad y tabla unificada.

Las filas que no cumplen el grain se marcan en rojo en las dos tablas.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.pipeline import PipelineRun
from services.rag.tables import COLUMNS, GRAIN_LABEL, SCORE_LABELS, candidate_rows
from ui.components import esc
from ui.theme import ERROR, ERROR_BG

KIND_LABELS = {"metric": "Métrica", "dimension": "Dimensión", "attribute": "Atributo", "filter": "Filtro",
               "time_range": "Periodo"}
COLUMN_ORDER = [label for label, _ in COLUMNS.values()]


def _grain_text(value) -> str:
    if value is True:
        return "Sí"
    if value is False:
        return "No"
    return "—"


def styled(rows: list[dict]) -> "pd.io.formats.style.Styler":
    """Tabla con las similitudes a 2 decimales y en rojo las filas que no cumplen el grain."""
    df = pd.DataFrame(rows)
    ordered = [c for c in COLUMN_ORDER if c in df.columns] + [c for c in df.columns if c not in COLUMN_ORDER]
    df = df[ordered]
    if "Tipo" in df.columns:
        df["Tipo"] = df["Tipo"].map(lambda v: KIND_LABELS.get(v, v))
    fails = df[GRAIN_LABEL].map(lambda v: v is False).tolist() if GRAIN_LABEL in df.columns else [False] * len(df)
    if GRAIN_LABEL in df.columns:
        df[GRAIN_LABEL] = df[GRAIN_LABEL].map(_grain_text)

    def paint(row: pd.Series) -> list[str]:
        css = f"background-color: {ERROR_BG}; color: {ERROR}; font-weight: 600" if fails[row.name] else ""
        return [css] * len(row)

    scores = [c for c in SCORE_LABELS if c in df.columns]
    return (df.style.apply(paint, axis=1)
            .format({c: lambda v: "—" if v is None or pd.isna(v) else f"{v:.2f}".replace(".", ",") for c in scores}))


def render(run: PipelineRun) -> None:
    rag = run.rag
    off_grain = sum(1 for c in rag.candidates if c.meets_grain is False)
    sentence = (f"Ha evaluado {len(rag.candidates)} candidatos para las entidades del pseudocódigo "
                f"y autoriza {len(rag.tables)} {'tabla' if len(rag.tables) == 1 else 'tablas'}.")
    if not rag.candidates:
        sentence = f"Tu RAG autoriza {len(rag.tables)} {'tabla' if len(rag.tables) == 1 else 'tablas'}."
    st.html(f'<p class="ada-understood">{esc(sentence)}</p>')

    st.html('<div class="ada-section">Candidatos por entidad</div>')
    if rag.candidates:
        st.dataframe(styled(candidate_rows(rag.candidates)), hide_index=True, width="stretch")
        if off_grain:
            st.html(f'<p style="color:{ERROR};font-size:0.9rem;margin:0.2rem 0 0 0">En rojo, '
                    f'{off_grain} {"candidato que no cumple" if off_grain == 1 else "candidatos que no cumplen"} el grain.</p>')
    else:
        st.info("Tu RAG todavía no devuelve la tabla de candidatos («rag_candidates» en el contexto). "
                "Formato en el README.", icon=":material/info:")

    st.html('<div class="ada-section">Tabla unificada</div>')
    if rag.unified:
        st.dataframe(styled(rag.unified), hide_index=True, width="stretch")
        if rag.unified_derived:
            st.caption("Tu RAG no aporta la tabla unificada («rag_unified»): se muestra el mejor candidato por entidad, "
                       "primero los que cumplen el grain y después por similitud ponderada.")
    else:
        st.caption("Sin tabla unificada.")

    if rag.tables:
        st.html('<div class="ada-section">Tablas autorizadas</div>')
        st.dataframe(
            pd.DataFrame([{"Tabla": t.name, "Campos": len(t.fields), "Descripción": t.description} for t in rag.tables]),
            hide_index=True, width="stretch",
        )
        empty = [t.short_name for t in rag.tables if not t.fields]
        if empty:
            st.warning(f"Sin campos en schema_context: **{', '.join(empty)}**. El LLM conoce la tabla pero no sus columnas.",
                       icon=":material/table_rows:")
        if rag.unauthorized_field_tables:
            st.warning("schema_context describe campos de tablas que no están en authorized_tables y no se usan: "
                       f"**{', '.join(t.split('.')[-1] for t in rag.unauthorized_field_tables)}**.", icon=":material/block:")
        if rag.duplicated_fields:
            st.caption(f"{rag.duplicated_fields} campos venían repetidos en varios bloques de schema_context y se cuentan una vez.")
        for table in rag.tables:
            if not table.fields:
                continue
            with st.expander(f"Campos de {table.short_name} ({len(table.fields)})", icon=":material/view_column:"):
                st.dataframe(
                    pd.DataFrame([{"Campo": f.name, "Etiqueta": f.label, "Descripción": f.description} for f in table.fields]),
                    hide_index=True, width="stretch", height=min(420, 38 + 35 * len(table.fields)),
                )
    with st.expander("Contexto devuelto por tu RAG (JSON)"):
        st.json(rag.raw_context or {}, expanded=False)
