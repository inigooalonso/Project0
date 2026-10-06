"""Paso 3 · Joins y glosario."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.pipeline import PipelineRun
from ui.components import esc
from ui.graphs import joins_dot


def _key_fields(run: PipelineRun) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {t: [] for t in run.knowledge.tables}
    for rule in run.knowledge.joins:
        for table, field in ((rule.left_table, rule.left_field), (rule.right_table, rule.right_field)):
            if field not in fields.setdefault(table, []):
                fields[table].append(field)
    for owner in run.rag.owners:
        for table in owner.tables:
            for f in table.fields:
                if f.selected and table.name in fields and f.name not in fields[table.name] and len(fields[table.name]) < 5:
                    fields[table.name].append(f.name)
    return fields


def _labels(run: PipelineRun) -> dict[str, str]:
    labels = {t.name: t.label for o in run.rag.owners for t in o.tables}
    return labels


def render(run: PipelineRun, technical: bool) -> None:
    k = run.knowledge
    if len(k.tables) <= 1:
        sentence = "Toda la información está en una única tabla: no hace falta unir nada."
    else:
        sentence = (f"Para responder hay que unir {len(k.tables)} tablas mediante {len(k.joins)} "
                    f"{'regla' if len(k.joins) == 1 else 'reglas'} de join definidas por el equipo de datos.")
    st.html(f'<p class="ada-understood">{esc(sentence)}</p>')
    if k.tables:
        st.graphviz_chart(joins_dot(k, _key_fields(run), _labels(run)), width="content")
    for bridge in k.bridge_tables:
        st.info(f"Ha añadido **{bridge.split('.')[-1]}** como tabla puente: las tablas encontradas no se unen "
                "directamente y el camino más corto pasa por ella.", icon=":material/alt_route:")
    for table in k.disconnected_tables:
        st.warning(f"No hay ninguna regla de join para **{table.split('.')[-1]}**. Defínela en joins.yaml.",
                   icon=":material/link_off:")

    st.html('<div class="ada-section">Glosario de negocio aplicado</div>')
    for surface, terms in k.ambiguous_terms.items():
        st.warning(f"«{surface}» tiene {len(terms)} definiciones en el glosario: {', '.join(terms)}. "
                   "El agente lo tendrá en cuenta antes de escribir la SQL.", icon=":material/call_split:")
    if not k.glossary:
        st.caption("Ningún término del glosario aplica a esta pregunta.")
    cards = []
    for term in k.glossary:
        fields = "".join(f'<span class="code">{esc(f.split(".", 1)[-1])}</span>' for f in term.fields)
        formula = f'<div class="formula">{esc(term.formula)}</div>' if term.formula and technical else ""
        matched = f'<div style="margin-top:0.45rem;font-size:0.82rem;color:#6B7480">activado por «{esc(term.matched_by)}»</div>' if term.matched_by else ""
        variant = "warn" if any(term.term in names for names in k.ambiguous_terms.values()) else "accent"
        cards.append(f'<div class="ada-card {variant}"><h4>{esc(term.term)}</h4><p>{esc(term.definition)}</p>'
                     f'<div style="margin-top:0.45rem">{fields}</div>{formula}{matched}</div>')
    if cards:
        st.html('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(18rem,1fr));gap:0.8rem">'
                + "".join(cards) + "</div>")

    if technical and k.joins:
        st.html('<div class="ada-section">Reglas de join (join_rules)</div>')
        st.dataframe(
            pd.DataFrame([{"ON": r.on_clause, "Tipo": r.type, "Cardinalidad": r.cardinality, "Significado": r.description}
                          for r in k.joins]),
            hide_index=True, width="stretch",
        )
        st.caption(f"Fuente: {k.source}")
