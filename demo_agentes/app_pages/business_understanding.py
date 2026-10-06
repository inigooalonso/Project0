"""Business Understanding: agente en desarrollo (estado «Próximamente»)."""
from __future__ import annotations

import streamlit as st

from ui.components import esc, page_header, pill
from ui.graphs import bu_flow_dot

head_left, head_right = st.columns([0.75, 0.25], vertical_alignment="bottom")
with head_left:
    page_header(
        "Agente 2 · En desarrollo",
        "Business Understanding",
        "Un agente de RAG agéntico que entiende el contexto de negocio del banco: qué significa cada indicador, "
        "qué política aplica y por qué un dato es como es, citando siempre la fuente.",
    )
with head_right:
    st.html(f'<div style="text-align:right">{pill("Próximamente", "warn", dot=False)}</div>')

st.space("small")
left, right = st.columns([0.42, 0.58], gap="large")
with left:
    st.html('<div class="ada-section">Qué hará</div>')
    capabilities = [
        ("Planifica antes de buscar", "Descompone la pregunta en subpreguntas y decide qué fuentes consultar."),
        ("Busca y contrasta", "Recorre glosario, políticas, normativa interna y documentación de datos; "
                              "si la evidencia no basta, reformula y vuelve a buscar."),
        ("Responde con citas", "Cada afirmación enlaza con su fuente. Si no encuentra respaldo, lo dice."),
    ]
    st.html("".join(
        f'<div class="ada-card accent" style="margin-bottom:0.7rem"><h4>{esc(t)}</h4><p>{esc(d)}</p></div>'
        for t, d in capabilities
    ))
with right:
    st.html('<div class="ada-section">Flujo previsto</div>')
    st.graphviz_chart(bu_flow_dot(), width="content")

st.html('<div class="ada-section">Preguntas que podrá responder</div>')
examples = [
    "¿Cómo se define la tasa de mora y en qué se diferencia del ratio de impagados?",
    "¿Qué criterios se usan para clasificar a un cliente como Pyme?",
    "¿Qué tablas contienen la franquicia de Global Markets y quién es su propietario?",
]
st.html('<div class="ada-tags">' + "".join(
    f'<span class="ada-pill" style="font-size:0.95rem;padding:0.45rem 0.9rem">{esc(e)}</span>' for e in examples
) + "</div>")

st.space("small")
st.html(
    '<div class="ada-soon"><b style="color:#072146">Cómo se complementa con ADA.</b> '
    '<span style="color:#46505C">ADA responde «¿cuánto?» con datos; Business Understanding responderá «¿qué significa, '
    "qué regla aplica y por qué?». Ambos compartirán el catálogo y el glosario de negocio, de modo que una misma "
    "definición se use igual al explicar un concepto que al calcularlo.</span></div>"
)
