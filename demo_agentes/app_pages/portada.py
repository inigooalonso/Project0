"""Portada: qué son los agentes y por qué son fiables."""
from __future__ import annotations

import streamlit as st

from ui.components import esc, pill

st.html(
    '<div class="ada-hero"><p class="ada-eyebrow">Inteligencia artificial aplicada al dato</p>'
    "<h1>Agentes de IA que responden<br/>con los datos del banco</h1>"
    "<p>Preguntas de negocio en lenguaje natural y respuestas trazables, dentro del perímetro de datos "
    "autorizado. Un agente ya convierte preguntas en consultas; el siguiente entenderá el contexto de negocio.</p>"
    "</div>"
)

left, right = st.columns(2, gap="large")
with left:
    with st.container(border=True, key="card_ada"):
        st.html(
            f'<div>{pill("Disponible", "ok")}</div><div class="ada-agent-card" style="border:0;padding:0">'
            '<div class="name">ADA · Text2SQL</div>'
            "<p>Convierte una pregunta de negocio en una consulta SQL validada, la ejecuta y explica el resultado.</p>"
            "<ul><li>Entiende la pregunta y la formaliza antes de tocar un dato.</li>"
            "<li>Encuentra los datos en el catálogo: UUAA, tabla y campo.</li>"
            "<li>Pregunta cuando una definición es ambigua, con todas sus dudas a la vez.</li>"
            "<li>Muestra cada paso, de la pregunta al resultado de la base de datos.</li></ul></div>"
        )
        st.page_link("app_pages/ada_text2sql.py", label="Abrir ADA", icon=":material/arrow_forward:")
with right:
    with st.container(border=True, key="card_bu"):
        st.html(
            f'<div>{pill("Próximamente", "warn")}</div><div class="ada-agent-card" style="border:0;padding:0">'
            '<div class="name">Business Understanding</div>'
            "<p>Agentic RAG que entiende el contexto de negocio: definiciones, políticas y procesos, citando siempre las fuentes.</p>"
            "<ul><li>Planifica la búsqueda y la repite si falta evidencia.</li>"
            "<li>Consulta glosario, normativa interna y documentación de datos.</li>"
            "<li>Responde con citas y reconoce lo que no sabe.</li>"
            "<li>Comparte catálogo y glosario con ADA.</li></ul></div>"
        )
        st.page_link("app_pages/business_understanding.py", label="Ver qué hará", icon=":material/arrow_forward:")

st.html('<div class="ada-section" style="font-size:1.25rem;margin-top:1.8rem">Por qué es fiable</div>')
guarantees = [
    ("lock", "Solo lectura", "Toda consulta se valida antes de ejecutarse: cualquier operación de escritura se bloquea."),
    ("verified_user", "Solo datos autorizados", "El modelo solo ve las tablas del catálogo que el buscador autoriza para esa pregunta."),
    ("forum", "Pregunta ante la duda", "Si una definición es ambigua, pide aclaración en lugar de suponer."),
    ("route", "Trazable de principio a fin", "Cada paso queda a la vista: qué entendió, qué buscó, qué escribió y qué ejecutó."),
]
cols = st.columns(4, gap="medium")
for col, (icon, title, text) in zip(cols, guarantees):
    with col:
        st.html(f'<div class="ada-guarantee"><span class="ic">{icon}</span><div><b>{esc(title)}</b>'
                f"<span>{esc(text)}</span></div></div>")

