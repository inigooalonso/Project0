"""Business Understanding: agentic RAG sobre la documentación de negocio (tu código en agents/business_understanding)."""
from __future__ import annotations

import streamlit as st

from core.formatting import fmt_int
from ui.bu import runtime as bu
from ui.bu import views
from ui.components import esc, page_header, pill

service, error = bu.get_service()

head_left, head_right = st.columns([0.72, 0.28], vertical_alignment="bottom")
with head_left:
    page_header(
        "Agente 2 · Disponible",
        "Business Understanding",
        "Agentic RAG sobre la documentación de negocio: el modelo decide qué buscar, lee lo necesario "
        "y responde citando cada fragmento.",
    )
with head_right:
    pills = [pill("Disponible", "ok")]
    if service is not None:
        try:
            if service.exists():
                chunks = sum(i["chunks"] for d in service.catalog().values() for i in d.values())
                pills.append(pill(f"{service.collection} · {fmt_int(chunks)} fragmentos", "info"))
            else:
                pills.append(pill(f"{service.collection} · sin indexar", "warn"))
        except Exception:
            pills.append(pill("Base de conocimiento no disponible", "err"))
    st.html('<div style="display:flex;gap:0.4rem;justify-content:flex-end;flex-wrap:wrap">' + "".join(pills) + "</div>")

with st.expander("Cómo funciona", icon=":material/schema:", expanded=not bu.turns()):
    views.how_it_works(service)

if error is not None:
    st.html(f'<div class="ada-error"><div class="t">{esc(error.title)}</div><div class="m">{esc(error.message)}</div></div>')
    st.button("Reintentar", icon=":material/refresh:", type="primary", key="bu_retry")
    if error.detail:
        with st.expander("Detalle técnico", expanded=True):
            st.code(error.detail, language="text", wrap_lines=True)
    st.stop()

tab_ask, tab_kb, tab_search = st.tabs([":material/forum: Preguntar al agente", ":material/library_books: Base de conocimiento",
                                       ":material/search: Búsqueda directa"])
with tab_ask:
    if not service.exists():
        views.missing_collection(service)
    else:
        views.ask_tab(service)
with tab_kb:
    views.knowledge_tab(service)
with tab_search:
    views.search_tab(service)
