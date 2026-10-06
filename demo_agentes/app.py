"""Punto de entrada de la demo de agentes de IA.

    cd demo_agentes
    streamlit run app.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Agentes de IA · Demo",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from services.factory import warm_up  # noqa: E402
from ui import settings_panel  # noqa: E402
from ui.theme import inject_css, register_plotly_template  # noqa: E402

inject_css()
register_plotly_template()
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")


@st.cache_resource(show_spinner="Preparando los datos simulados (solo la primera vez)…")
def _warm_up() -> bool:
    try:
        warm_up()
    except Exception:  # la ejecución simulada informará del problema en su paso
        logging.getLogger(__name__).exception("No se ha podido preparar la base simulada")
    return True


_warm_up()
settings_panel.render()

pages = [
    st.Page("app_pages/portada.py", title="Portada", icon=":material/home:", default=True),
    st.Page("app_pages/ada_text2sql.py", title="ADA · Text2SQL", icon=":material/query_stats:", url_path="ada"),
    st.Page("app_pages/business_understanding.py", title="Business Understanding", icon=":material/menu_book:",
            url_path="business-understanding"),
]
page = st.navigation(pages, position="top")

try:
    page.run()
except Exception:  # red de seguridad: nunca una traza delante de la audiencia
    logging.getLogger(__name__).exception("Error no controlado en la página")
    st.html(
        '<div class="ada-error"><div class="t">Algo no ha ido como se esperaba</div>'
        '<div class="m">La demo sigue disponible. Puedes reiniciar la pregunta en curso y continuar.</div></div>'
    )
    from ui import runtime

    st.button("Reiniciar la pregunta en curso", icon=":material/restart_alt:", on_click=runtime.reset_run)
