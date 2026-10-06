"""Ajustes de la demo en la barra lateral (plegada por defecto, discreta)."""
from __future__ import annotations

import streamlit as st

from services.factory import real_mode_availability
from ui import runtime

AGENT_LABELS = {"mock": "Simulado (guion)", "bedrock": "Bedrock · tu agente"}
RAG_LABELS = {"mock": "Catálogo simulado", "agent": "Tu RAG (retrieve_context_for_sql)"}
EXECUTOR_LABELS = {"mock": "DuckDB · datos simulados", "athena": "Amazon Athena"}
SPEEDS = {0.0: "Instantáneo", 0.5: "Rápido", 1.0: "Realista", 1.5: "Pausado"}


def _default(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def render() -> None:
    base = runtime.base_settings()
    _default("cfg_agent", base.agent_mode)
    _default("cfg_rag", base.rag_mode)
    _default("cfg_executor", base.executor_mode)
    _default("cfg_autoplay", base.autoplay)
    _default("cfg_speed", min(SPEEDS, key=lambda s: abs(s - base.speed)))
    available = real_mode_availability()

    with st.sidebar:
        st.markdown("### Ajustes de la demo")
        st.caption("Por defecto vienen de config/settings.toml. Los servicios se fijan al lanzar cada pregunta; "
                   "el avance y el ritmo cambian al momento.")
        st.radio("LLM · pasos 1, 5 y 6", list(AGENT_LABELS), format_func=AGENT_LABELS.get, key="cfg_agent")
        if st.session_state["cfg_agent"] == "bedrock" and not available["bedrock"]:
            st.caption(":orange[Faltan las librerías del modo real: pip install -r requirements-aws.txt]")
        st.radio("RAG · paso 2", list(RAG_LABELS), format_func=RAG_LABELS.get, key="cfg_rag")
        if st.session_state["cfg_rag"] == "agent" and not available["agent_rag"]:
            st.caption(":orange[Faltan las librerías del modo real: pip install -r requirements-aws.txt]")
        st.radio("Ejecución · paso 7", list(EXECUTOR_LABELS), format_func=EXECUTOR_LABELS.get, key="cfg_executor")
        if st.session_state["cfg_executor"] == "athena" and not available["athena"]:
            st.caption(":orange[Falta awswrangler: pip install -r requirements-aws.txt]")
        if st.session_state["cfg_executor"] == "athena" and st.session_state["cfg_rag"] == "mock":
            st.caption(":orange[Athena no contiene las tablas del catálogo simulado: usa tu RAG para consultar Athena.]")

        st.divider()
        st.toggle("Avance automático", key="cfg_autoplay",
                  help="Desactívalo para el modo presentador: el pipeline espera a «Siguiente paso» (tecla AvPág).")
        st.select_slider("Ritmo de la animación", options=list(SPEEDS), format_func=SPEEDS.get, key="cfg_speed",
                         help="Multiplica las latencias simuladas. Los servicios reales siempre van a su velocidad.")
        st.divider()
        st.button("Reiniciar la pregunta en curso", icon=":material/restart_alt:", on_click=runtime.reset_run, width="stretch")
