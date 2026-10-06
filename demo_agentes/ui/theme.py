"""Tema visual: tokens de la paleta BBVA y CSS propio.

El tema base (colores, radios, tipografía) está en .streamlit/config.toml. Aquí
solo va lo que Streamlit no cubre: tarjetas, etiquetas, stepper y cabeceras.
"""
from __future__ import annotations

import streamlit as st

NAVY = "#072146"
CORE_BLUE = "#004481"
MEDIUM_BLUE = "#1973B8"
LIGHT_BLUE = "#5BBEFF"
AQUA = "#2DCCCD"
DARK_AQUA = "#02A5A5"
SKY = "#D4EDFC"
SKY_SOFT = "#EEF6FD"
ORANGE = "#F7893B"
VIOLET = "#8F7AE5"
MUSTARD = "#D8A013"
PINK = "#E06290"

TEXT = "#121212"
TEXT_2 = "#46505C"
MUTED = "#6B7480"
BORDER = "#E1E6EC"
GRID = "#E9EDF2"
SURFACE = "#FFFFFF"
SURFACE_2 = "#F5F7FA"

SUCCESS = "#2E7D45"
SUCCESS_BG = "#E8F5EC"
WARNING = "#8A6500"
WARNING_BG = "#FDF4DA"
ERROR = "#B92A45"
ERROR_BG = "#FCEBEE"

FONT_STACK = '"Source Sans Pro", "Source Sans 3", "Source Sans", system-ui, -apple-system, "Segoe UI", sans-serif'

# Colores de las etiquetas del IR (identidad por texto + punto de color).
KIND_COLORS = {
    "metric": (MEDIUM_BLUE, "#EAF2FA"),
    "dimension": (DARK_AQUA, "#E5F5F5"),
    "attribute": (VIOLET, "#F1EEFC"),
    "filter": (ORANGE, "#FEF1E7"),
    "time_range": (MUSTARD, "#FBF4E0"),
    "order": (MUTED, "#F1F3F5"),
}


CSS = f"""
<style>
:root {{
  --ada-navy: {NAVY};
  --ada-blue: {CORE_BLUE};
  --ada-blue-2: {MEDIUM_BLUE};
  --ada-sky: {SKY};
  --ada-sky-soft: {SKY_SOFT};
  --ada-text: {TEXT};
  --ada-text-2: {TEXT_2};
  --ada-muted: {MUTED};
  --ada-border: {BORDER};
  --ada-surface-2: {SURFACE_2};
}}

/* Cromo de Streamlit: fuera indicadores que distraen en una demo proyectada */
[data-testid="stStatusWidget"], [data-testid="stDecoration"], [data-testid="stAppDeployButton"] {{ display: none !important; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 4.6rem; padding-bottom: 4rem; max-width: 1680px; }}
h1, h2, h3, h4 {{ color: var(--ada-navy); letter-spacing: -0.01em; }}

/* Tipografía de apoyo */
.ada-eyebrow {{ font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ada-blue-2); margin: 0 0 0.35rem 0; }}
.ada-title {{ font-size: 2.15rem; line-height: 1.15; font-weight: 700; color: var(--ada-navy); margin: 0; }}
.ada-subtitle {{ font-size: 1.12rem; color: var(--ada-text-2); margin: 0.45rem 0 0 0; max-width: 62rem; }}
.ada-muted {{ color: var(--ada-muted); }}
.ada-section {{ font-size: 1.05rem; font-weight: 700; color: var(--ada-navy); margin: 1.1rem 0 0.5rem 0; }}

/* Píldoras y etiquetas */
.ada-pill {{ display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.22rem 0.7rem; border-radius: 999px;
  font-size: 0.82rem; font-weight: 600; border: 1px solid var(--ada-border); color: var(--ada-text-2); background: #fff; white-space: nowrap; }}
.ada-pill .dot {{ width: 0.5rem; height: 0.5rem; border-radius: 50%; background: #B5BDC7; }}
.ada-pill.ok {{ color: {SUCCESS}; background: {SUCCESS_BG}; border-color: transparent; }}
.ada-pill.warn {{ color: {WARNING}; background: {WARNING_BG}; border-color: transparent; }}
.ada-pill.err {{ color: {ERROR}; background: {ERROR_BG}; border-color: transparent; }}
.ada-pill.info {{ color: var(--ada-blue); background: var(--ada-sky-soft); border-color: transparent; }}

.ada-tags {{ display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.4rem 0 0.2rem 0; }}
.ada-tag {{ display: inline-flex; flex-direction: column; gap: 0.1rem; padding: 0.55rem 0.85rem 0.55rem 0.8rem;
  border-radius: 10px; border-left: 4px solid var(--tag-color); background: var(--tag-bg); min-width: 9rem; }}
.ada-tag .kind {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ada-text-2); }}
.ada-tag .text {{ font-size: 1.08rem; font-weight: 600; color: var(--ada-navy); }}
.ada-tag .sub {{ font-size: 0.82rem; color: var(--ada-text-2); }}

/* Tarjetas */
.ada-card {{ border: 1px solid var(--ada-border); border-radius: 14px; padding: 1.1rem 1.25rem; background: #fff; height: 100%; }}
.ada-card h4 {{ margin: 0 0 0.35rem 0; font-size: 1.05rem; }}
.ada-card p {{ margin: 0; color: var(--ada-text-2); }}
.ada-card .code {{ font-family: "Source Code Pro", monospace; font-size: 0.82rem; color: var(--ada-blue); background: var(--ada-sky-soft);
  padding: 0.1rem 0.4rem; border-radius: 6px; display: inline-block; margin: 0.2rem 0.3rem 0 0; }}
.ada-card .formula {{ font-family: "Source Code Pro", monospace; font-size: 0.8rem; color: var(--ada-text-2); margin-top: 0.5rem; }}
.ada-card.accent {{ border-top: 4px solid var(--ada-blue); }}
.ada-card.warn {{ border-top: 4px solid {MUSTARD}; }}

/* KPI */
.ada-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(10.5rem, 1fr)); gap: 0.9rem; margin: 0.6rem 0 1rem 0; }}
.ada-kpi {{ border: 1px solid var(--ada-border); border-radius: 14px; padding: 0.95rem 1.15rem; background: #fff; }}
.ada-kpi .label {{ font-size: 0.86rem; color: var(--ada-text-2); font-weight: 600; }}
.ada-kpi .value {{ font-size: 1.9rem; font-weight: 700; color: var(--ada-navy); line-height: 1.2; margin-top: 0.2rem; overflow-wrap: anywhere; }}
.ada-kpi .caption {{ font-size: 0.85rem; color: var(--ada-muted); margin-top: 0.15rem; }}

/* Escenario: cabecera de paso */
.ada-stage-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; padding-bottom: 0.8rem;
  border-bottom: 1px solid var(--ada-border); margin-bottom: 1rem; }}
.ada-stage-head .num {{ font-size: 0.8rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ada-blue-2); }}
.ada-stage-head .title {{ font-size: 1.6rem; font-weight: 700; color: var(--ada-navy); line-height: 1.2; margin-top: 0.1rem; }}
.ada-stage-head .what {{ color: var(--ada-text-2); margin-top: 0.3rem; font-size: 1rem; max-width: 58rem; }}
.ada-stage-head .why {{ color: var(--ada-blue); margin-top: 0.25rem; font-size: 0.95rem; font-weight: 600; }}
.ada-stage-head .meta {{ display: flex; gap: 0.4rem; flex-wrap: wrap; justify-content: flex-end; }}

.ada-understood {{ font-size: 1.35rem; line-height: 1.45; color: var(--ada-navy); font-weight: 600; margin: 0.2rem 0 0.8rem 0; }}
.ada-question {{ font-size: 1.2rem; color: var(--ada-text-2); border-left: 4px solid var(--ada-sky); padding: 0.2rem 0 0.2rem 0.9rem; margin: 0 0 0.8rem 0; }}
.ada-headline {{ font-size: 1.55rem; line-height: 1.35; font-weight: 700; color: var(--ada-navy); margin: 0.2rem 0 0.4rem 0; }}

/* Banner de paso en curso */
.ada-running {{ position: relative; overflow: hidden; border: 1px solid #CFE3F5; background: var(--ada-sky-soft); border-radius: 14px;
  padding: 0.85rem 1.1rem; margin-bottom: 1rem; }}
.ada-running .title {{ font-weight: 700; color: var(--ada-blue); display: flex; align-items: center; gap: 0.55rem; }}
.ada-running .text {{ color: var(--ada-text-2); margin-top: 0.15rem; }}
.ada-running .lines {{ margin-top: 0.45rem; font-family: "Source Code Pro", monospace; font-size: 0.82rem; color: var(--ada-text-2); }}
.ada-running .lines div:last-child {{ color: var(--ada-blue); font-weight: 600; }}
.ada-running .bar {{ position: absolute; left: 0; bottom: 0; height: 3px; width: 35%; background: var(--ada-blue-2);
  animation: ada-slide 1.3s ease-in-out infinite; }}
.ada-spinner {{ width: 0.95rem; height: 0.95rem; border-radius: 50%; border: 2px solid #B9D7F1; border-top-color: var(--ada-blue);
  animation: ada-spin 0.8s linear infinite; display: inline-block; }}
@keyframes ada-slide {{ 0% {{ left: -35%; }} 100% {{ left: 100%; }} }}
@keyframes ada-spin {{ to {{ transform: rotate(360deg); }} }}

/* Stepper vertical (columna fija) */
div[data-testid="stColumn"]:has(.st-key-ada_stepper) {{ position: sticky; top: 4.6rem; align-self: flex-start; }}
.st-key-ada_stepper {{ gap: 0.35rem; }}
.ada-stepper-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.3rem; }}
.ada-stepper-head .t {{ font-weight: 700; color: var(--ada-navy); font-size: 0.95rem; letter-spacing: 0.02em; }}
.ada-stepper-head .p {{ color: var(--ada-muted); font-size: 0.85rem; }}
[class*="st-key-stp_"] {{ border: 1px solid var(--ada-border); border-left: 4px solid #D5DBE3; border-radius: 12px; padding: 0.35rem 0.6rem 0.5rem 0.55rem;
  background: #fff; gap: 0 !important; transition: background 0.2s ease; }}
[class*="st-key-stp_"] button {{ justify-content: flex-start; padding: 0.1rem 0.2rem; min-height: 0; font-weight: 700; color: var(--ada-navy); }}
[class*="st-key-stp_"] button > div {{ justify-content: flex-start; width: 100%; }}
[class*="st-key-stp_"] button p {{ font-size: 1rem; font-weight: 700; }}
[class*="st-key-stp_"] button:disabled {{ color: #9AA3AE; }}
[class*="st-key-stp_"] button:disabled p {{ color: #9AA3AE; }}
[class*="st-key-stp_"][class*="_done"] {{ border-left-color: var(--ada-blue-2); }}
[class*="st-key-stp_"][class*="_done"] button [data-testid="stIconMaterial"] {{ color: var(--ada-blue-2); }}
[class*="st-key-stp_"][class*="_running"] {{ border-left-color: var(--ada-blue); background: var(--ada-sky-soft); }}
[class*="st-key-stp_"][class*="_running"] button [data-testid="stIconMaterial"] {{ color: var(--ada-blue); animation: ada-spin 1s linear infinite; }}
[class*="st-key-stp_"][class*="_waiting"] {{ border-left-color: {MUSTARD}; background: {WARNING_BG}; }}
[class*="st-key-stp_"][class*="_waiting"] button [data-testid="stIconMaterial"] {{ color: {WARNING}; }}
[class*="st-key-stp_"][class*="_error"] {{ border-left-color: {ERROR}; background: {ERROR_BG}; }}
[class*="st-key-stp_"][class*="_error"] button [data-testid="stIconMaterial"] {{ color: {ERROR}; }}
[class*="st-key-stp_"][class*="_sel"] {{ box-shadow: 0 0 0 2px var(--ada-blue-2) inset; }}
.ada-step-meta {{ font-size: 0.8rem; color: var(--ada-muted); padding-left: 1.85rem; line-height: 1.35; }}
.ada-step-meta .sum {{ color: var(--ada-text-2); display: block; font-size: 0.86rem; }}

/* Ejemplos y barra de pregunta */
.st-key-ada_examples button {{ border-radius: 999px; border-color: #C9DCEE; background: var(--ada-sky-soft); color: var(--ada-blue); font-weight: 600; }}
.st-key-ada_examples button:hover {{ border-color: var(--ada-blue-2); color: var(--ada-navy); }}

/* Aclaraciones: varias preguntas por ronda */
.ada-q {{ display: flex; gap: 0.7rem; align-items: flex-start; margin: 0.9rem 0 0.35rem 0; }}
.ada-q .n {{ flex: none; width: 1.7rem; height: 1.7rem; border-radius: 50%; background: var(--ada-blue); color: #fff;
  font-weight: 700; font-size: 0.9rem; display: flex; align-items: center; justify-content: center; }}
.ada-q .t {{ color: var(--ada-navy); font-size: 1.02rem; line-height: 1.45; }}
.ada-round {{ border: 1px solid var(--ada-border); border-radius: 14px; padding: 0.4rem 1.1rem 0.9rem 1.1rem; margin-bottom: 0.8rem; background: #fff; }}
.ada-round .h {{ font-size: 0.8rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ada-blue-2); margin-top: 0.6rem; }}
.ada-round .a {{ margin: 0.2rem 0 0 2.4rem; color: var(--ada-text-2); border-left: 3px solid var(--ada-sky); padding-left: 0.7rem; }}

/* Tarjeta de error elegante */
.ada-error {{ border: 1px solid #F2C4CD; background: {ERROR_BG}; border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }}
.ada-error .t {{ color: {ERROR}; font-weight: 700; font-size: 1.1rem; }}
.ada-error .m {{ color: var(--ada-text-2); margin-top: 0.2rem; }}
.ada-info {{ border: 1px solid #CFE3F5; background: var(--ada-sky-soft); border-radius: 14px; padding: 0.9rem 1.15rem; color: var(--ada-text-2); }}
.ada-info b {{ color: var(--ada-navy); }}

/* Portada */
.ada-hero {{ background: linear-gradient(120deg, {NAVY} 0%, {CORE_BLUE} 70%, {MEDIUM_BLUE} 100%); border-radius: 20px; padding: 2.6rem 2.8rem;
  color: #fff; margin-bottom: 1.6rem; }}
.ada-hero .ada-eyebrow {{ color: {LIGHT_BLUE}; }}
.ada-hero h1 {{ color: #fff; font-size: 2.6rem; line-height: 1.12; margin: 0; }}
.ada-hero p {{ color: #D9E7F5; font-size: 1.2rem; max-width: 52rem; margin: 0.8rem 0 0 0; }}
.ada-agent-card {{ border: 1px solid var(--ada-border); border-radius: 18px; padding: 1.5rem 1.6rem; background: #fff; }}
.ada-agent-card .name {{ font-size: 1.5rem; font-weight: 700; color: var(--ada-navy); margin: 0.6rem 0 0.4rem 0; }}
.ada-agent-card p {{ color: var(--ada-text-2); margin: 0 0 0.8rem 0; font-size: 1.02rem; }}
.ada-agent-card ul {{ margin: 0; padding-left: 1.1rem; color: var(--ada-text-2); }}
.ada-agent-card li {{ margin: 0.2rem 0; }}
.ada-guarantee {{ display: flex; gap: 0.8rem; align-items: flex-start; }}
.ada-guarantee .ic {{ font-family: "Material Symbols Rounded"; font-size: 1.6rem; color: var(--ada-blue-2); line-height: 1; }}
.ada-guarantee b {{ color: var(--ada-navy); display: block; margin-bottom: 0.15rem; }}
.ada-guarantee span {{ color: var(--ada-text-2); font-size: 0.95rem; }}

/* Próximamente */
.ada-soon {{ border: 1px dashed #B9CDE2; border-radius: 18px; padding: 1.3rem 1.5rem; background: linear-gradient(180deg, #fff 0%, var(--ada-sky-soft) 100%); }}
</style>
"""


def inject_css() -> None:
    st.html(CSS)
