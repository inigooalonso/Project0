"""Piezas visuales reutilizables (HTML escapado + estilos de ui/theme.py)."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st

from ui.theme import KIND_COLORS


@dataclass
class Kpi:
    label: str
    value: str
    caption: str = ""


def esc(value) -> str:
    return escape(str(value), quote=True)


def pill(text: str, kind: str = "", dot: bool = False) -> str:
    dot_html = '<span class="dot"></span>' if dot else ""
    return f'<span class="ada-pill {kind}">{dot_html}{esc(text)}</span>'


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    sub = f'<p class="ada-subtitle">{esc(subtitle)}</p>' if subtitle else ""
    st.html(f'<div><p class="ada-eyebrow">{esc(eyebrow)}</p><p class="ada-title">{esc(title)}</p>{sub}</div>')


def tag(kind: str, kind_label: str, text: str, sub: str = "") -> str:
    color, bg = KIND_COLORS.get(kind, KIND_COLORS["order"])
    sub_html = f'<span class="sub">{esc(sub)}</span>' if sub else ""
    return (f'<div class="ada-tag" style="--tag-color:{color};--tag-bg:{bg}">'
            f'<span class="kind">{esc(kind_label)}</span><span class="text">{esc(text)}</span>{sub_html}</div>')


def tags(items: list[str]) -> None:
    st.html('<div class="ada-tags">' + "".join(items) + "</div>")


def kpi_row(kpis: list[Kpi]) -> None:
    if not kpis:
        return
    cells = "".join(
        f'<div class="ada-kpi"><div class="label">{esc(k.label)}</div><div class="value">{esc(k.value)}</div>'
        + (f'<div class="caption">{esc(k.caption)}</div>' if k.caption else "")
        + "</div>"
        for k in kpis
    )
    st.html(f'<div class="ada-kpis">{cells}</div>')

