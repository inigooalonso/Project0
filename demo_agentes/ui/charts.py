"""Gráficos Plotly del dashboard.

Siguen la guía de visualización: barras finas (≤ 24 px) con el extremo
redondeado, una sola tinta para categorías nominales con énfasis en la primera
posición, líneas de 2 px con punto final y etiqueta, rejilla mínima y sin
doble eje. Las cifras van en formato español.
"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from core.formatting import fmt_month, fmt_number, fmt_tokens, pretty_column
from core.models import ContextBundle, ResultProfile
from core.result_profile import format_measure
from ui.theme import BORDER, CATEGORICAL, DEEMPHASIS, EMPHASIS, MEDIUM_BLUE, TEXT, TEXT_2

CONFIG = {"displayModeBar": False, "responsive": True}
TRANSPARENT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if high <= low:
        return [low]
    raw = (high - low) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=raw)
    start = math.floor(low / step) * step
    ticks = [start]
    while ticks[-1] < high:
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def _tick_label(value: float, kind: str, ticks: list[float]) -> str:
    """Etiquetas de eje limpias: «220 M€» en lugar de «220,0 M€» cuando todas son enteras."""
    if kind == "currency" and all(abs(t) >= 1e6 and abs(t / 1e6 - round(t / 1e6)) < 1e-9 for t in ticks if t):
        return f"{fmt_number(value / 1e6, 0)} M€"
    return format_measure(value, kind)


def result_figure(df: pd.DataFrame, profile: ResultProfile) -> go.Figure | None:
    if profile.kind == "bar":
        return _bar(df, profile)
    if profile.kind == "line":
        return _line(df, profile)
    return None


def _bar(df: pd.DataFrame, profile: ResultProfile) -> go.Figure:
    x, y = profile.x, profile.y
    kind = profile.measure_kinds.get(y, "number")
    data = df[[x, y]].copy()
    values = data[y].astype(float)
    top_value = values.max() if profile.descending else values.min()
    colors = [EMPHASIS if v == top_value else DEEMPHASIS for v in values]
    labels = [format_measure(v, kind) for v in values]
    n = len(data)
    fig = go.Figure(
        go.Bar(
            x=values, y=data[x].astype(str), orientation="h",
            marker=dict(color=colors, cornerradius=4, line=dict(width=0)),
            text=labels, textposition="outside", cliponaxis=False,
            textfont=dict(color=TEXT, size=14),
            customdata=labels,
            hovertemplate="<b>%{y}</b><br>" + pretty_column(y) + ": %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        template="ada", height=max(260, 38 * n + 40), bargap=0.42,
        margin=dict(l=8, r=110, t=8, b=8), showlegend=False, **TRANSPARENT,
    )
    fig.update_xaxes(visible=False, range=[0, values.max() * 1.02] if values.min() >= 0 else None)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=14, color=TEXT), ticksuffix="  ",
                     automargin=True)
    return fig


def _line(df: pd.DataFrame, profile: ResultProfile) -> go.Figure:
    x, y = profile.x, profile.y
    kind = profile.measure_kinds.get(y, "number")
    data = df.copy()
    data[x] = pd.to_datetime(data[x])
    data = data.sort_values(x)
    fig = go.Figure()
    series = [(None, data)] if not profile.series else list(data.groupby(profile.series, sort=False))
    for i, (name, part) in enumerate(series):
        color = MEDIUM_BLUE if name is None else CATEGORICAL[i % len(CATEGORICAL)]
        labels = [format_measure(v, kind) for v in part[y]]
        months = [fmt_month(v) for v in part[x]]
        fig.add_trace(
            go.Scatter(
                x=part[x], y=part[y], mode="lines+markers", name=str(name) if name is not None else pretty_column(y),
                line=dict(color=color, width=2.5), marker=dict(size=7, color=color, line=dict(color="white", width=2)),
                customdata=list(zip(months, labels)),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra>" + ("" if name is None else str(name)) + "</extra>",
            )
        )
        last = part.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last[x]], y=[last[y]], mode="markers+text", showlegend=False, hoverinfo="skip",
                marker=dict(size=11, color=color, line=dict(color="white", width=2)),
                text=[f"  {format_measure(last[y], kind)}"], textposition="middle right",
                textfont=dict(color=TEXT, size=14), cliponaxis=False,
            )
        )
    low, high = float(data[y].min()), float(data[y].max())
    pad = (high - low) * 0.08 or abs(high) * 0.05 or 1
    ticks = _nice_ticks(low - pad, high + pad, count=6)
    fig.update_layout(
        template="ada", height=400, hovermode="x unified", showlegend=bool(profile.series),
        margin=dict(l=8, r=110, t=16, b=8), **TRANSPARENT,
    )
    fig.update_yaxes(range=[ticks[0], ticks[-1]], tickvals=ticks, ticktext=[_tick_label(t, kind, ticks) for t in ticks],
                     automargin=True, showgrid=True, gridcolor="#E9EDF2")
    fig.update_xaxes(
        tickvals=list(data[x].drop_duplicates()), ticktext=[fmt_month(v) for v in data[x].drop_duplicates()],
        showspikes=True, spikemode="across", spikethickness=1, spikecolor="#9AA5B1", spikedash="solid",
        linecolor=BORDER, automargin=True, showline=True,
    )
    return fig


def context_figure(bundle: ContextBundle) -> go.Figure:
    """Composición del contexto enviado al LLM (barra apilada de parte a todo)."""
    blocks = [(name, tokens) for name, tokens in bundle.tokens_by_block.items() if tokens > 0]
    total = sum(t for _, t in blocks) or 1
    fig = go.Figure()
    for i, (name, tokens) in enumerate(blocks):
        share = tokens / total
        fig.add_trace(
            go.Bar(
                x=[tokens], y=[""], orientation="h", name=name,
                marker=dict(color=CATEGORICAL[i % len(CATEGORICAL)], line=dict(color="white", width=2)),
                text=[f"{name} · ≈{fmt_tokens(tokens)}" if share > 0.22 else ""], textposition="inside",
                insidetextanchor="middle", textfont=dict(color="white", size=13),
                hovertemplate=f"<b>{name}</b><br>≈ {fmt_tokens(tokens)} tokens ({share:.0%})<extra></extra>",
            )
        )
    fig.update_layout(
        template="ada", barmode="stack", height=150, bargap=0.25, margin=dict(l=4, r=4, t=36, b=4),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0, traceorder="normal"), **TRANSPARENT,
    )
    fig.update_xaxes(visible=False, range=[0, total])
    fig.update_yaxes(visible=False)
    return fig


def latency_figure(rows: list[tuple[str, float, bool]]) -> go.Figure:
    """Duración de cada paso (barras horizontales; simulados en tono claro)."""
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [DEEMPHASIS if r[2] else MEDIUM_BLUE for r in rows]
    labels = [f"{v:.2f} s".replace(".", ",") for v in values]
    fig = go.Figure(
        go.Bar(x=values, y=names, orientation="h", marker=dict(color=colors, cornerradius=4), text=labels,
               textposition="outside", cliponaxis=False, textfont=dict(color=TEXT_2, size=13),
               hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>")
    )
    fig.update_layout(template="ada", height=34 * len(rows) + 30, bargap=0.45, margin=dict(l=8, r=70, t=6, b=6),
                      showlegend=False, **TRANSPARENT)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=13, color=TEXT_2), ticksuffix="  ",
                     automargin=True)
    return fig
