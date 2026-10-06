"""Lectura de la forma del resultado: qué gráfico, qué KPIs y qué titular.

- 1 fila con medidas → indicadores (sin gráfico).
- columna temporal + medida → líneas.
- categoría + medida (hasta 40 filas) → barras horizontales ordenadas.
- cualquier otra forma → solo tabla.
La medida principal es la columna del ORDER BY de la SQL si es numérica; si no,
el primer importe; si no, la primera medida.
"""
from __future__ import annotations

import pandas as pd

from core.formatting import fmt_change, fmt_eur, fmt_int, fmt_month, fmt_number, fmt_pct, is_integral, pretty_column
from core.models import Kpi, ResultProfile

TIME_NAMES = {"mes", "fecha", "periodo", "dia", "día", "semana", "trimestre", "anio", "año", "cutoff", "month", "date"}
MAX_BAR_ROWS = 40
MAX_SERIES = 6


def measure_kind(name: str) -> str:
    n = name.lower()
    if n.endswith(("_pct", "_per")) or n.startswith("pct_") or any(k in n for k in ("ratio", "tasa", "porcentaje")):
        return "percent"
    if n.endswith(("_eur", "_amount")) or any(k in n for k in ("importe", "saldo", "gasto", "euros")):
        return "currency"
    if n.startswith(("num_", "n_", "nro_")) or any(k in n for k in ("count", "numero", "número", "prestamos_con", "clientes")):
        return "count"
    return "number"


def _is_time(df: pd.DataFrame, col: str) -> bool:
    series = df[col]
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if col.lower() in TIME_NAMES and series.dtype == object:
        return pd.to_datetime(series, errors="coerce").notna().all()
    return False


def _is_id(col: str) -> bool:
    n = col.lower()
    return n.startswith("id_") or n.endswith("_id") or n in {"id", "codigo", "código"}


def profile_result(df: pd.DataFrame, order_by: list[str] | None = None) -> ResultProfile:
    if df is None or df.empty:
        return ResultProfile(kind="table")
    times = [c for c in df.columns if _is_time(df, c)]
    measures = [c for c in df.columns if c not in times and pd.api.types.is_numeric_dtype(df[c]) and not _is_id(c)]
    categories = [c for c in df.columns if c not in times and c not in measures and not _is_id(c)]
    kinds = {c: measure_kind(c) for c in measures}

    order = [c for c in (order_by or []) if c in measures]
    if order:
        y = order[0]
    else:
        y = next((c for c in measures if kinds[c] == "currency"), measures[0] if measures else None)
    descending = True
    if times and not order:
        descending = False

    rows = len(df)
    if y is None:
        return ResultProfile(kind="table", measures=measures, categories=categories, times=times, measure_kinds=kinds)
    if rows == 1:
        kind, x, series = "kpi", None, None
    elif times:
        kind, x = "line", times[0]
        series = next((c for c in categories if df[c].nunique() <= MAX_SERIES), None)
    elif categories and rows <= MAX_BAR_ROWS:
        kind, x, series = "bar", categories[0], None
        values = df[y].tolist()
        descending = values == sorted(values, reverse=True) or values != sorted(values)
    else:
        kind, x, series = "table", None, None
    return ResultProfile(kind=kind, x=x, y=y, series=series, measures=measures, categories=categories,
                         times=times, measure_kinds=kinds, descending=descending)


def percent_decimals(series: pd.Series) -> int:
    """Decimales que realmente tiene una columna de porcentajes (0, 1 o 2)."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    for decimals in (0, 1):
        if len(values) and ((values * 10**decimals - (values * 10**decimals).round()).abs() < 1e-9).all():
            return decimals
    return 2


def format_measure(value, kind: str, compact: bool = True, integral: bool = False, decimals: int = 2) -> str:
    if kind == "currency":
        return fmt_eur(value, compact=compact)
    if kind == "percent":
        return fmt_pct(value, decimals)
    if kind == "count" or integral:
        return fmt_int(value)
    return fmt_number(value, 2)


def _label(col: str) -> str:
    return pretty_column(col).replace(" (€)", "").replace(" (%)", "")


def _total_label(col: str) -> str:
    label = _label(col)
    return label if "total" in label.lower() else f"{label} · total"


def build_kpis(df: pd.DataFrame, profile: ResultProfile) -> list[Kpi]:
    if df is None or df.empty or profile.y is None:
        return []
    y, kinds = profile.y, profile.measure_kinds
    kind = kinds.get(y, "number")
    fmt = lambda col, v: format_measure(v, kinds.get(col, "number"), integral=is_integral(df[col]))  # noqa: E731

    if profile.kind == "kpi":
        row = df.iloc[0]
        return [Kpi(_label(c), fmt(c, row[c])) for c in profile.measures[:3]]

    if profile.kind == "line":
        data = df.sort_values(profile.x)
        if profile.series:
            data = data.groupby(profile.x, as_index=False)[y].sum()
        first, last = data.iloc[0], data.iloc[-1]
        peak = data.loc[data[y].idxmax()]
        kpis = [Kpi(f"{_label(y)} · último dato", fmt(y, last[y]), fmt_month(last[profile.x]))]
        if first[y]:
            kpis.append(Kpi("Variación en el periodo", fmt_change(last[y] / first[y] - 1), f"desde {fmt_month(first[profile.x])}"))
        kpis.append(Kpi("Máximo", fmt(y, peak[y]), fmt_month(peak[profile.x])))
        return kpis

    if profile.kind == "bar":
        x = profile.x
        top = df.loc[df[y].idxmax()] if profile.descending else df.loc[df[y].idxmin()]
        kpis = [Kpi("Primera posición", str(top[x]), fmt(y, top[y]))]
        if kind == "percent":
            kpis.append(Kpi(f"{_label(y)} · media", fmt_pct(df[y].mean()), f"{len(df)} resultados"))
        else:
            kpis.append(Kpi(_total_label(y), fmt(y, df[y].sum()), f"{len(df)} resultados"))
        share = next((c for c in profile.measures if c != y and kinds.get(c) == "percent"), None)
        secondary = next((c for c in profile.measures if c != y and kinds.get(c) == "currency"), None)
        if share is not None and kind != "percent":
            kpis.append(Kpi("Peso de la primera posición", fmt_pct(top[share], 1), _label(share)))
        elif secondary is not None:
            kpis.append(Kpi(_total_label(secondary), fmt(secondary, df[secondary].sum()), f"{len(df)} resultados"))
        else:
            count = next((c for c in profile.measures if c != y and kinds.get(c) == "count"), None)
            if count is not None:
                kpis.append(Kpi(_total_label(count), fmt(count, df[count].sum()), f"{len(df)} resultados"))
        return kpis[:3]

    return [Kpi("Filas", fmt_int(len(df)))]


def headline(df: pd.DataFrame, profile: ResultProfile) -> str | None:
    """Titular en lenguaje llano con el dato principal."""
    if df is None or df.empty or profile.y is None:
        return None
    y, kinds = profile.y, profile.measure_kinds
    kind = kinds.get(y, "number")
    label = _label(y).lower()
    if profile.kind == "kpi":
        return f"{_label(y)}: {format_measure(df.iloc[0][y], kind)}."
    if profile.kind == "line":
        data = df.sort_values(profile.x)
        if profile.series:
            data = data.groupby(profile.x, as_index=False)[y].sum()
        first, last = data.iloc[0], data.iloc[-1]
        change = f" ({fmt_change(last[y] / first[y] - 1)})" if first[y] else ""
        return (f"El {label} pasa de {format_measure(first[y], kind)} en {fmt_month(first[profile.x])} "
                f"a {format_measure(last[y], kind)} en {fmt_month(last[profile.x])}{change}.")
    if profile.kind == "bar":
        x = profile.x
        top = df.loc[df[y].idxmax()] if profile.descending else df.loc[df[y].idxmin()]
        if kind == "percent":
            return f"{top[x]} encabeza el resultado, con una {label} del {format_measure(top[y], kind)}."
        share = next((c for c in profile.measures if c != y and kinds.get(c) == "percent"), None)
        tail = f", el {fmt_pct(top[share], 1)} del total" if share is not None else ""
        noun = label.replace(" total", "")
        return f"{top[x]} encabeza el resultado con {format_measure(top[y], kind)} de {noun}{tail}."
    return None
