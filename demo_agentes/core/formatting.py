"""Formato español (es-ES) para cifras, fechas, tiempos y nombres de columna."""
from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
ACCENTS = {
    "direccion": "dirección", "categoria": "categoría", "credito": "crédito", "debito": "débito",
    "prestamos": "préstamos", "prestamo": "préstamo", "numero": "número", "operacion": "operación",
    "operaciones": "operaciones", "region": "región", "periodo": "periodo", "media": "media",
    "medio": "medio", "impagadas": "impagadas", "dias": "días", "codigo": "código", "ano": "año",
    "anio": "año", "captacion": "captación", "comision": "comisión", "transaccion": "transacción",
}


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NaT


def fmt_number(value, decimals: int = 0) -> str:
    if _is_missing(value):
        return "—"
    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", "§").replace(".", ",").replace("§", ".")


def fmt_int(value) -> str:
    return fmt_number(value, 0)


def fmt_eur(value, compact: bool = True) -> str:
    if _is_missing(value):
        return "—"
    v = float(value)
    if compact:
        if abs(v) >= 1e9:
            return f"{fmt_number(v / 1e9, 2)} mil M€"
        if abs(v) >= 1e6:
            return f"{fmt_number(v / 1e6, 1)} M€"
        if abs(v) >= 1e4:
            return f"{fmt_number(v / 1e3, 0)} mil €"
    return f"{fmt_number(v, 0 if abs(v) >= 100 else 2)} €"


def fmt_pct(value, decimals: int = 2) -> str:
    """Valor ya expresado en porcentaje (7.89 → «7,89 %»)."""
    return "—" if _is_missing(value) else f"{fmt_number(value, decimals)} %"


def fmt_change(ratio: float, decimals: int = 1) -> str:
    """Variación relativa (0.128 → «+12,8 %»)."""
    if _is_missing(ratio):
        return "—"
    sign = "+" if ratio >= 0 else "−"
    return f"{sign}{fmt_number(abs(ratio) * 100, decimals)} %"


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{int(round(seconds * 1000))} ms"
    return f"{fmt_number(seconds, 1)} s"


def fmt_tokens(n: int | None) -> str:
    if n is None:
        return "—"
    return f"{fmt_number(n / 1000, 1)} mil" if n >= 1000 else str(int(n))


def as_date(value) -> date | None:
    if _is_missing(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError):
        return None


def fmt_month(value) -> str:
    d = as_date(value)
    return f"{MONTHS[d.month - 1]} {d.year}" if d else str(value)


def fmt_date(value) -> str:
    d = as_date(value)
    return d.strftime("%d/%m/%Y") if d else str(value)


def pretty_column(name: str) -> str:
    """importe_concedido_eur → «Importe concedido (€)»; tasa_mora_pct → «Tasa mora (%)»."""
    words = [w for w in str(name).lower().split("_") if w]
    unit = ""
    if words and words[-1] in {"eur", "euros"}:
        unit, words = " (€)", words[:-1]
    elif words and words[-1] in {"pct", "per", "porcentaje"}:
        unit, words = " (%)", words[:-1]
    if words and words[0] == "pct":
        words = ["%"] + words[1:]
    if words and words[0] in {"num", "n", "nro"}:
        words = ["nº"] + words[1:]
    words = [ACCENTS.get(w, w) for w in words]
    text = " ".join(words) + unit
    return text[:1].upper() + text[1:] if text else name


def is_integral(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return bool(len(values)) and bool(((values - values.round()).abs() < 1e-9).all())
