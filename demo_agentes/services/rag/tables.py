"""Tablas del RAG multinivel: candidatos por entidad y tabla unificada.

Tu ``retrieve_context_for_sql`` las devuelve dentro del diccionario de contexto
como listas de filas (``df.to_dict("records")``):

    "rag_candidates": [{"Entidad": ..., "Tipo": ..., "Tabla": ..., "Cumple Grain": True,
                        "Sim. UUAA": 0.91, "Sim. Tabla": 0.84, "Sim. Campo": 0.77,
                        "Sim. Ponderado": 0.83}, ...],
    "rag_unified":    [{...}, ...]

Los nombres de columna admiten variantes (``cumple_grain``, ``Sim UUAA``,
``sim_weighted``…). La tabla unificada se muestra con las columnas que
traiga; si no la aportas, se calcula aquí con el mejor candidato por entidad.
Ninguna de las dos se envía al LLM: solo se pintan en pantalla.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from core.models import RAGCandidate

CANDIDATE_KEYS = ("rag_candidates", "rag_table", "candidates", "entity_matches")
UNIFIED_KEYS = ("rag_unified", "unified_table", "unified")

# Campo del modelo → (columna en pantalla, variantes aceptadas ya normalizadas).
COLUMNS: dict[str, tuple[str, tuple[str, ...]]] = {
    "entity": ("Entidad", ("entidad", "entity", "surface_form", "concepto")),
    "type": ("Tipo", ("tipo", "type", "kind")),
    "table": ("Tabla", ("tabla", "table", "table_name")),
    "meets_grain": ("Cumple Grain", ("cumple_grain", "grain", "grain_ok", "meets_grain", "cumple_granularidad")),
    "sim_uuaa": ("Sim. UUAA", ("sim_uuaa", "similitud_uuaa", "uuaa_score", "score_uuaa")),
    "sim_table": ("Sim. Tabla", ("sim_tabla", "sim_table", "similitud_tabla", "table_score", "score_tabla")),
    "sim_field": ("Sim. Campo", ("sim_campo", "sim_field", "similitud_campo", "field_score", "score_campo")),
    "sim_weighted": ("Sim. Ponderado", ("sim_ponderado", "sim_ponderada", "sim_weighted", "weighted_score",
                                        "score_ponderado", "similitud_ponderada")),
}
DISPLAY = {field: label for field, (label, _) in COLUMNS.items()}
GRAIN_LABEL = DISPLAY["meets_grain"]
SCORE_LABELS = [DISPLAY[f] for f in ("sim_uuaa", "sim_table", "sim_field", "sim_weighted")]
_ALIASES = {alias: field for field, (_, aliases) in COLUMNS.items() for alias in aliases}

TRUE = {"true", "1", "si", "s", "yes", "y", "ok", "cumple", "x"}
FALSE = {"false", "0", "no", "n", "no_cumple", "nok", ""}


def normalize_key(key: Any) -> str:
    text = unicodedata.normalize("NFKD", str(key)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def canonical(key: Any) -> str | None:
    """Campo del modelo al que corresponde una columna, o None si no es conocida."""
    return _ALIASES.get(normalize_key(key))


def to_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = normalize_key(value)
    if text in TRUE:
        return True
    if text in FALSE:
        return False if text else None
    return None


def to_score(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value).strip().replace("%", "").replace(",", "."))
    except ValueError:
        return None


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):  # {"columna": [valores]} (df.to_dict("list"))
        columns = list(value)
        length = max((len(v) for v in value.values() if isinstance(v, list)), default=0)
        return [{c: (value[c][i] if isinstance(value[c], list) and i < len(value[c]) else None) for c in columns}
                for i in range(length)]
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _first(context: dict[str, Any], keys: tuple[str, ...]) -> Any:
    return next((context[k] for k in keys if context.get(k) is not None), None)


def parse_candidates(context: dict[str, Any]) -> list[RAGCandidate]:
    rows = _rows(_first(context, CANDIDATE_KEYS))
    if not rows:  # también se aceptan en schema_context si traen las columnas
        rows = [r for r in _rows(context.get("schema_context")) if any(canonical(k) == "sim_weighted" for k in r)]
    candidates = []
    for row in rows:
        values: dict[str, Any] = {}
        for key, value in row.items():
            field = canonical(key)
            if field is None or field in values:
                continue
            if field == "meets_grain":
                values[field] = to_bool(value)
            elif field.startswith("sim_"):
                values[field] = to_score(value)
            else:
                values[field] = "" if value is None else str(value)
        candidates.append(RAGCandidate(**values))
    return candidates


def parse_unified(context: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Filas de la tabla unificada con los nombres de pantalla (None si tu RAG no la aporta)."""
    raw = _first(context, UNIFIED_KEYS)
    if raw is None:
        return None
    rows = []
    for row in _rows(raw):
        shown: dict[str, Any] = {}
        for key, value in row.items():
            field = canonical(key)
            if field == "meets_grain":
                shown[GRAIN_LABEL] = to_bool(value)
            elif field and field.startswith("sim_"):
                shown[DISPLAY[field]] = to_score(value)
            elif field:
                shown[DISPLAY[field]] = value
            else:
                shown[str(key)] = value
        rows.append(shown)
    return rows


def derive_unified(candidates: list[RAGCandidate]) -> list[dict[str, Any]]:
    """Mejor candidato por entidad: primero los que cumplen el grain, luego mayor similitud ponderada."""
    best: dict[str, RAGCandidate] = {}
    for c in candidates:
        key = c.entity or c.table
        current = best.get(key)
        rank = (c.meets_grain is not False, c.sim_weighted if c.sim_weighted is not None else -1)
        if current is None or rank > (current.meets_grain is not False,
                                      current.sim_weighted if current.sim_weighted is not None else -1):
            best[key] = c
    return candidate_rows(list(best.values()))


def candidate_rows(candidates: list[RAGCandidate]) -> list[dict[str, Any]]:
    return [{DISPLAY[f]: getattr(c, f) for f in COLUMNS} for c in candidates]
