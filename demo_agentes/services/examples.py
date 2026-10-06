"""Preguntas de ejemplo (data/real/examples.yaml).

Solo hace falta la pregunta: el pseudocódigo, la SQL y el resultado los produce
tu agente en tiempo de ejecución.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from core.settings import REAL_DIR
from services.text import normalize

EXAMPLES_PATH = REAL_DIR / "examples.yaml"
DEFAULT_ICON = ":material/play_circle:"


@dataclass(frozen=True)
class Example:
    id: str
    label: str
    icon: str
    question: str


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize(text)).strip("_")[:40] or "ejemplo"


@lru_cache(maxsize=8)
def _parse(path: str, mtime: float) -> tuple[Example, ...]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    examples, seen = [], set()
    for item in raw.get("examples") or []:
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        label = str(item.get("label") or question).strip()
        ident = str(item.get("id") or _slug(label))
        while ident in seen:
            ident += "_"
        seen.add(ident)
        examples.append(Example(ident, label, str(item.get("icon") or DEFAULT_ICON), question))
    return tuple(examples)


def load_real_examples(path: Path = EXAMPLES_PATH) -> tuple[Example, ...]:
    """Se relee automáticamente cuando cambia el fichero (no hace falta reiniciar)."""
    if not path.exists():
        return ()
    return _parse(str(path), path.stat().st_mtime)
