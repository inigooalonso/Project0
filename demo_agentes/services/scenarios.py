"""Escenarios guionizados del agente simulado (data/mock/scenarios.yaml)."""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from core.models import ClarificationOption, Pydantic_SemanticQueryIR
from core.settings import MOCK_DIR
from services.text import normalize, tokens

MATCH_THRESHOLD = 0.80


@dataclass(frozen=True)
class Scenario:
    id: str
    label: str
    icon: str
    question: str
    aliases: tuple[str, ...]
    semantic_ir: dict
    clarification_question: str | None = None
    options: tuple[ClarificationOption, ...] = ()
    option_keywords: dict[str, tuple[str, ...]] = field(default_factory=dict)
    sql: str | None = None
    sql_variants: dict[str, str] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    assumptions_variants: dict[str, tuple[str, ...]] = field(default_factory=dict)
    explanation: str = ""
    explanation_variants: dict[str, str] = field(default_factory=dict)

    @property
    def has_clarification(self) -> bool:
        return bool(self.clarification_question)

    def ir(self) -> Pydantic_SemanticQueryIR:
        return Pydantic_SemanticQueryIR.model_validate(self.semantic_ir)

    def option_for(self, answer: str) -> str | None:
        """Interpreta la respuesta del usuario (botón o texto libre) como una de las opciones."""
        if not self.options:
            return None
        norm = normalize(answer)
        for opt in self.options:
            if norm == normalize(opt.label) or norm == normalize(opt.value):
                return opt.value
        answer_tokens = set(tokens(answer))
        scored = []
        for opt in self.options:
            keys = set(tokens(" ".join(self.option_keywords.get(opt.value, ())) + " " + opt.label))
            scored.append((len(answer_tokens & keys), opt.value))
        scored.sort(reverse=True)
        return scored[0][1] if scored and scored[0][0] > 0 else self.options[0].value

    def sql_for(self, option: str | None) -> str:
        if option and option in self.sql_variants:
            return self.sql_variants[option]
        if self.sql:
            return self.sql
        return next(iter(self.sql_variants.values()))

    def assumptions_for(self, option: str | None) -> list[str]:
        if option and option in self.assumptions_variants:
            return list(self.assumptions_variants[option])
        return list(self.assumptions)

    def explanation_for(self, option: str | None) -> str:
        if option and option in self.explanation_variants:
            return self.explanation_variants[option]
        return self.explanation


def _parse(raw: dict) -> Scenario:
    clar = raw.get("clarification") or {}
    options = tuple(ClarificationOption(value=o["value"], label=o["label"]) for o in clar.get("options", []))
    return Scenario(
        id=raw["id"],
        label=raw["label"],
        icon=raw.get("icon", ":material/help:"),
        question=raw["question"].strip(),
        aliases=tuple(raw.get("aliases", [])),
        semantic_ir=raw["semantic_ir"],
        clarification_question=(clar.get("question") or "").strip() or None,
        options=options,
        option_keywords={o["value"]: tuple(str(k) for k in o.get("keywords", [])) for o in clar.get("options", [])},
        sql=raw.get("sql"),
        sql_variants=dict(raw.get("sql_variants", {})),
        assumptions=tuple(raw.get("assumptions", [])),
        assumptions_variants={k: tuple(v) for k, v in raw.get("assumptions_variants", {}).items()},
        explanation=(raw.get("explanation") or "").strip(),
        explanation_variants={k: v.strip() for k, v in raw.get("explanation_variants", {}).items()},
    )


@lru_cache(maxsize=2)
def load_scenarios(path: Path = MOCK_DIR / "scenarios.yaml") -> tuple[Scenario, ...]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return tuple(_parse(s) for s in raw["scenarios"])


def match_scenario(question: str, scenarios: tuple[Scenario, ...] | None = None) -> Scenario | None:
    """Asocia una pregunta escrita a mano con un escenario si es casi idéntica."""
    scenarios = scenarios if scenarios is not None else load_scenarios()
    q = normalize(question)
    if not q:
        return None
    best, best_ratio = None, 0.0
    for sc in scenarios:
        for candidate in (sc.question, *sc.aliases):
            ratio = difflib.SequenceMatcher(None, q, normalize(candidate)).ratio()
            if ratio > best_ratio:
                best, best_ratio = sc, ratio
    return best if best_ratio >= MATCH_THRESHOLD else None
