"""Joins y glosario definidos a mano en YAML (paso 3).

- Joins: busca el camino más corto en el grafo de reglas que conecta las tablas
  encontradas por el RAG y añade como «tabla puente» las intermedias.
- Glosario: incorpora los términos cuyo nombre o sinónimos aparecen en el IR y
  detecta cuando un mismo concepto tiene varias definiciones (ambigüedad).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Protocol

import yaml

from core.models import GlossaryTerm, JoinRule, KnowledgeResult, Pydantic_SemanticQueryIR, RAGResult
from services.rag.base import ir_items
from services.text import normalize, raw_similarity

GLOSSARY_MIN = 0.62
MAX_TERMS = 6


class KnowledgeService(Protocol):
    name: str

    def resolve(self, ir: Pydantic_SemanticQueryIR, rag: RAGResult) -> KnowledgeResult:
        ...


def _split(qualified: str) -> tuple[str, str]:
    table, _, column = qualified.rpartition(".")
    return table, column


class YamlKnowledgeService:
    def __init__(self, joins_path: Path, glossary_path: Path, name: str = "Reglas y glosario (YAML)") -> None:
        self.name = name
        self.joins = self._load_joins(joins_path)
        self.terms = self._load_terms(glossary_path)

    @staticmethod
    def _load_joins(path: Path) -> list[JoinRule]:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        rules = []
        for j in raw.get("joins") or []:
            lt, lf = _split(j["left"])
            rt, rf = _split(j["right"])
            rules.append(JoinRule(left_table=lt, left_field=lf, right_table=rt, right_field=rf,
                                  type=j.get("type", "INNER"), cardinality=str(j.get("cardinality", "N:1")),
                                  description=j.get("description", "")))
        return rules

    @staticmethod
    def _load_terms(path: Path) -> list[GlossaryTerm]:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return [GlossaryTerm(**t) for t in raw.get("terms") or []]

    # ------------------------------------------------------------------
    def resolve(self, ir: Pydantic_SemanticQueryIR, rag: RAGResult) -> KnowledgeResult:
        tables, bridges, disconnected, joins = self._connect(rag.selected_tables)
        glossary, ambiguous = self._glossary(ir, tables)
        return KnowledgeResult(tables=tables, bridge_tables=bridges, disconnected_tables=disconnected,
                               joins=joins, glossary=glossary, ambiguous_terms=ambiguous, source=self.name)

    def _connect(self, wanted: list[str]) -> tuple[list[str], list[str], list[str], list[JoinRule]]:
        if not wanted:
            return [], [], [], []
        adjacency: dict[str, list[tuple[str, JoinRule]]] = {}
        for rule in self.joins:
            adjacency.setdefault(rule.left_table, []).append((rule.right_table, rule))
            adjacency.setdefault(rule.right_table, []).append((rule.left_table, rule))

        connected = [wanted[0]]
        used: list[JoinRule] = []
        disconnected: list[str] = []
        for target in wanted[1:]:
            if target in connected:
                continue
            path = self._shortest_path(target, set(connected), adjacency)
            connected.append(target)
            if path is None:
                disconnected.append(target)
                continue
            for node, rule in path:
                if node not in connected:
                    connected.append(node)
                if rule not in used:
                    used.append(rule)
        bridges = [t for t in connected if t not in wanted]
        return connected, bridges, disconnected, used

    @staticmethod
    def _shortest_path(start: str, goals: set[str], adjacency) -> list[tuple[str, JoinRule]] | None:
        """BFS desde `start` hasta cualquier tabla ya conectada: [(tabla alcanzada, regla usada)]."""
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            if node in goals:
                return path
            for nxt, rule in adjacency.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + [(nxt, rule)]))
        return None

    def _glossary(self, ir: Pydantic_SemanticQueryIR, tables: list[str]) -> tuple[list[GlossaryTerm], dict[str, list[str]]]:
        signals = []
        for item in ir_items(ir):
            signals.append(item.surface_form)
            signals.append(item.concept)
            if isinstance(item.value, str):
                signals.append(item.value)
        signals += list(ir.unresolved_concepts)

        found: list[GlossaryTerm] = []
        by_signal: dict[str, list[str]] = {}
        for term in self.terms:
            names = (term.term, *term.synonyms)
            best, best_signal = 0.0, None
            for signal in signals:
                for name in names:
                    if normalize(name) == normalize(signal):
                        score = 1.0
                    else:
                        score = raw_similarity(signal, name, "", (name,))
                    if score > best:
                        best, best_signal = score, signal
            relevant = any(f.rpartition(".")[0] in tables for f in term.fields)
            if best >= GLOSSARY_MIN and (relevant or best >= 0.95):
                found.append(term.model_copy(update={"matched_by": best_signal, "score": round(best, 2)}))
                by_signal.setdefault(normalize(best_signal), []).append(term.term)

        found.sort(key=lambda t: -(t.score or 0))
        found = found[:MAX_TERMS]
        ambiguous = {}
        for term in found:
            key = normalize(term.matched_by or "")
            names = [t.term for t in found if normalize(t.matched_by or "") == key]
            if len(names) > 1:
                ambiguous[term.matched_by] = names
        return found, ambiguous

