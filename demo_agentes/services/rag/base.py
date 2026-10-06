"""Interfaz del RAG multinivel (paso 2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.models import Pydantic_SemanticQueryIR, RAGResult


class RAGService(Protocol):
    name: str
    simulated: bool

    def search(self, ir: Pydantic_SemanticQueryIR) -> RAGResult:
        """Busca, para cada entidad del IR, propietario → tabla → campo."""
        ...


@dataclass(frozen=True)
class IRItem:
    """Elemento del IR que se busca en el catálogo."""

    id: str
    kind: str  # metric | dimension | attribute | filter | time_range
    surface_form: str
    entity: str
    concept: str
    value: object = None
    is_temporal: bool = False
    is_measure: bool = False


KIND_LABELS = {
    "metric": "Métrica",
    "dimension": "Dimensión",
    "attribute": "Atributo",
    "filter": "Filtro",
    "time_range": "Periodo",
}


def ir_items(ir: Pydantic_SemanticQueryIR) -> list[IRItem]:
    items: list[IRItem] = []
    for m in ir.metrics:
        items.append(IRItem(m.id, "metric", m.surface_form, m.entity, m.concept, is_measure=True))
    for d in ir.dimensions:
        items.append(IRItem(d.id, "dimension", d.surface_form, d.entity, d.concept, is_temporal=d.grain is not None))
    for a in ir.attributes:
        items.append(IRItem(a.id, "attribute", a.surface_form, a.entity, a.concept))
    for f in ir.filters:
        items.append(IRItem(f.id, "filter", f.surface_form, f.entity, f.concept, value=f.value))
    if ir.time_range is not None:
        t = ir.time_range
        items.append(IRItem(t.id, "time_range", t.surface_form, t.entity, t.concept, is_temporal=True))
    return items
