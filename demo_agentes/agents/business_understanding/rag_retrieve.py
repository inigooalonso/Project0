"""
Recuperación sobre la colección creada por rag_save.

    from rag_retrieve import RagRetrieve
    r = RagRetrieve()
    for hit in r.search("cómo instalar en linux", k=3, topic="guias"):
        print(hit.chunk_id, hit.path, hit.score)

Además de la búsqueda semántica expone la estructura de los documentos
(temas, índice de cabeceras, secciones completas y fragmentos vecinos), que
es lo que el agente usa para navegar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from rag_common import COLLECTION, bedrock_client, embed, make_filter, match, open_qdrant


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc: str
    topic: str
    title: str
    headings: tuple[str, ...]
    level: int
    position: int
    text: str
    score: float | None = None

    @property
    def path(self) -> str:
        return " > ".join(self.headings)

    @classmethod
    def from_point(cls, point: Any) -> "Hit":
        p = point.payload or {}
        return cls(
            chunk_id=p.get("chunk_id", str(point.id)), doc=p.get("doc", ""), topic=p.get("topic", ""),
            title=p.get("title", ""), headings=tuple(p.get("headings") or ()), level=p.get("level", 0),
            position=p.get("position", 0), text=p.get("text", ""), score=getattr(point, "score", None),
        )


class RagRetrieve:
    def __init__(self, qdrant: QdrantClient | None = None, bedrock: Any = None,
                 collection: str = COLLECTION):
        self.qdrant = qdrant or open_qdrant()
        self.bedrock = bedrock or bedrock_client()
        self.collection = collection

    def exists(self) -> bool:
        return self.qdrant.collection_exists(self.collection)

    # ---- búsqueda semántica -------------------------------------------------

    def search(self, query: str, k: int = 5, topic: str | None = None,
               doc: str | None = None) -> list[Hit]:
        result = self.qdrant.query_points(
            collection_name=self.collection,
            query=embed(self.bedrock, query),
            query_filter=make_filter(topic=topic, doc=doc),
            limit=k,
            with_payload=True,
        )
        return [Hit.from_point(p) for p in result.points]

    # ---- navegación por estructura -----------------------------------------

    def _scroll(self, flt: models.Filter | None) -> list[Hit]:
        hits, offset = [], None
        while True:
            points, offset = self.qdrant.scroll(
                collection_name=self.collection, scroll_filter=flt, limit=256,
                offset=offset, with_payload=True, with_vectors=False,
            )
            hits.extend(Hit.from_point(p) for p in points)
            if offset is None:
                return hits

    def document(self, doc: str) -> list[Hit]:
        """Todos los fragmentos de un documento, en orden."""
        return sorted(self._scroll(make_filter(doc=doc)), key=lambda h: h.position)

    def neighbors(self, chunk_id: str, window: int = 1) -> list[Hit]:
        """El fragmento pedido más los `window` anteriores y posteriores del mismo documento."""
        center = self._scroll(make_filter(chunk_id=chunk_id))
        if not center:
            return []
        c = center[0]
        flt = models.Filter(must=[
            match("doc", c.doc),
            models.FieldCondition(key="position", range=models.Range(gte=c.position - window, lte=c.position + window)),
        ])
        return sorted(self._scroll(flt), key=lambda h: h.position)

    def section(self, doc: str, heading: str) -> list[Hit]:
        """Fragmentos de la sección cuya cabecera contiene `heading`, incluidas sus subsecciones."""
        needle = heading.strip().lower()
        return [h for h in self.document(doc) if any(needle in t.lower() for t in h.headings)]

    def outline(self, doc: str) -> list[tuple[int, str, str]]:
        """Índice del documento: (nivel, ruta de cabeceras, primer chunk_id de esa sección)."""
        seen: dict[tuple[str, ...], tuple[int, str, str]] = {}
        for h in self.document(doc):
            # También las cabeceras padre, aunque no tengan texto propio.
            for depth in range(1, len(h.headings) + 1):
                prefix = h.headings[:depth]
                if prefix not in seen:
                    seen[prefix] = (depth, " > ".join(prefix), h.chunk_id)
        return list(seen.values())

    def catalog(self) -> dict[str, dict[str, dict[str, Any]]]:
        """{tema: {documento: {"title": ..., "chunks": n}}}"""
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for h in self._scroll(None):
            entry = out.setdefault(h.topic, {}).setdefault(h.doc, {"title": h.title, "chunks": 0})
            entry["chunks"] += 1
        return {t: dict(sorted(d.items())) for t, d in sorted(out.items())}