"""Lo que se pinta del agente Business Understanding: pasos, herramientas y fragmentos."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Mismo patrón que usa AgenticRAG para reconocer citas: [documento.md#3]
CITATION_RE = re.compile(r"\[([^\[\]\n]+#\d+)\]")

TOOL_LABELS = {
    "search": "Buscar",
    "read_context": "Leer contexto",
    "document_outline": "Índice del documento",
    "read_section": "Leer sección",
    "list_catalog": "Catálogo",
}


@dataclass(frozen=True)
class HitView:
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

    @property
    def section(self) -> str:
        return self.headings[-1] if self.headings else "(sin cabecera)"

    @classmethod
    def from_hit(cls, hit: Any) -> "HitView":
        return cls(
            chunk_id=str(hit.chunk_id), doc=str(hit.doc), topic=str(hit.topic), title=str(hit.title),
            headings=tuple(hit.headings or ()), level=int(hit.level or 0), position=int(hit.position or 0),
            text=str(hit.text or ""), score=getattr(hit, "score", None),
        )


@dataclass
class LLMEvent:
    """Una llamada al modelo: lo que ha pensado y qué herramientas pide."""

    step: int
    text: str
    tool_calls: list[dict[str, Any]]
    elapsed_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


@dataclass
class ToolEvent:
    """Una herramienta ejecutada, con los fragmentos que ha devuelto el retriever."""

    step: int
    name: str
    args: dict[str, Any]
    status: str = "ok"  # ok | error
    error: str = ""
    hits: list[HitView] = field(default_factory=list)
    outline: list[tuple[int, str, str]] | None = None
    catalog: dict[str, dict[str, dict[str, Any]]] | None = None
    output: str = ""
    elapsed_s: float = 0.0

    @property
    def label(self) -> str:
        return TOOL_LABELS.get(self.name, self.name)


@dataclass
class BUError:
    title: str
    message: str
    detail: str = ""


@dataclass
class Turn:
    """Una pregunta al agente y todo lo que ha hecho para responderla."""

    question: str
    followup: bool = False
    events: list[LLMEvent | ToolEvent] = field(default_factory=list)
    answer: str = ""
    sources: list[str] = field(default_factory=list)  # citas verificadas (AgenticRAG.Answer.sources)
    steps: int = 0
    elapsed_s: float = 0.0
    messages: list[Any] = field(default_factory=list)  # historia para continuar la conversación
    error: BUError | None = None

    @property
    def tool_events(self) -> list[ToolEvent]:
        return [e for e in self.events if isinstance(e, ToolEvent)]

    @property
    def llm_events(self) -> list[LLMEvent]:
        return [e for e in self.events if isinstance(e, LLMEvent)]

    @property
    def cited(self) -> list[str]:
        """Todas las citas del texto, en orden, verificadas o no."""
        return list(dict.fromkeys(CITATION_RE.findall(self.answer)))

    @property
    def seen(self) -> dict[str, HitView]:
        """Fragmentos que el modelo ha visto, con su mejor puntuación."""
        out: dict[str, HitView] = {}
        for event in self.tool_events:
            for hit in event.hits:
                current = out.get(hit.chunk_id)
                if current is None or (hit.score or 0) > (current.score or 0):
                    out[hit.chunk_id] = hit
        return out

    @property
    def documents(self) -> list[str]:
        return list(dict.fromkeys(h.doc for h in self.seen.values()))

    @property
    def input_tokens(self) -> int | None:
        values = [e.input_tokens for e in self.llm_events if e.input_tokens is not None]
        return sum(values) if values else None

    @property
    def output_tokens(self) -> int | None:
        values = [e.output_tokens for e in self.llm_events if e.output_tokens is not None]
        return sum(values) if values else None
