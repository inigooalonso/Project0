"""Grabación de lo que hace AgenticRAG, sin tocar su código.

Se envuelven los tres objetos que el agente recibe o construye:
- el retriever (RagRetrieve): cada llamada guarda los fragmentos devueltos;
- el LLM: cada invoke() guarda el texto, las herramientas pedidas y los tokens;
- las herramientas: cada ejecución abre un ToolEvent al que se asocian los
  fragmentos del retriever.
Cada evento se notifica al momento, así la interfaz pinta el recorrido en vivo.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from services.bu.models import HitView, LLMEvent, ToolEvent

MAX_OUTPUT_CHARS = 4000


class Recorder:
    def __init__(self, on_event: Callable[[LLMEvent | ToolEvent], None] | None = None) -> None:
        self.events: list[LLMEvent | ToolEvent] = []
        self.step = 0
        self.current: ToolEvent | None = None
        self._on_event = on_event

    def emit(self, event: LLMEvent | ToolEvent) -> None:
        self.events.append(event)
        if self._on_event is not None:
            self._on_event(event)


def message_text(message: Any) -> str:
    """Texto de un AIMessage (Converse puede devolver str o una lista de bloques)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    return "".join(
        b if isinstance(b, str) else b.get("text", "")
        for b in content
        if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")
    ).strip()


class RecordingRetriever:
    """Delegado de RagRetrieve que anota los fragmentos en la herramienta en curso."""

    def __init__(self, inner: Any, recorder: Recorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _hits(self, hits: list[Any]) -> list[Any]:
        if self._recorder.current is not None:
            self._recorder.current.hits.extend(HitView.from_hit(h) for h in hits)
        return hits

    def search(self, *args, **kwargs):
        return self._hits(self._inner.search(*args, **kwargs))

    def neighbors(self, *args, **kwargs):
        return self._hits(self._inner.neighbors(*args, **kwargs))

    def section(self, *args, **kwargs):
        return self._hits(self._inner.section(*args, **kwargs))

    def outline(self, *args, **kwargs):
        outline = self._inner.outline(*args, **kwargs)
        if self._recorder.current is not None:
            self._recorder.current.outline = list(outline)
        return outline

    def catalog(self, *args, **kwargs):
        catalog = self._inner.catalog(*args, **kwargs)
        if self._recorder.current is not None:
            self._recorder.current.catalog = catalog
        return catalog


class RecordingTool:
    """Delegado de una herramienta de LangChain: mide y registra cada ejecución."""

    def __init__(self, inner: Any, recorder: Recorder) -> None:
        self._inner = inner
        self._recorder = recorder
        self.name = inner.name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def invoke(self, args: dict[str, Any], *rest, **kwargs):
        event = ToolEvent(step=self._recorder.step, name=self.name, args=dict(args or {}))
        self._recorder.current = event
        started = time.perf_counter()
        try:
            output = self._inner.invoke(args, *rest, **kwargs)
            event.output = str(output)[:MAX_OUTPUT_CHARS]
            return output
        except Exception as exc:  # AgenticRAG lo captura y se lo pasa al modelo como "Error: ..."
            event.status, event.error = "error", str(exc).splitlines()[0][:300] if str(exc) else type(exc).__name__
            raise
        finally:
            event.elapsed_s = time.perf_counter() - started
            self._recorder.current = None
            self._recorder.emit(event)


class RecordingLLM:
    """Delegado del chat model: bind_tools() devuelve un runnable que registra cada invoke()."""

    def __init__(self, inner: Any, recorder: Recorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def bind_tools(self, tools, **kwargs):
        return _RecordingRunnable(self._inner.bind_tools(tools, **kwargs), self._recorder)


class _RecordingRunnable:
    def __init__(self, inner: Any, recorder: Recorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def invoke(self, messages, *args, **kwargs):
        started = time.perf_counter()
        response = self._inner.invoke(messages, *args, **kwargs)
        self._recorder.step += 1
        usage = getattr(response, "usage_metadata", None) or {}
        self._recorder.emit(LLMEvent(
            step=self._recorder.step,
            text=message_text(response),
            tool_calls=[{"name": c["name"], "args": dict(c.get("args") or {}), "id": c.get("id")}
                        for c in (getattr(response, "tool_calls", None) or [])],
            elapsed_s=time.perf_counter() - started,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        ))
        return response
