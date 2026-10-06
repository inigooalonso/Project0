"""Captura de prompts y tokens de las llamadas al LLM sin tocar el agente.

Registra un callback de LangChain mediante una variable de contexto (el mismo
mecanismo que get_usage_metadata_callback). Cualquier llm.invoke() que se
ejecute dentro de ``capture_llm_calls()`` queda registrado, aunque el nodo no
reciba configuración.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook


class LLMCapture(BaseCallbackHandler):
    def __init__(self) -> None:
        self.prompts: list[dict[str, str]] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.has_usage = False

    def on_chat_model_start(self, serialized: dict[str, Any], messages, **kwargs: Any) -> None:
        self.calls += 1
        for batch in messages:
            for message in batch:
                content = message.content if isinstance(message.content, str) else str(message.content)
                self.prompts.append({"role": message.type, "content": content})

    def on_llm_end(self, response, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    self.has_usage = True
                    self.input_tokens += int(usage.get("input_tokens", 0))
                    self.output_tokens += int(usage.get("output_tokens", 0))


_capture: ContextVar[LLMCapture | None] = ContextVar("ada_llm_capture", default=None)
register_configure_hook(_capture, inheritable=True)


@contextmanager
def capture_llm_calls() -> Iterator[LLMCapture]:
    handler = LLMCapture()
    token = _capture.set(handler)
    try:
        yield handler
    finally:
        _capture.reset(token)
