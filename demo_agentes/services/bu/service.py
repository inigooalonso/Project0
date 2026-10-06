"""Adaptador del agente Business Understanding (agents/business_understanding).

Tu código se usa tal cual: rag_common, rag_retrieve (RagRetrieve) y
agentic_rag (AgenticRAG). Como sus módulos se importan entre sí por nombre
(`from rag_common import ...`), su carpeta se añade a sys.path.

Qdrant en modo local solo admite un cliente abierto por carpeta: la interfaz
crea un único BUService por proceso y lo comparte.
"""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace
from typing import Any, Callable

from core.settings import BU_AGENT_DIR, Settings
from services.bu.models import BUError, HitView, LLMEvent, ToolEvent, Turn
from services.bu.recorder import Recorder, RecordingLLM, RecordingRetriever, RecordingTool
from services.errors import ServiceError, describe_exception, friendly_aws_error


def load_modules() -> SimpleNamespace:
    """Importa tus módulos. Los fallos llegan como ServiceError presentable."""
    path = str(BU_AGENT_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import agentic_rag
        import rag_common
        import rag_retrieve
        import rag_save
    except ImportError as exc:
        raise ServiceError("bu", "No se puede cargar el agente",
                           "Faltan librerías (pip install -r requirements.txt).", describe_exception(exc)) from exc
    return SimpleNamespace(common=rag_common, retrieve=rag_retrieve, agentic=agentic_rag, save=rag_save)


def translate_error(exc: BaseException) -> BUError:
    if isinstance(exc, ServiceError):
        return BUError(exc.title, exc.message, exc.detail)
    text = str(exc)
    if type(exc).__module__.startswith(("botocore", "boto3")):
        error = friendly_aws_error("bu", exc, "Amazon Bedrock")
        return BUError(error.title, error.message, error.detail)
    if "already accessed by another instance" in text:
        return BUError("La base de conocimiento está abierta en otro proceso",
                       "Qdrant local solo admite un cliente por carpeta. Cierra el notebook o el script que la use "
                       "y recarga la página.", describe_exception(exc))
    return BUError("El agente no ha podido completar la respuesta",
                   "Se ha producido un error inesperado.", describe_exception(exc))


class BUService:
    """Fachada sobre tu RagRetrieve y tu AgenticRAG."""

    def __init__(self, retriever: Any, llm: Any, modules: SimpleNamespace, settings: Settings) -> None:
        self.retriever = retriever
        self.llm = llm
        self.modules = modules
        self.settings = settings

    # ---- base de conocimiento ------------------------------------------
    @property
    def collection(self) -> str:
        return self.retriever.collection

    @property
    def model(self) -> str:
        return self.settings.bu_model or self.modules.common.LLM_MODEL

    @property
    def embed_model(self) -> str:
        return f"{self.modules.common.EMBED_MODEL} · {self.modules.common.EMBED_DIM} dim"

    def exists(self) -> bool:
        return self.retriever.exists()

    def catalog(self) -> dict[str, dict[str, dict[str, Any]]]:
        return self.retriever.catalog()

    def search(self, query: str, k: int = 5, topic: str | None = None, doc: str | None = None) -> list[HitView]:
        return [HitView.from_hit(h) for h in self.retriever.search(query, k, topic, doc)]

    def document(self, doc: str) -> list[HitView]:
        return [HitView.from_hit(h) for h in self.retriever.document(doc)]

    def outline(self, doc: str) -> list[tuple[int, str, str]]:
        return list(self.retriever.outline(doc))

    # ---- agente --------------------------------------------------------
    def ask(self, question: str, history: list[Any] | None = None,
            on_event: Callable[[LLMEvent | ToolEvent], None] | None = None) -> Turn:
        """Ejecuta AgenticRAG.ask y devuelve la respuesta con todo el recorrido."""
        recorder = Recorder(on_event)
        turn = Turn(question=question, followup=bool(history))
        started = time.perf_counter()
        try:
            rag = self.modules.agentic.AgenticRAG(
                RecordingRetriever(self.retriever, recorder),
                llm=RecordingLLM(self.llm, recorder),
                max_steps=self.settings.bu_max_steps,
                catalog_in_prompt=self.settings.bu_catalog_in_prompt,
            )
            tools = getattr(rag, "_tools_by_name", None)
            if isinstance(tools, dict):
                rag._tools_by_name = {name: RecordingTool(t, recorder) for name, t in tools.items()}
            answer = rag.ask(question, history=history)
            turn.answer, turn.sources, turn.steps = answer.text, list(answer.sources), answer.steps
            turn.messages = list(answer.messages)
        except Exception as exc:  # nunca una traza delante de la audiencia
            turn.error = translate_error(exc)
        turn.events = recorder.events
        turn.elapsed_s = time.perf_counter() - started
        return turn


def build_bu_service(settings: Settings) -> BUService:
    """Construye el servicio real: Qdrant local, Bedrock y tu LLM."""
    modules = load_modules()
    try:
        qdrant = modules.common.open_qdrant(settings.bu_qdrant_path)
        bedrock = modules.common.bedrock_client()
        retriever = modules.retrieve.RagRetrieve(qdrant=qdrant, bedrock=bedrock, collection=settings.bu_collection)
        llm = modules.agentic.build_llm(settings.bu_model or modules.common.LLM_MODEL)
    except Exception as exc:
        error = translate_error(exc)
        raise ServiceError("bu", error.title, error.message, error.detail) from exc
    return BUService(retriever, llm, modules, settings)
