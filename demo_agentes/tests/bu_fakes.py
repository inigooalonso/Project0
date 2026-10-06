"""Entorno sin red para el agente Business Understanding.

Se ejecuta TU código (rag_save, rag_retrieve, agentic_rag) con:
- Qdrant en memoria (QdrantClient(":memory:"));
- un cliente de Bedrock falso que calcula embeddings deterministas por palabras
  (mismo formato de respuesta que Titan v2);
- un LLM guionizado con bind_tools(), que pide herramientas y luego responde.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import unicodedata
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from core.settings import Settings
from services.bu.service import BUService, load_modules

DOCS = Path(__file__).parent / "fixtures" / "bu_docs"
DIM = 1024


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return [w for w in re.findall(r"[a-z0-9]+", text) if len(w) > 2]


class FakeBedrock:
    """invoke_model con la forma de respuesta de Titan v2: {"embedding": [...]}."""

    def invoke_model(self, modelId, contentType, accept, body):
        text = json.loads(body)["inputText"]
        vector = [0.0] * DIM
        for word in _words(text):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        payload = json.dumps({"embedding": [v / norm for v in vector]}).encode()
        return {"body": io.BytesIO(payload)}


class ScriptedLLM:
    """Chat model guionizado. `script` es una lista de funciones (mensajes) -> AIMessage.

    AgenticRAG llama a bind_tools() en cada pregunta: el guion vuelve a empezar.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        bound = ScriptedLLM(self.script)
        self.last = bound
        return bound

    def invoke(self, messages, *args, **kwargs):
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        message = step(messages)
        message.usage_metadata = {"input_tokens": 1000 + 100 * self.calls, "output_tokens": 80, "total_tokens": 0}
        return message


def call(name, args, id_):
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


def first_id(messages, doc_hint: str = "") -> str:
    """Primer chunk_id que aparece en la última respuesta de herramienta."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            ids = re.findall(r"\[([^\[\]\n]+#\d+)\]", str(m.content))
            ids = [i for i in ids if doc_hint in i] or ids
            if ids:
                return ids[0]
    return "desconocido.md#0"


def default_script():
    return [
        lambda m: AIMessage(content="Busco la definición y el proceso de cálculo.", tool_calls=[
            call("search", {"query": "qué es la Franquicia de Distribución y qué mide", "k": 4}, "c1"),
            call("search", {"query": "cálculo de la franquicia", "topic": "procesos"}, "c2"),
        ]),
        lambda m: AIMessage(content="", tool_calls=[
            call("document_outline", {"doc": "procesos/analitica_clientes_europa.md"}, "c3"),
            call("read_section", {"doc": "procesos/analitica_clientes_europa.md", "heading": "Cálculo"}, "c4"),
            call("read_context", {"chunk_id": "no-existe.md#9"}, "c5"),
        ]),
        lambda m: AIMessage(content=(
            "La **Franquicia de Distribución** es el margen de Global Markets atribuible a la red que origina la "
            f"operación [{first_id(m[:4], 'glosario')}]. Se calcula sumando el importe de franquicia resultado de la "
            f"operación por mesa y cliente [{first_id(m, 'europa')}]. "
            "Su umbral de revisión no aparece en la documentación [inventado.md#7]."
        )),
    ]


def index_fixtures(retriever_qdrant, bedrock, collection: str = "rag_md") -> dict[str, int]:
    modules = load_modules()
    saver = modules.save.RagSave(qdrant=retriever_qdrant, bedrock=bedrock, collection=collection)
    return saver.save_directory(DOCS, verbose=False)


def make_service(script=None, indexed: bool = True, settings: Settings | None = None) -> BUService:
    from qdrant_client import QdrantClient

    modules = load_modules()
    qdrant, bedrock = QdrantClient(":memory:"), FakeBedrock()
    if indexed:
        index_fixtures(qdrant, bedrock)
    retriever = modules.retrieve.RagRetrieve(qdrant=qdrant, bedrock=bedrock, collection="rag_md")
    return BUService(retriever, ScriptedLLM(script or default_script()), modules, settings or Settings())
