"""
Configuración y utilidades compartidas por rag_save, rag_retrieve y agentic_rag.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.config import Config
from qdrant_client import QdrantClient, models

REGION = os.getenv("AWS_REGION", "eu-south-2")
LLM_MODEL = os.getenv("RAG_MODEL", "eu.anthropic.claude-sonnet-5") #"eu.anthropic.claude-opus-5-5")
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024
COLLECTION = os.getenv("RAG_COLLECTION", "rag_md")
QDRANT_PATH = os.getenv("RAG_QDRANT_PATH", "./agentic_rag_qdrant_db")


def bedrock_client() -> Any:
    # Reintentos adaptativos: al indexar se hacen muchas llamadas seguidas de embeddings.
    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}),
    )


def open_qdrant(path: str = QDRANT_PATH) -> QdrantClient:
    """Qdrant en modo local. OJO: solo admite UN cliente abierto por carpeta;
    en un mismo proceso (o notebook) crea uno y compártelo."""
    return QdrantClient(path=path)


def embed(bedrock: Any, text: str, model: str = EMBED_MODEL, dim: int = EMBED_DIM) -> list[float]:
    """Embedding con Titan v2. Se usa igual al indexar y al consultar."""
    response = bedrock.invoke_model(
        modelId=model,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": text,
            "dimensions": dim,
            "normalize": True,
            "embeddingTypes": ["float"],
        }),
    )
    return json.loads(response["body"].read())["embedding"]


def match(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def make_filter(**equals: Any) -> models.Filter | None:
    """Filtro AND de igualdades sobre el payload; ignora los valores None."""
    conditions = [match(k, v) for k, v in equals.items() if v is not None]
    return models.Filter(must=conditions) if conditions else None