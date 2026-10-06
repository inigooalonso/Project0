"""Selección de implementaciones (mock o real) según la configuración.

Es el único punto que conoce las clases concretas: la interfaz gráfica y la
orquestación trabajan contra las interfaces de services/*/base.py.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from core.settings import MOCK_DIR, REAL_DIR, Settings
from services.agent.base import AgentService
from services.catalog import Catalog, load_catalog
from services.executor.base import SQLExecutor
from services.knowledge.yaml_store import KnowledgeService, YamlKnowledgeService
from services.rag.base import RAGService


@dataclass
class Services:
    agent: AgentService
    rag: RAGService
    knowledge: KnowledgeService
    executor: SQLExecutor
    catalog: Catalog | None  # catálogo simulado, solo cuando el RAG es el simulado


def build_services(settings: Settings) -> Services:
    if settings.agent_mode == "bedrock":
        from services.agent.langgraph_adapter import LangGraphAgentAdapter

        agent: AgentService = LangGraphAgentAdapter(settings.agent_module, settings.llm_timeout_s)
    else:
        from services.agent.scripted import ScriptedAgent

        agent = ScriptedAgent(max_clarifications=settings.max_clarifications)

    catalog = None
    if settings.rag_mode == "agent":
        from services.rag.agent_adapter import AgentRAGAdapter

        rag: RAGService = AgentRAGAdapter(settings.agent_module, settings.dialect, settings.rag_timeout_s)
        knowledge = YamlKnowledgeService(REAL_DIR / "joins.yaml", REAL_DIR / "glossary.yaml")
    else:
        from services.rag.mock import MockRAGService

        catalog = load_catalog()
        rag = MockRAGService(catalog, settings.dialect)
        knowledge = YamlKnowledgeService(MOCK_DIR / "joins.yaml", MOCK_DIR / "glossary.yaml")

    if settings.executor_mode == "athena":
        from services.executor.athena import AthenaExecutor

        executor: SQLExecutor = AthenaExecutor(settings.database, settings.workgroup, settings.max_rows,
                                               settings.executor_timeout_s)
    else:
        from services.executor.duckdb_mock import DuckDBMockExecutor

        executor = DuckDBMockExecutor(settings.sqlglot_dialect, settings.max_rows)
    return Services(agent=agent, rag=rag, knowledge=knowledge, executor=executor, catalog=catalog)


def real_mode_availability() -> dict[str, bool]:
    """Qué servicios reales tienen sus librerías instaladas en este equipo."""
    def has(*modules: str) -> bool:
        return all(importlib.util.find_spec(m) is not None for m in modules)

    return {
        "bedrock": has("boto3", "langchain_aws", "langgraph", "langchain_core"),
        "agent_rag": has("boto3", "langchain_aws", "langgraph", "langchain_core"),
        "athena": has("awswrangler"),
    }


def warm_up() -> None:
    """Prepara la base simulada antes de la primera pregunta."""
    from services.executor.duckdb_mock import get_mock_database

    get_mock_database()
