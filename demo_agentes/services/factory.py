"""Construye los servicios a partir de la configuración.

Es el único punto que conoce las clases concretas: la interfaz gráfica y la
orquestación trabajan contra las interfaces de services/*/base.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.settings import REAL_DIR, Settings
from services.agent.base import AgentService
from services.executor.base import SQLExecutor
from services.knowledge.yaml_store import KnowledgeService, YamlKnowledgeService
from services.rag.base import RAGService


@dataclass
class Services:
    agent: AgentService
    rag: RAGService
    knowledge: KnowledgeService
    executor: SQLExecutor


def build_services(settings: Settings) -> Services:
    from services.agent.langgraph_adapter import LangGraphAgentAdapter
    from services.executor.athena import AthenaExecutor
    from services.rag.agent_adapter import AgentRAGAdapter

    return Services(
        agent=LangGraphAgentAdapter(settings.agent_module, settings.llm_timeout_s,
                                    getattr(settings, "clarification_mode", "node")),
        rag=AgentRAGAdapter(settings.agent_module, settings.dialect, settings.rag_timeout_s),
        knowledge=YamlKnowledgeService(REAL_DIR / "joins.yaml", REAL_DIR / "glossary.yaml"),
        executor=AthenaExecutor(settings.database, settings.workgroup, settings.max_rows, settings.executor_timeout_s),
    )
