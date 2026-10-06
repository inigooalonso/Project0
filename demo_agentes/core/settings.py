"""Configuración de la demo: config/settings.toml + variables de entorno."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "config" / "settings.toml"
REAL_DIR = ROOT / "data" / "real"
BU_AGENT_DIR = ROOT / "agents" / "business_understanding"


@dataclass(frozen=True)
class Settings:
    agent_module: str = "agents.ada_text2sql.agent"
    max_clarification_rounds: int = 3
    max_questions_per_round: int = 5
    dialect: str = "AWS Athena (Trino SQL)"
    sqlglot_dialect: str = "athena"
    database: str = "ho_master"
    workgroup: str = "sandbox"
    max_rows: int = 500
    llm_timeout_s: float = 60.0
    rag_timeout_s: float = 30.0
    executor_timeout_s: float = 120.0
    autoplay: bool = True
    # Business Understanding (agentic RAG sobre Qdrant)
    bu_qdrant_path: str = str(ROOT / "agentic_rag_qdrant_db")
    bu_collection: str = "rag_md"
    bu_model: str = ""  # vacío = el LLM_MODEL de rag_common.py
    bu_max_steps: int = 10
    bu_catalog_in_prompt: bool = True

    def with_overrides(self, **overrides) -> "Settings":
        clean = {k: v for k, v in overrides.items() if v is not None and hasattr(self, k)}
        return replace(self, **clean)


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "si", "sí", "on"}


def load_settings(path: Path = SETTINGS_FILE) -> Settings:
    raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    agent = raw.get("agent", {})
    sql = raw.get("sql", {})
    demo = raw.get("demo", {})
    timeouts = raw.get("timeouts", {})
    business = raw.get("business", {})
    env = os.environ

    base = Settings()
    return Settings(
        agent_module=env.get("ADA_AGENT_MODULE", agent.get("module", base.agent_module)),
        max_clarification_rounds=int(agent.get("max_clarification_rounds", base.max_clarification_rounds)),
        max_questions_per_round=int(agent.get("max_questions_per_round", base.max_questions_per_round)),
        dialect=sql.get("dialect", base.dialect),
        sqlglot_dialect=sql.get("sqlglot_dialect", base.sqlglot_dialect),
        database=sql.get("database", base.database),
        workgroup=sql.get("workgroup", base.workgroup),
        max_rows=int(sql.get("max_rows", base.max_rows)),
        llm_timeout_s=float(timeouts.get("llm", base.llm_timeout_s)),
        rag_timeout_s=float(timeouts.get("rag", base.rag_timeout_s)),
        executor_timeout_s=float(timeouts.get("executor", base.executor_timeout_s)),
        autoplay=_env_bool(env["ADA_AUTOPLAY"]) if "ADA_AUTOPLAY" in env else bool(demo.get("autoplay", base.autoplay)),
        # Mismas variables de entorno que usa rag_common.py.
        bu_qdrant_path=_resolve(env.get("RAG_QDRANT_PATH", business.get("qdrant_path", base.bu_qdrant_path))),
        bu_collection=env.get("RAG_COLLECTION", business.get("collection", base.bu_collection)),
        bu_model=env.get("RAG_MODEL", business.get("model", base.bu_model)),
        bu_max_steps=int(business.get("max_steps", base.bu_max_steps)),
        bu_catalog_in_prompt=bool(business.get("catalog_in_prompt", base.bu_catalog_in_prompt)),
    )


def _resolve(path: str) -> str:
    """Rutas relativas respecto a demo_agentes/, no al directorio desde el que se lanza."""
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else ROOT / p)
