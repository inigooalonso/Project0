"""Configuración de la demo: config/settings.toml + variables de entorno."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "config" / "settings.toml"
REAL_DIR = ROOT / "data" / "real"


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
    )
