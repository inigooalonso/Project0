"""Configuración de la demo: config/settings.toml + variables de entorno."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "config" / "settings.toml"
MOCK_DIR = ROOT / "data" / "mock"
REAL_DIR = ROOT / "data" / "real"

AGENT_MODES = ("mock", "bedrock")
RAG_MODES = ("mock", "agent")
EXECUTOR_MODES = ("mock", "athena")


@dataclass(frozen=True)
class Settings:
    agent_mode: str = "mock"
    rag_mode: str = "mock"
    executor_mode: str = "mock"
    agent_module: str = "agents.ada_text2sql.agent"
    max_clarifications: int = 3
    dialect: str = "AWS Athena (Trino SQL)"
    sqlglot_dialect: str = "athena"
    database: str = "ho_master"
    workgroup: str = "sandbox"
    max_rows: int = 500
    llm_timeout_s: float = 60.0
    rag_timeout_s: float = 30.0
    executor_timeout_s: float = 120.0
    speed: float = 1.0
    autoplay: bool = True

    @property
    def simulated_services(self) -> list[str]:
        """Nombres legibles de los servicios que usan datos simulados."""
        names = []
        if self.agent_mode == "mock":
            names.append("LLM")
        if self.rag_mode == "mock":
            names.append("RAG")
        if self.executor_mode == "mock":
            names.append("ejecución")
        return names

    def with_overrides(self, **overrides) -> "Settings":
        clean = {k: v for k, v in overrides.items() if v is not None and hasattr(self, k)}
        return replace(self, **clean)


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "si", "sí", "on"}


def _validated(value: str, allowed: tuple[str, ...], default: str) -> str:
    return value if value in allowed else default


def load_settings(path: Path = SETTINGS_FILE) -> Settings:
    raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    services = raw.get("services", {})
    agent = raw.get("agent", {})
    sql = raw.get("sql", {})
    demo = raw.get("demo", {})
    timeouts = raw.get("timeouts", {})
    env = os.environ

    base = Settings()
    return Settings(
        agent_mode=_validated(env.get("ADA_AGENT", services.get("agent", base.agent_mode)), AGENT_MODES, base.agent_mode),
        rag_mode=_validated(env.get("ADA_RAG", services.get("rag", base.rag_mode)), RAG_MODES, base.rag_mode),
        executor_mode=_validated(
            env.get("ADA_EXECUTOR", services.get("executor", base.executor_mode)), EXECUTOR_MODES, base.executor_mode
        ),
        agent_module=env.get("ADA_AGENT_MODULE", agent.get("module", base.agent_module)),
        max_clarifications=int(agent.get("max_clarifications", base.max_clarifications)),
        dialect=sql.get("dialect", base.dialect),
        sqlglot_dialect=sql.get("sqlglot_dialect", base.sqlglot_dialect),
        database=sql.get("database", base.database),
        workgroup=sql.get("workgroup", base.workgroup),
        max_rows=int(sql.get("max_rows", base.max_rows)),
        llm_timeout_s=float(timeouts.get("llm", base.llm_timeout_s)),
        rag_timeout_s=float(timeouts.get("rag", base.rag_timeout_s)),
        executor_timeout_s=float(timeouts.get("executor", base.executor_timeout_s)),
        speed=float(env.get("ADA_SPEED", demo.get("speed", base.speed))),
        autoplay=_env_bool(env["ADA_AUTOPLAY"]) if "ADA_AUTOPLAY" in env else bool(demo.get("autoplay", base.autoplay)),
    )
