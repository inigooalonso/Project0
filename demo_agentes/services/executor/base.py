"""Interfaz del ejecutor de SQL (paso 7)."""
from __future__ import annotations

from typing import Protocol

from core.models import ExecutionResult


class SQLExecutor(Protocol):
    name: str

    def execute(self, sql: str) -> ExecutionResult:
        """Ejecuta una consulta de solo lectura y devuelve el resultado."""
        ...
