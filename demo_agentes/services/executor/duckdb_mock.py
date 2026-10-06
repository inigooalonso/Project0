"""Ejecutor SIMULADO: DuckDB con los datos sintéticos del catálogo.

La SQL que muestra la interfaz es la que se ejecuta: solo se traduce del
dialecto de Athena al de DuckDB con sqlglot. Las tablas se crean a partir del
catálogo, con los mismos nombres físicos y tipos.

La base se genera una vez y se guarda en data/mock/.cache/ (el nombre incluye
una huella del generador y del catálogo), así que los arranques siguientes son
instantáneos. Si la carpeta no se puede escribir, se genera en memoria.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import duckdb

from core.models import ExecutionResult
from core.settings import MOCK_DIR, ROOT
from services.catalog import Catalog, load_catalog
from services.errors import ServiceError, describe_exception
from services.sql_guard import to_duckdb

DUCK_TYPES = {"string": "VARCHAR", "decimal": "DOUBLE", "integer": "BIGINT", "date": "DATE"}
_lock = threading.Lock()


CACHE_DIR = MOCK_DIR / ".cache"


def _fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (ROOT / "data" / "synthetic.py", MOCK_DIR / "catalog.yaml"):
        digest.update(path.read_bytes())
    digest.update(duckdb.__version__.encode())
    return digest.hexdigest()[:16]


def populate(con: duckdb.DuckDBPyConnection, catalog: Catalog) -> None:
    from data.synthetic import generate

    dataset = generate()
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {catalog.database}")
    for table in catalog.tables():
        df = dataset.tables[table.name]  # noqa: F841  (DuckDB lo lee por nombre)
        columns = ", ".join(
            f"CAST({f.name} AS {DUCK_TYPES.get(f.type, 'VARCHAR')}) AS {f.name}" for f in table.fields
        )
        con.execute(f"CREATE TABLE {table.fq_name} AS SELECT {columns} FROM df")


def _cached_path(catalog: Catalog) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{catalog.database}_{_fingerprint()}.duckdb"
    if path.exists():
        return path
    tmp = CACHE_DIR / f"build_{os.getpid()}.duckdb"
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        populate(con, catalog)
    finally:
        con.close()
    tmp.replace(path)
    for old in CACHE_DIR.glob(f"{catalog.database}_*.duckdb"):
        if old != path:
            old.unlink(missing_ok=True)
    return path


@lru_cache(maxsize=1)
def get_mock_database() -> duckdb.DuckDBPyConnection:
    """Base simulada compartida por todo el proceso (se genera una sola vez)."""
    catalog = load_catalog()
    try:
        return duckdb.connect(str(_cached_path(catalog)), read_only=True)
    except (OSError, duckdb.Error):
        con = duckdb.connect(database=":memory:")
        populate(con, catalog)
        return con


class DuckDBMockExecutor:
    name = "DuckDB · datos simulados"
    simulated = True

    def __init__(self, dialect: str = "athena", max_rows: int = 500) -> None:
        self.dialect = dialect
        self.max_rows = max_rows

    def execute(self, sql: str) -> ExecutionResult:
        try:
            duck_sql = to_duckdb(sql, self.dialect)
        except Exception as exc:  # sqlglot no entiende la consulta
            raise ServiceError("executor", "La consulta no se ha podido traducir",
                               "La SQL generada usa una sintaxis que el motor simulado no reconoce.",
                               describe_exception(exc), can_fallback=False) from exc
        start = time.perf_counter()
        try:
            with _lock:
                cursor = get_mock_database().cursor()
            df = cursor.execute(duck_sql).df()
        except duckdb.CatalogException as exc:
            raise ServiceError("executor", "La consulta usa tablas o campos que no existen en los datos simulados",
                               "Los datos simulados solo contienen el catálogo bancario de la demo.",
                               describe_exception(exc), can_fallback=False) from exc
        except duckdb.Error as exc:
            raise ServiceError("executor", "La consulta no se ha podido ejecutar",
                               "El motor simulado ha rechazado la consulta.",
                               describe_exception(exc), can_fallback=False) from exc
        elapsed = time.perf_counter() - start
        truncated = len(df) > self.max_rows
        return ExecutionResult(
            df=df.head(self.max_rows), engine=self.name, elapsed_s=elapsed, row_count=len(df),
            truncated=truncated, executed_sql=duck_sql, simulated=True,
        )
