"""Ejecutor REAL: Amazon Athena con awswrangler (tu llamada original)."""
from __future__ import annotations

import time

from core.models import ExecutionResult
from services.errors import ServiceError, ServiceUnavailable, describe_exception, friendly_aws_error
from services.watchdog import run_with_timeout


class AthenaExecutor:
    name = "Amazon Athena"
    simulated = False

    def __init__(self, database: str = "ho_master", workgroup: str = "sandbox", max_rows: int = 500,
                 timeout_s: float = 120.0) -> None:
        self.database = database
        self.workgroup = workgroup
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    def execute(self, sql: str) -> ExecutionResult:
        try:
            import awswrangler as wr
        except ImportError as exc:
            raise ServiceUnavailable("executor", "Athena no está disponible en este equipo",
                                     "Falta la librería awswrangler (requirements-aws.txt).",
                                     describe_exception(exc)) from exc

        DATABASE_MASTER = self.database
        # Athena no necesita el ';' final que añade validate_read_only_sql.
        sql_query = sql.strip().rstrip(";")
        start = time.perf_counter()

        def query():
            return wr.athena.read_sql_query(
                database=DATABASE_MASTER, sql=sql_query, workgroup=self.workgroup, ctas_approach=False
            )

        try:
            df_tables = run_with_timeout(query, timeout=self.timeout_s, service="executor", what="Amazon Athena")
        except ServiceError:
            raise
        except Exception as exc:
            if type(exc).__module__.startswith(("botocore", "boto3")):
                raise friendly_aws_error("executor", exc, "Amazon Athena") from exc
            raise ServiceError("executor", "Athena ha rechazado la consulta",
                               "La consulta no se ha podido ejecutar en Athena.",
                               describe_exception(exc)) from exc
        elapsed = time.perf_counter() - start
        return ExecutionResult(
            df=df_tables.head(self.max_rows), engine=self.name, elapsed_s=elapsed, row_count=len(df_tables),
            truncated=len(df_tables) > self.max_rows, executed_sql=sql_query, simulated=False,
        )
