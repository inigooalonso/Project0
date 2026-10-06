"""Inspección y validación de la SQL con sqlglot.

Implementa lo que sugiere el comentario de tu validate_read_only_sql
("Optionally add sqlglot parsing and table allow-list validation here"):
una única sentencia, solo lectura y solo tablas autorizadas.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

from core.models import SQLInspection

WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Grant, exp.Command,
)


def _table_name(table: exp.Table) -> str:
    parts = [p for p in (table.catalog, table.db, table.name) if p]
    return ".".join(parts)


def inspect_sql(sql: str, authorized_tables: list[str], dialect: str = "athena") -> SQLInspection:
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as exc:
        return SQLInspection(parsed=False, parse_error=str(exc).splitlines()[0][:300], read_only=False)
    if not statements:
        return SQLInspection(parsed=False, parse_error="La consulta está vacía.", read_only=False)

    stmt = statements[0]
    cte_names = {cte.alias_or_name for cte in stmt.find_all(exp.CTE)}
    tables = []
    for table in stmt.find_all(exp.Table):
        name = _table_name(table)
        if name and table.name not in cte_names and name not in tables:
            tables.append(name)
    authorized = {t.lower() for t in authorized_tables}
    unauthorized = [t for t in tables if t.lower() not in authorized]
    read_only = isinstance(stmt, (exp.Select, exp.Union, exp.Except, exp.Intersect)) and not any(
        isinstance(node, WRITE_NODES) for node in stmt.walk()
    )
    order = stmt.args.get("order")
    order_by = [o.this.alias_or_name or o.this.sql() for o in order.expressions] if order else []
    return SQLInspection(
        parsed=True,
        read_only=read_only,
        single_statement=len(statements) == 1,
        tables=tables,
        unauthorized_tables=unauthorized,
        join_count=len(list(stmt.find_all(exp.Join))),
        cte_count=len(cte_names),
        has_limit=stmt.args.get("limit") is not None,
        order_by=order_by,
    )


def validate_read_only(sql: str, dialect: str = "athena") -> str:
    """Mismo contrato que validate_read_only_sql del agente, con sqlglot."""
    normalized = sql.strip().rstrip(";")
    inspection = inspect_sql(normalized, [], dialect)
    if not inspection.parsed:
        raise ValueError(f"SQL no válida: {inspection.parse_error}")
    if not inspection.single_statement or not inspection.read_only:
        raise ValueError("Only read-only SELECT/CTE SQL is allowed.")
    return normalized + ";"
