"""Carga del catálogo simulado (propietario → tabla → campo)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from core.settings import MOCK_DIR


@dataclass(frozen=True)
class CatalogField:
    name: str
    label: str
    description: str
    type: str
    role: str | None = None
    values: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogTable:
    name: str  # nombre físico sin base: t_pred_branches
    label: str
    description: str
    keywords: tuple[str, ...]
    fields: tuple[CatalogField, ...]
    database: str
    owner_code: str

    @property
    def fq_name(self) -> str:
        return f"{self.database}.{self.name}"

    def field(self, name: str) -> CatalogField | None:
        return next((f for f in self.fields if f.name == name), None)

    def authorized_block(self) -> str:
        """Bloque de texto con el mismo formato que la variable `table` del agente."""
        lines = [f"{self.fq_name}:", f"-Description:{self.description}"]
        for i, f in enumerate(self.fields):
            entry = f"{f.name}: {f.label}, {f.description}"
            if f.values:
                entry += f" Valores: {', '.join(f.values)}."
            lines.append(f"-Fields:{entry}" if i == 0 else f"* {entry}")
        return "\n".join(lines)


@dataclass(frozen=True)
class CatalogOwner:
    code: str
    name: str
    business_owner: str
    description: str
    keywords: tuple[str, ...]
    tables: tuple[CatalogTable, ...]


@dataclass(frozen=True)
class Catalog:
    database: str
    owners: tuple[CatalogOwner, ...]

    def tables(self) -> list[CatalogTable]:
        return [t for o in self.owners for t in o.tables]

    def table(self, fq_or_short: str) -> CatalogTable | None:
        short = fq_or_short.split(".")[-1]
        return next((t for t in self.tables() if t.name == short), None)

    def owner_of(self, table: CatalogTable) -> CatalogOwner:
        return next(o for o in self.owners if o.code == table.owner_code)


def _tuple(values) -> tuple[str, ...]:
    return tuple(str(v) for v in (values or []))


def parse_catalog(raw: dict) -> Catalog:
    database = raw["database"]
    owners = []
    for o in raw["owners"]:
        tables = []
        for t in o["tables"]:
            fields = tuple(
                CatalogField(
                    name=f["name"],
                    label=f.get("label", ""),
                    description=f.get("description", ""),
                    type=f.get("type", "string"),
                    role=f.get("role"),
                    values=_tuple(f.get("values")),
                    keywords=_tuple(f.get("keywords")),
                )
                for f in t["fields"]
            )
            tables.append(
                CatalogTable(
                    name=t["name"],
                    label=t.get("label", ""),
                    description=t.get("description", ""),
                    keywords=_tuple(t.get("keywords")),
                    fields=fields,
                    database=database,
                    owner_code=o["code"],
                )
            )
        owners.append(
            CatalogOwner(
                code=o["code"],
                name=o["name"],
                business_owner=o.get("business_owner", ""),
                description=o.get("description", ""),
                keywords=_tuple(o.get("keywords")),
                tables=tuple(tables),
            )
        )
    return Catalog(database=database, owners=tuple(owners))


@lru_cache(maxsize=4)
def load_catalog(path: Path = MOCK_DIR / "catalog.yaml") -> Catalog:
    return parse_catalog(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
