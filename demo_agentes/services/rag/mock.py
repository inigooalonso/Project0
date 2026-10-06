"""RAG multinivel SIMULADO sobre el catálogo bancario.

Para cada entidad del IR recorre los tres niveles como lo haría un buscador
vectorial jerárquico:
  1. propietario: por la entidad de negocio (señal de tabla en el IR);
  2. tabla: dentro de los mejores propietarios;
  3. campo: por el concepto (señal de columna en el IR) y, en los filtros, por
     coincidencia con los valores reales del campo.
Se queda con el mejor camino por entidad y conserva los candidatos descartados
para que la interfaz pueda enseñarlos con su puntuación.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.models import EntitySearch, FieldMatch, OwnerMatch, Pydantic_SemanticQueryIR, RAGResult, TableMatch
from services.catalog import Catalog, CatalogField, CatalogOwner, CatalogTable
from services.rag.base import IRItem, ir_items
from services.text import best_similarity, coverage, display_score, normalize

TOP_OWNERS = 2
TOP_TABLES = 2
TOP_FIELDS = 3
MIN_RAW = 0.18
NUMERIC_TYPES = {"decimal", "integer"}


@dataclass
class _Node:
    score: float = 0.0
    selected: bool = False
    entity_ids: set[str] = field(default_factory=set)


def _owner_raw(item: IRItem, owner: CatalogOwner) -> float:
    keylabel = " ".join([owner.name, *owner.keywords, *(t.label for t in owner.tables)])
    phrases = owner.keywords + tuple(k for t in owner.tables for k in t.keywords)
    queries = [(item.entity, 1.0), (f"{item.entity} {item.surface_form}", 0.95), (item.surface_form, 0.85), (item.concept, 0.7)]
    return best_similarity(queries, keylabel, owner.description, phrases)


def _table_raw(item: IRItem, table: CatalogTable) -> float:
    keylabel = " ".join([table.label, *table.keywords])
    queries = [(item.entity, 1.0), (f"{item.entity} {item.concept}", 0.95), (item.surface_form, 0.85), (item.concept, 0.75)]
    raw = best_similarity(queries, keylabel, table.description, table.keywords)
    # El concepto también orienta la tabla: la que contiene el mejor campo gana enteros.
    field_best = max((_field_raw(item, f) for f in table.fields), default=0.0)
    return 0.7 * raw + 0.3 * field_best


def _value_match(value: object, fld: CatalogField) -> bool:
    if value is None or not fld.values:
        return False
    values = value if isinstance(value, (list, tuple)) else [value]
    for v in values:
        nv = normalize(str(v))
        if not nv:
            continue
        for candidate in fld.values:
            nc = normalize(candidate)
            if nv == nc or nv in nc or nc in nv or coverage(nv, nc) >= 0.99:
                return True
    return False


def _field_raw(item: IRItem, fld: CatalogField) -> float:
    keylabel = " ".join([fld.label, *fld.keywords, *fld.values])
    phrases = fld.keywords + (fld.label,)
    queries = [(item.concept, 1.0), (f"{item.concept} {item.surface_form}", 0.95), (item.surface_form, 0.85)]
    raw = best_similarity(queries, keylabel, fld.description, phrases)
    if item.kind == "filter" and _value_match(item.value, fld):
        raw = max(raw, 0.92)
    # Coherencia de tipos: periodos y dimensiones temporales → fechas; métricas → importes.
    if item.is_temporal:
        raw = raw * 1.1 if fld.type == "date" else raw * 0.6
    elif item.is_measure:
        raw = raw * 1.08 if fld.type in NUMERIC_TYPES else raw * 0.75
    elif item.kind in {"dimension", "attribute"} and fld.type in NUMERIC_TYPES:
        raw *= 0.7
    return min(raw, 1.0)


class MockRAGService:
    name = "Catálogo simulado"
    simulated = True

    def __init__(self, catalog: Catalog, dialect: str) -> None:
        self.catalog = catalog
        self.dialect = dialect

    def search(self, ir: Pydantic_SemanticQueryIR) -> RAGResult:
        owners: dict[str, _Node] = {}
        tables: dict[str, _Node] = {}
        fields: dict[tuple[str, str], _Node] = {}
        searches: list[EntitySearch] = []
        selected_tables: list[str] = []
        schema_context: list[dict] = []

        def touch(store: dict, key, score: float, entity_id: str) -> _Node:
            node = store.setdefault(key, _Node())
            node.score = max(node.score, score)
            node.entity_ids.add(entity_id)
            return node

        for item in ir_items(ir):
            ranked_owners = sorted(((o, _owner_raw(item, o)) for o in self.catalog.owners), key=lambda x: -x[1])
            top_owners = [x for x in ranked_owners[:TOP_OWNERS] if x[1] >= MIN_RAW] or ranked_owners[:1]
            best = None
            seen = 0
            for owner, o_raw in top_owners:
                o_score = display_score(o_raw, f"{item.id}|{owner.code}")
                touch(owners, owner.code, o_score, item.id)
                ranked_tables = sorted(((t, _table_raw(item, t)) for t in owner.tables), key=lambda x: -x[1])[:TOP_TABLES]
                for table, t_raw in ranked_tables:
                    t_score = display_score(t_raw, f"{item.id}|{table.name}")
                    touch(tables, table.fq_name, t_score, item.id)
                    ranked_fields = sorted(((f, _field_raw(item, f)) for f in table.fields), key=lambda x: -x[1])[:TOP_FIELDS]
                    for fld, f_raw in ranked_fields:
                        seen += 1
                        f_score = display_score(f_raw, f"{item.id}|{table.name}.{fld.name}")
                        touch(fields, (table.fq_name, fld.name), f_score, item.id)
                        combined = 0.25 * o_raw + 0.30 * t_raw + 0.45 * f_raw
                        if best is None or combined > best[0]:
                            best = (combined, owner, o_score, table, t_score, fld, f_score)

            _, owner, o_score, table, t_score, fld, f_score = best
            owners[owner.code].selected = True
            tables[table.fq_name].selected = True
            fields[(table.fq_name, fld.name)].selected = True
            if table.fq_name not in selected_tables:
                selected_tables.append(table.fq_name)
            searches.append(
                EntitySearch(
                    entity_id=item.id, kind=item.kind, surface_form=item.surface_form, entity=item.entity,
                    concept=item.concept, owner=owner.code, owner_score=o_score, table=table.fq_name,
                    table_score=t_score, field=fld.name, field_score=f_score, candidates_seen=seen,
                )
            )
            entry = {
                "entity_id": item.id,
                "owner": f"{owner.code} · {owner.name}",
                "table": table.fq_name,
                "field": fld.name,
                "description": f"{fld.label}: {fld.description}",
                "type": fld.type,
                "score": f_score,
            }
            if fld.values:
                entry["values"] = list(fld.values)
            schema_context.append(entry)

        return RAGResult(
            source="mock",
            dialect=self.dialect,
            owners=self._tree(owners, tables, fields),
            searches=searches,
            selected_tables=selected_tables,
            authorized_tables=[self.catalog.table(t).authorized_block() for t in selected_tables],
            schema_context=schema_context,
            has_scores=True,
        )

    def _tree(self, owners: dict, tables: dict, fields: dict) -> list[OwnerMatch]:
        result: list[OwnerMatch] = []
        for owner in self.catalog.owners:
            if owner.code not in owners:
                continue
            o_node = owners[owner.code]
            table_matches = []
            for table in owner.tables:
                if table.fq_name not in tables:
                    continue
                t_node = tables[table.fq_name]
                field_matches = [
                    FieldMatch(
                        name=f.name, label=f.label, description=f.description, type=f.type, role=f.role,
                        values=list(f.values), score=fields[(table.fq_name, f.name)].score,
                        selected=fields[(table.fq_name, f.name)].selected,
                        entity_ids=sorted(fields[(table.fq_name, f.name)].entity_ids),
                    )
                    for f in table.fields
                    if (table.fq_name, f.name) in fields
                ]
                field_matches.sort(key=lambda f: (not f.selected, -(f.score or 0)))
                table_matches.append(
                    TableMatch(name=table.fq_name, label=table.label, description=table.description,
                               score=t_node.score, selected=t_node.selected, fields=field_matches)
                )
            table_matches.sort(key=lambda t: (not t.selected, -(t.score or 0)))
            result.append(
                OwnerMatch(code=owner.code, name=owner.name, business_owner=owner.business_owner,
                           score=o_node.score, selected=o_node.selected, tables=table_matches)
            )
        result.sort(key=lambda o: (not o.selected, -(o.score or 0)))
        return result
