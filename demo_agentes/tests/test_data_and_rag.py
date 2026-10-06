"""Coherencia entre catálogo, datos simulados, RAG, joins, SQL y resultado."""
import pytest

from core.settings import MOCK_DIR
from services.catalog import load_catalog
from services.executor.duckdb_mock import DuckDBMockExecutor, get_mock_database
from services.knowledge.yaml_store import YamlKnowledgeService
from services.rag.mock import MockRAGService
from services.sql_guard import inspect_sql, validate_read_only

EXPECTED_TABLES = {
    "hipotecas_oficina": {"ho_master.t_prsg_loans", "ho_master.t_pred_branches"},
    "saldo_vista": {"ho_master.t_pcta_account_balances_monthly", "ho_master.t_pcta_accounts", "ho_master.t_pcli_customers"},
    "gasto_tarjeta": {"ho_master.t_pmpg_card_transactions", "ho_master.t_pmpg_cards"},
    "morosidad_oficinas": {"ho_master.t_prsg_loan_delinquency", "ho_master.t_pred_branches", "ho_master.t_prsg_loans"},
}


@pytest.fixture(scope="module")
def rag():
    return MockRAGService(load_catalog(), "AWS Athena (Trino SQL)")


@pytest.fixture(scope="module")
def knowledge():
    return YamlKnowledgeService(MOCK_DIR / "joins.yaml", MOCK_DIR / "glossary.yaml")


def test_duckdb_tables_match_the_catalog_field_by_field():
    con = get_mock_database()
    for table in load_catalog().tables():
        columns = [row[0] for row in con.execute(f"DESCRIBE {table.fq_name}").fetchall()]
        assert columns == [f.name for f in table.fields], table.name
        assert con.execute(f"SELECT COUNT(*) FROM {table.fq_name}").fetchone()[0] > 0


def test_authorized_block_uses_the_agent_text_format():
    block = load_catalog().table("t_prsg_loans").authorized_block()
    lines = block.splitlines()
    assert lines[0] == "ho_master.t_prsg_loans:"
    assert lines[1].startswith("-Description:")
    assert lines[2].startswith("-Fields:gf_loan_id: ")
    assert all(line.startswith("* gf_") for line in lines[3:])


@pytest.mark.parametrize("scenario_id", list(EXPECTED_TABLES))
def test_rag_joins_sql_and_result_are_coherent(scenario_id, scenarios, rag, knowledge):
    scenario = next(s for s in scenarios if s.id == scenario_id)
    ir = scenario.ir()
    found = rag.search(ir)
    resolved = knowledge.resolve(ir, found)
    assert set(resolved.tables) == EXPECTED_TABLES[scenario_id]
    assert not resolved.disconnected_tables
    # Todas las entidades del IR se han buscado y tienen puntuación en los tres niveles.
    assert len(found.searches) >= len(ir.metrics) + len(ir.dimensions)
    for search in found.searches:
        assert 0.4 <= search.owner_score <= 1 and 0.4 <= search.table_score <= 1 and 0.4 <= search.field_score <= 1

    variants = scenario.sql_variants.values() if scenario.sql_variants else [scenario.sql]
    for sql in variants:
        inspection = inspect_sql(sql, resolved.tables)
        assert inspection.read_only and inspection.single_statement
        assert set(inspection.tables) == set(resolved.tables)  # SQL ⇔ tablas del RAG + puentes
        assert not inspection.unauthorized_tables
        result = DuckDBMockExecutor().execute(validate_read_only(sql))
        assert result.row_count > 0


def test_morosidad_needs_a_bridge_table_and_flags_the_ambiguity(scenarios, rag, knowledge):
    scenario = next(s for s in scenarios if s.id == "morosidad_oficinas")
    resolved = knowledge.resolve(scenario.ir(), rag.search(scenario.ir()))
    assert resolved.bridge_tables == ["ho_master.t_prsg_loans"]
    assert set(resolved.ambiguous_terms.get("morosidad", [])) == {"Tasa de mora", "Ratio de impagados"}


def test_both_definitions_of_morosidad_rank_offices_differently(scenarios):
    scenario = next(s for s in scenarios if s.id == "morosidad_oficinas")
    executor = DuckDBMockExecutor()
    first = executor.execute(scenario.sql_for("tasa_mora")).df.iloc[0]["oficina"]
    second = executor.execute(scenario.sql_for("impagados")).df.iloc[0]["oficina"]
    assert first != second


def test_mock_data_is_deterministic():
    from data.synthetic import build_branches
    import numpy as np

    a = build_branches(np.random.default_rng(1))
    b = build_branches(np.random.default_rng(1))
    assert a.equals(b)
