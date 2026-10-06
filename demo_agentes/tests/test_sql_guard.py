import pytest

from services.sql_guard import inspect_sql, validate_read_only

AUTH = ["ho_master.t_prsg_loans", "ho_master.t_pred_branches"]


def test_read_only_select_passes_and_keeps_the_agent_contract():
    # Igual que validate_read_only_sql: quita espacios y ';' finales y añade uno.
    assert validate_read_only("  SELECT 1;;  ") == "SELECT 1;"


@pytest.mark.parametrize("sql", [
    "DELETE FROM ho_master.t_prsg_loans",
    "DROP TABLE ho_master.t_prsg_loans",
    "INSERT INTO ho_master.t_prsg_loans VALUES (1)",
    "SELECT 1; DELETE FROM ho_master.t_prsg_loans",
])
def test_write_operations_are_blocked(sql):
    with pytest.raises(ValueError):
        validate_read_only(sql)


def test_unauthorized_tables_are_detected_and_ctes_are_ignored():
    sql = """
        WITH base AS (SELECT * FROM ho_master.t_prsg_loans)
        SELECT * FROM base JOIN ho_master.t_pcli_customers c ON base.gf_customer_id = c.gf_customer_id
    """
    inspection = inspect_sql(sql, AUTH)
    assert set(inspection.tables) == {"ho_master.t_prsg_loans", "ho_master.t_pcli_customers"}
    assert inspection.unauthorized_tables == ["ho_master.t_pcli_customers"]
    assert inspection.cte_count == 1 and inspection.join_count == 1
