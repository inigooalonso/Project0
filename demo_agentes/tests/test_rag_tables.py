"""Tablas del RAG: nombres de columna flexibles, grain y tabla unificada."""
from services.rag.agent_adapter import build_result
from services.rag.tables import GRAIN_LABEL, canonical, parse_candidates, to_bool
from tests.fakes import CONTEXT, TABLE


def test_column_names_accept_the_usual_variants():
    for key in ("Sim. UUAA", "Sim UUAA", "sim_uuaa", "SIM-UUAA"):
        assert canonical(key) == "sim_uuaa"
    assert canonical("Sim Ponderado") == canonical("sim_weighted") == "sim_weighted"
    assert canonical("Cumple Grain") == canonical("cumple_grain") == "meets_grain"
    assert canonical("Campo libre") is None


def test_grain_values_are_read_as_booleans():
    assert [to_bool(v) for v in (True, "Sí", "si", 1, "No", 0, False, "nok")] == [True] * 4 + [False] * 4
    assert to_bool(None) is None and to_bool("quizá") is None


def test_candidates_keep_scores_and_grain():
    rows = parse_candidates(CONTEXT)
    assert len(rows) == 3
    assert rows[0].sim_table == 0.88 and rows[0].sim_weighted == 0.85
    assert [r.meets_grain for r in rows] == [True, True, False]
    # Columnas en formato {"columna": [valores]} y similitudes como texto con coma decimal.
    columnar = {"rag_candidates": {"entidad": ["x"], "sim_ponderado": ["0,7"], "cumple_grain": ["No"]}}
    row = parse_candidates(columnar)[0]
    assert (row.entity, row.sim_weighted, row.meets_grain) == ("x", 0.7, False)


def test_unified_table_from_the_rag_is_shown_as_is():
    context = dict(CONTEXT, rag_unified=[{"Tabla": TABLE, "cumple_grain": "Sí", "Entidades": "negocio, mesa"}])
    result = build_result(context, "AWS Athena (Trino SQL)", "RAG")
    assert not result.unified_derived
    assert result.unified == [{"Tabla": TABLE, GRAIN_LABEL: True, "Entidades": "negocio, mesa"}]


def test_unified_table_is_derived_when_missing():
    result = build_result(dict(CONTEXT), "AWS Athena (Trino SQL)", "RAG")
    assert result.unified_derived
    best = {row["Entidad"]: row for row in result.unified}
    assert best["mesa"]["Tabla"] == TABLE and best["mesa"][GRAIN_LABEL] is True  # descarta el que no cumple grain
    assert result.selected_tables == [TABLE] and len(result.tables[0].fields) == 3
    assert result.raw_context["original_dialect"] == "snowflake" and result.raw_context["dialect"].startswith("AWS")


def test_rag_tables_never_reach_the_llm_context():
    from core.context import assemble_context
    from core.models import KnowledgeResult

    result = build_result(dict(CONTEXT), "AWS Athena (Trino SQL)", "RAG")
    bundle = assemble_context(result, KnowledgeResult(), "AWS Athena (Trino SQL)")
    assert "rag_candidates" not in bundle.context and bundle.context["authorized_tables"] == result.authorized_tables


def test_fields_in_schema_context_are_merged_into_authorized_tables():
    from core.context import assemble_context
    from core.models import KnowledgeResult

    daily, monthly = "ho_master.t_x_daily", "ho_master.t_x_monthly"
    context = {
        "authorized_tables": [f"{daily}:\n-Description:diaria\n", f"{monthly}:\n-Description:mensual\n"],
        "schema_context": [
            f"{daily}:\n* f1: uno, primero\n* f2: dos, segundo\n",
            f"{daily}:\n* f2: dos, segundo\n",               # repetido
            f"{monthly}:\n* m1: mes, campo mensual\n",
            "ho_master.t_otra:\n* z: zeta, no autorizada\n",  # tabla no autorizada
            {"table": daily, "field": "f1"},                  # no es texto: se ignora
        ],
    }
    result = build_result(context, "AWS Athena (Trino SQL)", "RAG")
    assert [(t.name, [f.name for f in t.fields]) for t in result.tables] == [(daily, ["f1", "f2"]), (monthly, ["m1"])]
    assert result.tables[0].fields[0].label == "uno" and result.tables[0].description == "diaria"
    assert result.duplicated_fields == 1 and result.unauthorized_field_tables == ["ho_master.t_otra"]
    bundle = assemble_context(result, KnowledgeResult(), "AWS Athena (Trino SQL)")
    assert bundle.field_count == 3 and bundle.context["schema_context"] == context["schema_context"]
