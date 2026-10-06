import pandas as pd

from core.formatting import fmt_change, fmt_eur, fmt_month, fmt_number, fmt_pct, fmt_seconds, pretty_column
from core.result_profile import percent_decimals, profile_result


def test_spanish_number_formats():
    assert fmt_number(1234567.891, 2) == "1.234.567,89"
    assert fmt_eur(42625500) == "42,6 M€"
    assert fmt_eur(551467) == "551 mil €"
    assert fmt_eur(42625500, compact=False) == "42.625.500 €"
    assert fmt_pct(7.891) == "7,89 %"
    assert fmt_change(0.127) == "+12,7 %"
    assert fmt_seconds(0.42) == "420 ms" and fmt_seconds(2.5) == "2,5 s"
    assert fmt_month(pd.Timestamp("2026-09-30")) == "sep 2026"


def test_column_labels():
    assert pretty_column("importe_concedido_eur") == "Importe concedido (€)"
    assert pretty_column("tasa_de_mora_pct") == "Tasa de mora (%)"
    assert pretty_column("num_hipotecas") == "Nº hipotecas"
    assert pretty_column("direccion_territorial") == "Dirección territorial"


def test_chart_choice_follows_the_shape_of_the_data():
    one_row = pd.DataFrame({"total_eur": [10.0]})
    assert profile_result(one_row).kind == "kpi"
    series = pd.DataFrame({"mes": pd.date_range("2026-01-31", periods=4, freq="ME"), "saldo_eur": [1.0, 2, 3, 4]})
    assert profile_result(series).kind == "line"
    ranking = pd.DataFrame({"oficina": list("abc"), "importe_eur": [3.0, 2, 1], "pct": [50.0, 30, 20]})
    profile = profile_result(ranking, ["importe_eur"])
    assert (profile.kind, profile.x, profile.y) == ("bar", "oficina", "importe_eur")
    wide = pd.DataFrame({"a": [f"x{i}" for i in range(80)], "b": range(80)})
    assert profile_result(wide).kind == "table"


def test_percent_decimals_follow_the_data():
    assert percent_decimals(pd.Series([42.4, 17.8])) == 1
    assert percent_decimals(pd.Series([10.65, 9.38])) == 2
