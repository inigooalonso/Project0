import pandas as pd

from core.formatting import fmt_change, fmt_eur, fmt_month, fmt_number, fmt_pct, fmt_seconds, pretty_column


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
