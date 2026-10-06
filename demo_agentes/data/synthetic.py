"""Generador determinista de los datos bancarios simulados.

Produce las 9 tablas del catálogo (data/mock/catalog.yaml) para una red de
62 oficinas, con fecha de último cierre 30/09/2026 y 24 meses de historia.
La semilla es fija: la misma pregunta devuelve siempre el mismo resultado.

Rasgos que hacen creíble la demo:
- las oficinas grandes y de zonas de renta alta conceden más hipotecas;
- los saldos a la vista tienen estacionalidad (pagas extra, verano, enero);
- las tarjetas de crédito concentran viajes, moda y tecnología, con pico en verano;
- la morosidad se simula como un proceso mensual de impago y cura, con dos
  factores por oficina (frecuencia de impago y escalado a más de 90 días), de
  modo que la tasa de mora y el ratio de impagados no ordenan igual las oficinas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 20261006
LAST_CUTOFF = pd.Timestamp("2026-09-30")
N_MONTHS = 24
BURN_IN_MONTHS = 24
N_CUSTOMERS = 30_000
N_MOVEMENTS = 300_000
N_CARD_TXNS = 600_000

# (nombre, municipio, provincia, dirección territorial, tipo, tamaño, renta, frecuencia de impago, escalado a >90 días)
BRANCHES: list[tuple[str, str, str, str, str, float, float, float, float]] = [
    ("Madrid · Castellana", "Madrid", "Madrid", "Centro", "Urbana", 2.6, 1.55, 0.70, 0.80),
    ("Madrid · Goya", "Madrid", "Madrid", "Centro", "Urbana", 1.8, 1.40, 0.75, 0.85),
    ("Madrid · Chamberí", "Madrid", "Madrid", "Centro", "Urbana", 2.0, 1.45, 0.70, 0.80),
    ("Madrid · Arturo Soria", "Madrid", "Madrid", "Centro", "Urbana", 1.7, 1.35, 0.80, 0.85),
    ("Madrid · Princesa", "Madrid", "Madrid", "Centro", "Urbana", 1.3, 1.20, 0.90, 0.95),
    ("Madrid · Retiro", "Madrid", "Madrid", "Centro", "Urbana", 1.2, 1.30, 0.85, 0.90),
    ("Madrid · Las Tablas", "Madrid", "Madrid", "Centro", "Urbana", 1.5, 1.25, 0.85, 0.90),
    ("Pozuelo de Alarcón", "Pozuelo de Alarcón", "Madrid", "Centro", "Urbana", 1.6, 1.60, 0.65, 0.75),
    ("Alcobendas", "Alcobendas", "Madrid", "Centro", "Empresas", 1.4, 1.20, 0.85, 1.00),
    ("Las Rozas", "Las Rozas", "Madrid", "Centro", "Urbana", 1.4, 1.40, 0.75, 0.85),
    ("Getafe", "Getafe", "Madrid", "Centro", "Urbana", 1.1, 0.90, 1.70, 0.80),
    ("Alcalá de Henares", "Alcalá de Henares", "Madrid", "Centro", "Urbana", 1.0, 0.90, 1.55, 1.00),
    ("Toledo · Zocodover", "Toledo", "Toledo", "Centro", "Comarcal", 0.7, 0.85, 1.05, 1.10),
    ("Guadalajara · Centro", "Guadalajara", "Guadalajara", "Centro", "Comarcal", 0.6, 0.85, 1.10, 1.05),
    ("Barcelona · Diagonal", "Barcelona", "Barcelona", "Cataluña", "Urbana", 2.4, 1.50, 0.70, 0.80),
    ("Barcelona · Passeig de Gràcia", "Barcelona", "Barcelona", "Cataluña", "Urbana", 1.9, 1.50, 0.70, 0.85),
    ("Barcelona · Sants", "Barcelona", "Barcelona", "Cataluña", "Urbana", 1.2, 1.05, 1.15, 1.00),
    ("Barcelona · Gràcia", "Barcelona", "Barcelona", "Cataluña", "Urbana", 1.1, 1.20, 0.95, 0.95),
    ("Sabadell · Centre", "Sabadell", "Barcelona", "Cataluña", "Empresas", 1.1, 1.00, 1.10, 1.10),
    ("Terrassa", "Terrassa", "Barcelona", "Cataluña", "Urbana", 0.9, 0.95, 1.20, 1.05),
    ("L'Hospitalet", "L'Hospitalet de Llobregat", "Barcelona", "Cataluña", "Urbana", 1.0, 0.85, 1.95, 0.60),
    ("Girona · Jaume I", "Girona", "Girona", "Cataluña", "Comarcal", 0.7, 1.05, 0.85, 0.95),
    ("Tarragona · Rambla Nova", "Tarragona", "Tarragona", "Cataluña", "Comarcal", 0.7, 0.95, 1.00, 1.10),
    ("Lleida · Blondel", "Lleida", "Lleida", "Cataluña", "Comarcal", 0.6, 0.90, 0.95, 1.05),
    ("Valencia · Colón", "Valencia", "Valencia", "Este", "Urbana", 1.9, 1.25, 0.85, 0.90),
    ("Valencia · Ruzafa", "Valencia", "Valencia", "Este", "Urbana", 1.1, 1.00, 1.10, 1.00),
    ("Valencia · Campanar", "Valencia", "Valencia", "Este", "Urbana", 1.0, 1.05, 1.00, 1.00),
    ("Alicante · Maisonnave", "Alicante", "Alicante", "Este", "Urbana", 1.1, 1.00, 1.15, 1.20),
    ("Elche", "Elche", "Alicante", "Este", "Urbana", 0.9, 0.85, 1.35, 1.55),
    ("Castellón · Puerta del Sol", "Castellón de la Plana", "Castellón", "Este", "Comarcal", 0.7, 0.90, 1.05, 1.10),
    ("Murcia · Gran Vía", "Murcia", "Murcia", "Este", "Urbana", 1.1, 0.90, 1.20, 1.35),
    ("Cartagena", "Cartagena", "Murcia", "Este", "Comarcal", 0.7, 0.85, 1.05, 2.10),
    ("Palma · Jaume III", "Palma", "Illes Balears", "Este", "Urbana", 1.3, 1.30, 0.90, 0.95),
    ("Ibiza", "Ibiza", "Illes Balears", "Este", "Comarcal", 0.6, 1.35, 0.85, 1.00),
    ("Sevilla · Nervión", "Sevilla", "Sevilla", "Sur", "Urbana", 1.5, 1.05, 1.05, 1.10),
    ("Sevilla · Triana", "Sevilla", "Sevilla", "Sur", "Urbana", 1.0, 0.95, 1.70, 0.80),
    ("Sevilla · Los Remedios", "Sevilla", "Sevilla", "Sur", "Urbana", 1.0, 1.15, 0.95, 1.00),
    ("Málaga · Larios", "Málaga", "Málaga", "Sur", "Urbana", 1.5, 1.20, 0.95, 1.05),
    ("Marbella", "Marbella", "Málaga", "Sur", "Urbana", 1.3, 1.65, 0.80, 1.00),
    ("Córdoba · Tendillas", "Córdoba", "Córdoba", "Sur", "Urbana", 0.9, 0.90, 1.20, 1.30),
    ("Granada · Gran Vía", "Granada", "Granada", "Sur", "Urbana", 0.9, 0.90, 1.15, 1.25),
    ("Cádiz · San José", "Cádiz", "Cádiz", "Sur", "Comarcal", 0.6, 0.85, 1.15, 1.25),
    ("Jerez", "Jerez de la Frontera", "Cádiz", "Sur", "Comarcal", 0.8, 0.80, 1.15, 2.80),
    ("Almería · Paseo", "Almería", "Almería", "Sur", "Comarcal", 0.7, 0.85, 1.10, 2.30),
    ("Bilbao · Gran Vía", "Bilbao", "Bizkaia", "Norte", "Urbana", 1.7, 1.35, 0.65, 0.75),
    ("Bilbao · Indautxu", "Bilbao", "Bizkaia", "Norte", "Urbana", 1.1, 1.25, 0.70, 0.80),
    ("Getxo", "Getxo", "Bizkaia", "Norte", "Urbana", 0.8, 1.45, 0.60, 0.75),
    ("San Sebastián · Centro", "Donostia-San Sebastián", "Gipuzkoa", "Norte", "Urbana", 1.1, 1.45, 0.60, 0.75),
    ("Vitoria · Dato", "Vitoria-Gasteiz", "Araba/Álava", "Norte", "Urbana", 0.9, 1.15, 0.70, 0.80),
    ("Santander · Paseo Pereda", "Santander", "Cantabria", "Norte", "Urbana", 0.9, 1.10, 0.80, 0.90),
    ("Pamplona · Carlos III", "Pamplona", "Navarra", "Norte", "Urbana", 0.9, 1.15, 0.70, 0.80),
    ("Zaragoza · Independencia", "Zaragoza", "Zaragoza", "Norte", "Urbana", 1.3, 1.05, 0.90, 0.95),
    ("A Coruña · Cantones", "A Coruña", "A Coruña", "Noroeste", "Urbana", 1.0, 1.05, 0.85, 0.95),
    ("Vigo · Urzaiz", "Vigo", "Pontevedra", "Noroeste", "Urbana", 1.0, 0.95, 0.95, 1.05),
    ("Oviedo · Uría", "Oviedo", "Asturias", "Noroeste", "Urbana", 0.9, 1.00, 0.85, 0.95),
    ("Gijón", "Gijón", "Asturias", "Noroeste", "Urbana", 0.8, 0.95, 0.90, 1.00),
    ("Valladolid · Zorrilla", "Valladolid", "Valladolid", "Noroeste", "Urbana", 0.9, 1.00, 0.85, 0.95),
    ("Salamanca · Plaza Mayor", "Salamanca", "Salamanca", "Noroeste", "Comarcal", 0.6, 0.95, 0.85, 0.95),
    ("Las Palmas · Triana", "Las Palmas de Gran Canaria", "Las Palmas", "Canarias", "Urbana", 1.1, 0.95, 1.35, 1.80),
    ("Las Palmas · Mesa y López", "Las Palmas de Gran Canaria", "Las Palmas", "Canarias", "Urbana", 0.9, 1.00, 1.20, 1.30),
    ("Santa Cruz · Plaza de España", "Santa Cruz de Tenerife", "Santa Cruz de Tenerife", "Canarias", "Urbana", 1.0, 1.00, 1.20, 1.35),
    ("La Laguna", "San Cristóbal de La Laguna", "Santa Cruz de Tenerife", "Canarias", "Comarcal", 0.8, 0.85, 1.75, 0.80),
]

SEGMENTS = ["Particulares", "Banca Personal", "Pymes", "Empresas"]
CATEGORIES = [
    "Alimentación", "Restauración", "Viajes y alojamiento", "Combustible", "Moda",
    "Ocio y cultura", "Salud y farmacia", "Hogar y bricolaje", "Tecnología", "Transporte",
]
MCC = {
    "Alimentación": "5411", "Restauración": "5812", "Viajes y alojamiento": "4722",
    "Combustible": "5541", "Moda": "5651", "Ocio y cultura": "7832",
    "Salud y farmacia": "5912", "Hogar y bricolaje": "5200", "Tecnología": "5732",
    "Transporte": "4121",
}


@dataclass(frozen=True)
class MockDataset:
    """Tablas simuladas, con el nombre físico del catálogo como clave."""

    tables: dict[str, pd.DataFrame]

    @property
    def row_counts(self) -> dict[str, int]:
        return {name: len(df) for name, df in self.tables.items()}


def month_ends(n: int, last: pd.Timestamp = LAST_CUTOFF) -> pd.DatetimeIndex:
    return pd.date_range(end=last, periods=n, freq="ME")


def _random_dates(rng: np.random.Generator, start: pd.Timestamp, end: pd.Timestamp, n: int) -> np.ndarray:
    span = (end - start).days
    return (start + pd.to_timedelta(rng.integers(0, span + 1, n), unit="D")).values


def _unique_ids(rng: np.random.Generator, prefix: str, n: int, digits: int) -> list[str]:
    """Identificadores únicos aleatorios de longitud fija (sin materializar el rango)."""
    low = 10 ** (digits - 1)
    codes = rng.choice(9 * low, size=n, replace=False) + low
    return (prefix + pd.Series(codes).astype(str)).tolist()


def _months_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Meses completos entre dos arrays datetime64 (b posterior a a)."""
    a_m = a.astype("datetime64[M]").astype(int)
    b_m = b.astype("datetime64[M]").astype(int)
    return b_m - a_m


def build_branches(rng: np.random.Generator) -> pd.DataFrame:
    codes = rng.choice(np.arange(101, 9900), size=len(BRANCHES), replace=False)
    rows = []
    for code, (name, city, province, region, kind, *_rest) in zip(sorted(codes), BRANCHES):
        rows.append(
            {
                "gf_branch_id": f"{code:04d}",
                "gf_branch_name": name,
                "gf_branch_type_desc": kind,
                "gf_city_name": city,
                "gf_province_name": province,
                "gf_region_name": region,
                "gf_opening_date": pd.Timestamp("1965-01-01")
                + pd.Timedelta(days=int(rng.integers(0, 365 * 50))),
            }
        )
    return pd.DataFrame(rows)


def build_customers(rng: np.random.Generator, branches: pd.DataFrame) -> pd.DataFrame:
    size = np.array([b[5] for b in BRANCHES])
    affluence = np.array([b[6] for b in BRANCHES])
    is_business_branch = np.array([b[4] == "Empresas" for b in BRANCHES])

    branch_idx = rng.choice(len(BRANCHES), size=N_CUSTOMERS, p=size / size.sum())

    # Segmento: más Banca Personal en zonas de renta alta, más empresas en oficinas de empresas.
    p_bp = np.clip(0.06 + 0.10 * (affluence[branch_idx] - 1.0), 0.03, 0.16)
    p_pyme = np.where(is_business_branch[branch_idx], 0.16, 0.08)
    p_emp = np.where(is_business_branch[branch_idx], 0.06, 0.025)
    u = rng.random(N_CUSTOMERS)
    segment = np.where(
        u < p_emp, "Empresas",
        np.where(u < p_emp + p_pyme, "Pymes", np.where(u < p_emp + p_pyme + p_bp, "Banca Personal", "Particulares")),
    )
    is_person = np.isin(segment, ["Particulares", "Banca Personal"])

    # Antigüedad: mezcla de clientes históricos y captación reciente creciente.
    recent = rng.random(N_CUSTOMERS) < 0.45
    old_dates = _random_dates(rng, pd.Timestamp("1988-01-01"), pd.Timestamp("2015-12-31"), N_CUSTOMERS)
    years = 2016 + np.floor(10.75 * np.sqrt(rng.random(N_CUSTOMERS))).astype(int)
    years = np.minimum(years, 2026)
    day_of_year = rng.integers(0, np.where(years == 2026, 273, 365))
    new_dates = (pd.to_datetime(years.astype(str) + "-01-01") + pd.to_timedelta(day_of_year, unit="D")).values
    entry = np.where(recent, new_dates, old_dates)
    entry = np.minimum(entry, LAST_CUTOFF.to_datetime64())

    entry_year = pd.DatetimeIndex(entry).year.values
    p_digital = np.clip(0.04 + 0.065 * (entry_year - 2015), 0.02, 0.68)
    v = rng.random(N_CUSTOMERS)
    channel = np.where(v < p_digital, "Digital", np.where(v < p_digital + 0.07, "Agente colaborador", "Oficina"))

    age_band = rng.choice(["18-29", "30-44", "45-64", "65 o más"], size=N_CUSTOMERS, p=[0.17, 0.31, 0.33, 0.19])
    age_band = np.where(is_person, age_band, "No aplica")

    customer_ids = np.array(_unique_ids(rng, "C", N_CUSTOMERS, 8))
    return pd.DataFrame(
        {
            "gf_customer_id": customer_ids,
            "gf_customer_type_desc": np.where(is_person, "Persona física", "Persona jurídica"),
            "gf_customer_segment_desc": segment,
            "gf_age_band_desc": age_band,
            "gf_customer_entry_date": entry,
            "gf_acquisition_channel_desc": channel,
            "gf_branch_id": branches["gf_branch_id"].values[branch_idx],
            "gf_province_name": branches["gf_province_name"].values[branch_idx],
            "_branch_idx": branch_idx,
        }
    )


def build_accounts(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    n = len(customers)
    parts = []
    for product, prob in (("Cuenta a la vista", 1.0), ("Cuenta de ahorro", 0.25), ("Depósito a plazo", 0.12)):
        mask = rng.random(n) < prob
        sub = customers.loc[mask, ["gf_customer_id", "gf_branch_id", "gf_customer_entry_date", "gf_customer_segment_desc", "_branch_idx"]].copy()
        sub["gf_product_type_desc"] = product
        parts.append(sub)
    acc = pd.concat(parts, ignore_index=True)
    m = len(acc)

    entry = acc["gf_customer_entry_date"].values
    lag_days = np.where(acc["gf_product_type_desc"].values == "Cuenta a la vista", 0, rng.integers(0, 3000, m))
    opening = entry + pd.to_timedelta(lag_days, unit="D").values
    opening = np.minimum(opening, LAST_CUTOFF.to_datetime64())

    cancelled = rng.random(m) < 0.07
    cancel_offset = rng.integers(60, 6000, m)
    cancellation = opening + pd.to_timedelta(cancel_offset, unit="D").values
    cancellation = np.where(cancelled & (cancellation < LAST_CUTOFF.to_datetime64()), cancellation, np.datetime64("NaT"))

    acc["gf_account_id"] = _unique_ids(rng, "A", m, 10)
    acc["gf_opening_date"] = opening
    acc["gf_cancellation_date"] = pd.to_datetime(cancellation)
    acc["gf_currency_id"] = "EUR"
    return acc


def build_balances(rng: np.random.Generator, accounts: pd.DataFrame) -> pd.DataFrame:
    cutoffs = month_ends(N_MONTHS)
    m = len(accounts)
    seg = accounts["gf_customer_segment_desc"].values
    prod = accounts["gf_product_type_desc"].values
    affluence = np.array([b[6] for b in BRANCHES])[accounts["_branch_idx"].values]

    medians = {
        ("Cuenta a la vista", "Particulares"): 6_000, ("Cuenta a la vista", "Banca Personal"): 28_000,
        ("Cuenta a la vista", "Pymes"): 24_000, ("Cuenta a la vista", "Empresas"): 140_000,
        ("Cuenta de ahorro", "Particulares"): 11_000, ("Cuenta de ahorro", "Banca Personal"): 52_000,
        ("Cuenta de ahorro", "Pymes"): 35_000, ("Cuenta de ahorro", "Empresas"): 160_000,
        ("Depósito a plazo", "Particulares"): 22_000, ("Depósito a plazo", "Banca Personal"): 90_000,
        ("Depósito a plazo", "Pymes"): 70_000, ("Depósito a plazo", "Empresas"): 420_000,
    }
    median = pd.Series(list(zip(prod, seg))).map(medians).values.astype(float)
    base = median * affluence * rng.lognormal(0.0, 0.85, m)

    # Estacionalidad de las cuentas a la vista: pagas extra (jun, dic), verano y cuesta de enero.
    seasonal = {1: 0.955, 2: 0.965, 3: 0.975, 4: 0.985, 5: 0.995, 6: 1.055, 7: 1.045, 8: 0.975, 9: 0.985, 10: 0.99, 11: 1.0, 12: 1.075}
    trend_by_product = {"Cuenta a la vista": 0.0042, "Cuenta de ahorro": 0.0015, "Depósito a plazo": 0.0060}
    trend = pd.Series(prod).map(trend_by_product).values
    is_vista = prod == "Cuenta a la vista"

    opening = accounts["gf_opening_date"].values
    cancellation = accounts["gf_cancellation_date"].values
    frames = []
    for t, cutoff in enumerate(cutoffs):
        c64 = cutoff.to_datetime64()
        alive = (opening <= c64) & (np.isnat(cancellation) | (cancellation > c64))
        idx = np.flatnonzero(alive)
        s = np.where(is_vista[idx], seasonal[cutoff.month], 1.0)
        end_bal = base[idx] * s * (1 + trend[idx]) ** t * rng.lognormal(0.0, 0.06, idx.size)
        avg_bal = end_bal * rng.uniform(0.92, 1.04, idx.size)
        frames.append(
            pd.DataFrame(
                {
                    "gf_account_id": accounts["gf_account_id"].values[idx],
                    "gf_cutoff_date": c64,
                    "gf_end_month_balance_amount": np.round(end_bal, 2),
                    "gf_avg_month_balance_amount": np.round(avg_bal, 2),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_movements(rng: np.random.Generator, accounts: pd.DataFrame) -> pd.DataFrame:
    vista = accounts[accounts["gf_product_type_desc"] == "Cuenta a la vista"].reset_index(drop=True)
    start = LAST_CUTOFF - pd.DateOffset(months=N_MONTHS) + pd.Timedelta(days=1)
    pick = rng.integers(0, len(vista), N_MOVEMENTS)
    dates = _random_dates(rng, start, LAST_CUTOFF, N_MOVEMENTS)
    dates = np.maximum(dates, vista["gf_opening_date"].values[pick])

    concepts = np.array(["Nómina", "Transferencia emitida", "Transferencia recibida", "Bizum", "Recibo domiciliado",
                         "Retirada de efectivo", "Ingreso en efectivo", "Impuestos", "Comisión"])
    probs = np.array([0.10, 0.09, 0.07, 0.20, 0.26, 0.11, 0.04, 0.03, 0.10])
    concept = rng.choice(concepts, N_MOVEMENTS, p=probs)
    medians = {"Nómina": 1_850, "Transferencia emitida": 320, "Transferencia recibida": 290, "Bizum": 35,
               "Recibo domiciliado": 75, "Retirada de efectivo": 110, "Ingreso en efectivo": 280, "Impuestos": 380, "Comisión": 6}
    is_credit = np.isin(concept, ["Nómina", "Transferencia recibida", "Ingreso en efectivo"])
    bizum_in = (concept == "Bizum") & (rng.random(N_MOVEMENTS) < 0.5)
    is_credit = is_credit | bizum_in
    amount = pd.Series(concept).map(medians).values * rng.lognormal(0.0, 0.45, N_MOVEMENTS)
    amount = np.round(np.where(is_credit, amount, -amount), 2)
    channel = np.where(np.isin(concept, ["Retirada de efectivo", "Ingreso en efectivo"]),
                       rng.choice(["Cajero", "Oficina"], N_MOVEMENTS, p=[0.85, 0.15]),
                       rng.choice(["App", "Web", "Oficina"], N_MOVEMENTS, p=[0.72, 0.18, 0.10]))
    return pd.DataFrame(
        {
            "gf_movement_id": ("M" + pd.Series(np.arange(N_MOVEMENTS) + 10_000_000_000).astype(str)).values,
            "gf_account_id": vista["gf_account_id"].values[pick],
            "gf_operation_date": dates,
            "gf_movement_amount": amount,
            "gf_movement_type_desc": np.where(is_credit, "Abono", "Cargo"),
            "gf_movement_category_desc": concept,
            "gf_channel_desc": channel,
        }
    ).sort_values("gf_operation_date", kind="stable").reset_index(drop=True)


def build_cards(rng: np.random.Generator, customers: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    vista = accounts[accounts["gf_product_type_desc"] == "Cuenta a la vista"][["gf_customer_id", "gf_account_id", "gf_opening_date"]]
    cust = customers.merge(vista, on="gf_customer_id", how="inner")
    n = len(cust)
    is_person = cust["gf_customer_type_desc"].values == "Persona física"
    affluence = np.array([b[6] for b in BRANCHES])[cust["_branch_idx"].values]
    parts = []
    has_debit = rng.random(n) < np.where(is_person, 0.82, 0.55)
    parts.append(cust.loc[has_debit].assign(gf_card_type_desc="Débito"))
    has_credit = rng.random(n) < np.clip(np.where(is_person, 0.26, 0.40) * affluence, 0.05, 0.75)
    parts.append(cust.loc[has_credit].assign(gf_card_type_desc="Crédito"))
    cards = pd.concat(parts, ignore_index=True)
    m = len(cards)
    issue = cards["gf_opening_date"].values + pd.to_timedelta(rng.integers(0, 1500, m), unit="D").values
    issue = np.minimum(issue, LAST_CUTOFF.to_datetime64())
    status = rng.choice(["Activa", "Bloqueada", "Cancelada"], m, p=[0.93, 0.02, 0.05])
    return pd.DataFrame(
        {
            "gf_card_id": _unique_ids(rng, "T", m, 10),
            "gf_customer_id": cards["gf_customer_id"].values,
            "gf_account_id": cards["gf_account_id"].values,
            "gf_card_type_desc": cards["gf_card_type_desc"].values,
            "gf_card_brand_desc": rng.choice(["Visa", "Mastercard"], m, p=[0.62, 0.38]),
            "gf_issue_date": issue,
            "gf_card_status_desc": status,
            "_affluence": affluence[np.concatenate([np.flatnonzero(has_debit), np.flatnonzero(has_credit)])],
        }
    )


def build_card_transactions(rng: np.random.Generator, cards: pd.DataFrame) -> pd.DataFrame:
    usable = cards[cards["gf_card_status_desc"] != "Cancelada"].reset_index(drop=True)
    is_credit_card = usable["gf_card_type_desc"].values == "Crédito"
    activity = np.where(is_credit_card, 1.35, 1.0) * usable["_affluence"].values * rng.lognormal(0, 0.6, len(usable))
    pick = rng.choice(len(usable), N_CARD_TXNS, p=activity / activity.sum())
    start = LAST_CUTOFF - pd.DateOffset(months=N_MONTHS) + pd.Timedelta(days=1)
    dates = _random_dates(rng, start, LAST_CUTOFF, N_CARD_TXNS)
    dates = np.maximum(dates, usable["gf_issue_date"].values[pick])
    month = pd.DatetimeIndex(dates).month.values
    credit = is_credit_card[pick]

    base_debit = np.array([0.34, 0.17, 0.04, 0.12, 0.06, 0.06, 0.07, 0.05, 0.02, 0.07])
    base_credit = np.array([0.18, 0.17, 0.14, 0.06, 0.11, 0.08, 0.04, 0.07, 0.10, 0.05])
    summer = np.isin(month, [7, 8])
    sales = np.isin(month, [1, 7, 11, 12])
    weights = np.where(credit[:, None], base_credit, base_debit).astype(float)
    weights[summer, 2] *= 2.3   # viajes en verano
    weights[summer, 1] *= 1.25  # restauración en verano
    weights[sales, 4] *= 1.45   # moda en rebajas y Navidad
    weights[np.isin(month, [11, 12]), 8] *= 1.6  # tecnología en Black Friday y Navidad
    weights /= weights.sum(axis=1, keepdims=True)
    cum = weights.cumsum(axis=1)
    cat_idx = (rng.random(N_CARD_TXNS)[:, None] > cum).sum(axis=1)
    category = np.array(CATEGORIES)[cat_idx]

    medians = np.array([36, 27, 175, 52, 58, 33, 21, 64, 135, 16], dtype=float)
    amount = medians[cat_idx] * np.where(credit, 1.25, 1.0) * rng.lognormal(0, 0.55, N_CARD_TXNS)
    p_online = np.array([0.07, 0.04, 0.55, 0.0, 0.35, 0.30, 0.10, 0.18, 0.50, 0.25])[cat_idx]
    channel = np.where(rng.random(N_CARD_TXNS) < p_online, "Online", "Presencial")
    abroad = (cat_idx == 2) & (rng.random(N_CARD_TXNS) < 0.35)
    country = np.where(abroad, rng.choice(["FR", "PT", "IT", "GB", "DE", "US"], N_CARD_TXNS), "ES")
    return pd.DataFrame(
        {
            "gf_card_txn_id": ("X" + pd.Series(np.arange(N_CARD_TXNS) + 100_000_000_000).astype(str)).values,
            "gf_card_id": usable["gf_card_id"].values[pick],
            "gf_operation_date": dates,
            "gf_txn_amount": np.round(amount, 2),
            "gf_merchant_category_id": np.array([MCC[c] for c in CATEGORIES])[cat_idx],
            "gf_merchant_category_desc": category,
            "gf_channel_desc": channel,
            "gf_country_id": country,
        }
    ).sort_values("gf_operation_date", kind="stable").reset_index(drop=True)


def build_loans(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    persons = customers[customers["gf_customer_type_desc"] == "Persona física"].reset_index(drop=True)
    companies = customers[customers["gf_customer_type_desc"] == "Persona jurídica"].reset_index(drop=True)
    size = np.array([b[5] for b in BRANCHES])
    affluence = np.array([b[6] for b in BRANCHES])

    # Préstamos formalizados por año (nueva producción), de 2016 a septiembre de 2026.
    yearly = {"Hipoteca vivienda": 1_650, "Préstamo consumo": 2_500, "Préstamo vehículo": 700, "Préstamo empresa": 650}
    growth = {2016: 0.80, 2017: 0.85, 2018: 0.90, 2019: 0.95, 2020: 0.75, 2021: 0.95, 2022: 1.0,
              2023: 0.90, 2024: 0.98, 2025: 1.12, 2026: 0.85}  # 2026: hasta septiembre
    frames = []
    for product, per_year in yearly.items():
        pool = companies if product == "Préstamo empresa" else persons
        # Los clientes de oficinas grandes y de renta alta piden más hipotecas.
        w = size[pool["_branch_idx"].values] if product != "Hipoteca vivienda" else (
            size[pool["_branch_idx"].values] * affluence[pool["_branch_idx"].values])
        for year, factor in growth.items():
            k = int(per_year * factor * rng.uniform(0.95, 1.05))
            who = rng.choice(len(pool), k, p=w / w.sum())
            end = pd.Timestamp(f"{year}-12-31") if year < 2026 else LAST_CUTOFF
            frames.append(
                pd.DataFrame(
                    {
                        "gf_customer_id": pool["gf_customer_id"].values[who],
                        "gf_branch_id": pool["gf_branch_id"].values[who],
                        "_branch_idx": pool["_branch_idx"].values[who],
                        "gf_loan_product_desc": product,
                        "gf_formalization_date": _random_dates(rng, pd.Timestamp(f"{year}-01-01"), end, k),
                    }
                )
            )
    loans = pd.concat(frames, ignore_index=True)
    n = len(loans)
    prod = loans["gf_loan_product_desc"].values
    aff = affluence[loans["_branch_idx"].values]
    fy = pd.DatetimeIndex(loans["gf_formalization_date"]).year.values

    median = np.select(
        [prod == "Hipoteca vivienda", prod == "Préstamo consumo", prod == "Préstamo vehículo"],
        [165_000 * aff ** 0.9, 11_500 * aff ** 0.4, 21_000 * aff ** 0.3], 95_000 * aff ** 0.5,
    )
    sigma = np.select([prod == "Hipoteca vivienda", prod == "Préstamo consumo", prod == "Préstamo vehículo"], [0.33, 0.55, 0.35], 0.85)
    granted = np.round(median * rng.lognormal(0, sigma) / 100.0) * 100.0
    term = np.select(
        [prod == "Hipoteca vivienda", prod == "Préstamo consumo", prod == "Préstamo vehículo"],
        [rng.choice([240, 300, 360], n, p=[0.3, 0.4, 0.3]), rng.choice([36, 48, 60, 84, 96], n),
         rng.choice([48, 60, 72, 84, 96], n)], rng.choice([36, 60, 84, 120], n),
    )
    mortgage_rate = np.select([fy <= 2021, fy <= 2022, fy <= 2024], [1.9, 2.4, 3.9], 3.0)
    rate = np.select(
        [prod == "Hipoteca vivienda", prod == "Préstamo consumo", prod == "Préstamo vehículo"],
        [mortgage_rate, 7.9 + 0 * fy, 6.4 + 0 * fy], 4.6 + 0 * fy,
    ) + rng.normal(0, 0.35, n)

    elapsed = _months_between(loans["gf_formalization_date"].values, np.full(n, LAST_CUTOFF.to_datetime64()))
    early_repaid = rng.random(n) < np.where(prod == "Hipoteca vivienda", 0.012, 0.025) * np.maximum(elapsed, 0) / 12
    outstanding = granted * np.clip(1 - elapsed / term, 0, 1)
    outstanding = np.where(early_repaid, 0.0, np.round(outstanding, 2))

    loans["gf_loan_id"] = _unique_ids(rng, "P", n, 10)
    loans["gf_granted_amount"] = granted
    loans["gf_outstanding_amount"] = outstanding
    loans["gf_interest_rate_per"] = np.round(np.clip(rate, 0.5, 14.0), 2)
    loans["gf_term_months_number"] = term.astype(int)
    loans["gf_loan_status_desc"] = np.where(outstanding > 0, "Vivo", "Amortizado")
    loans["_early_repaid_after"] = np.where(early_repaid, rng.integers(6, np.maximum(elapsed, 7)), 10_000)
    return loans


def build_delinquency(rng: np.random.Generator, loans: pd.DataFrame) -> pd.DataFrame:
    """Simula mes a mes la entrada en impago, la cura y el escalado a más de 90 días."""
    early_factor = np.array([b[7] for b in BRANCHES])[loans["_branch_idx"].values]
    roll_factor = np.array([b[8] for b in BRANCHES])[loans["_branch_idx"].values]
    prod = loans["gf_loan_product_desc"].values
    p_enter = np.select(
        [prod == "Hipoteca vivienda", prod == "Préstamo consumo", prod == "Préstamo vehículo"], [0.0045, 0.0125, 0.0085], 0.0080,
    ) * 2.4 * early_factor ** 1.15 * rng.lognormal(0, 0.25, len(loans))
    p_cure_base = np.array([0.0, 0.60, 0.45, 0.30, 0.035])  # por número de cuotas impagadas (1, 2, 3, 4+)

    formal = loans["gf_formalization_date"].values
    granted = loans["gf_granted_amount"].values
    term = loans["gf_term_months_number"].values
    installment = granted / term * (1 + loans["gf_interest_rate_per"].values / 100 * 0.6)
    early_after = loans["_early_repaid_after"].values

    cutoffs = month_ends(N_MONTHS + BURN_IN_MONTHS)
    unpaid = np.zeros(len(loans), dtype=int)
    frames = []
    for t, cutoff in enumerate(cutoffs):
        c64 = cutoff.to_datetime64()
        elapsed = _months_between(formal, np.full(len(loans), c64))
        alive = (formal <= c64) & (elapsed < term) & (elapsed < early_after)
        unpaid[~alive] = 0

        current = alive & (unpaid == 0)
        enter = current & (rng.random(len(loans)) < p_enter)
        in_arrears = alive & (unpaid > 0)
        stage = np.minimum(unpaid, 4)
        p_cure = p_cure_base[stage] / np.where(stage >= 3, roll_factor ** 1.4, 1.0)
        cure = in_arrears & (rng.random(len(loans)) < p_cure)
        unpaid = np.where(cure, 0, np.where(in_arrears, np.minimum(unpaid + 1, 48), unpaid))
        unpaid = np.where(enter, 1, unpaid)

        if t < BURN_IN_MONTHS:
            continue
        idx = np.flatnonzero(alive)
        k = unpaid[idx]
        days = np.where(k > 0, (k - 1) * 30 + rng.integers(1, 31, idx.size), 0)
        outstanding = granted[idx] * np.clip(1 - elapsed[idx] / term[idx], 0, 1) + np.minimum(k, 6) * installment[idx] * 0.15
        frames.append(
            pd.DataFrame(
                {
                    "gf_loan_id": loans["gf_loan_id"].values[idx],
                    "gf_cutoff_date": c64,
                    "gf_outstanding_amount": np.round(outstanding, 2),
                    "gf_days_past_due_number": days.astype(int),
                    "gf_unpaid_installments_number": k.astype(int),
                    "gf_unpaid_amount": np.round(k * installment[idx], 2),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def generate() -> MockDataset:
    rng = np.random.default_rng(SEED)
    branches = build_branches(rng)
    customers = build_customers(rng, branches)
    accounts = build_accounts(rng, customers)
    balances = build_balances(rng, accounts)
    movements = build_movements(rng, accounts)
    cards = build_cards(rng, customers, accounts)
    card_txns = build_card_transactions(rng, cards)
    loans = build_loans(rng, customers)
    delinquency = build_delinquency(rng, loans)

    def public(df: pd.DataFrame) -> pd.DataFrame:
        return df[[c for c in df.columns if not c.startswith("_")]].reset_index(drop=True)

    accounts_public = accounts[
        ["gf_account_id", "gf_customer_id", "gf_branch_id", "gf_product_type_desc",
         "gf_opening_date", "gf_cancellation_date", "gf_currency_id"]
    ]
    return MockDataset(
        tables={
            "t_pred_branches": public(branches),
            "t_pcli_customers": public(customers),
            "t_pcta_accounts": public(accounts_public),
            "t_pcta_account_balances_monthly": public(balances),
            "t_pcta_account_movements": public(movements),
            "t_pmpg_cards": public(cards),
            "t_pmpg_card_transactions": public(card_txns),
            "t_prsg_loans": public(loans),
            "t_prsg_loan_delinquency": public(delinquency),
        }
    )
