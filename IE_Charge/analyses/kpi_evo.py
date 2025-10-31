"""Rebuild the Charges.kpi_evo table with updated success rates.

This script recalculates the monthly success rate (taux de réussite) for each
site stored in the ``Charges.kpi_sessions`` table. Charges that end with an
error classified as "Fin de charge" are now considered successful, matching the
new business expectation.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.mysql import insert


DB_CONFIG_KPI = {
    "host": "162.19.251.55",
    "port": 3306,
    "user": "nidec",
    "password": "MaV38f5xsGQp83",
    "database": "Charges",
}


def _build_engine(config: dict):
    """Create a SQLAlchemy engine using the provided configuration."""

    return create_engine(
        "mysql+pymysql://{user}:{password}@{host}:{port}/{database}".format(**config)
    )


engine_kpi = _build_engine(DB_CONFIG_KPI)


def fetch_sessions() -> pd.DataFrame:
    """Return session information needed to compute success rates."""

    query = text(
        """
        SELECT
            `Site`,
            `Datetime start` AS dt_start,
            `is_ok`,
            `moment`
        FROM Charges.kpi_sessions
        WHERE `Datetime start` IS NOT NULL
        """
    )

    df = pd.read_sql_query(query, con=engine_kpi)
    if df.empty:
        return df

    df["dt_start"] = pd.to_datetime(df["dt_start"], errors="coerce")
    df = df.dropna(subset=["dt_start"]).copy()

    site_series = df["Site"].astype(str).str.strip()
    site_series = site_series.replace("", "Unknown")
    site_series = site_series.mask(
        site_series.str.lower().isin({"none", "nan"}), "Unknown"
    )
    df["Site"] = site_series

    return df


def classify_success(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the success flag including end-of-charge errors as successes."""

    if df.empty:
        return df

    is_ok = pd.to_numeric(df.get("is_ok"), errors="coerce").fillna(0).astype(int)
    moment = df.get("moment").astype(str).str.strip().str.lower()
    fin_de_charge = moment.eq("fin de charge")
    df = df.copy()
    df["is_success"] = is_ok.eq(1) | (~is_ok.eq(1) & fin_de_charge)
    df["mois"] = df["dt_start"].dt.year.mul(100).add(df["dt_start"].dt.month)
    df["mois"] = df["mois"].astype(int)
    return df


def aggregate_success(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate success rate per site and month."""

    if df.empty:
        return df

    grouped = (
        df.groupby(["Site", "mois"], as_index=False)["is_success"].agg(
            total="count", successes="sum"
        )
    )

    grouped["tr"] = (
        grouped["successes"].div(grouped["total"].replace(0, pd.NA)).mul(100).round(2)
    ).fillna(0)

    return grouped[["Site", "mois", "tr"]]


def chunk_records(records: Iterable[dict], chunk_size: int = 500) -> Iterable[list[dict]]:
    """Yield records in chunks to avoid oversized INSERT statements."""

    chunk: list[dict] = []
    for record in records:
        chunk.append(record)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def upsert_kpi_evo(data: pd.DataFrame) -> None:
    """Replace the content of Charges.kpi_evo with the provided aggregates."""

    if data.empty:
        print("ℹ️  Aucun enregistrement à insérer dans Charges.kpi_evo")
        return

    metadata = MetaData()
    kpi_evo_table = Table(
        "kpi_evo", metadata, autoload_with=engine_kpi, schema="Charges"
    )

    with engine_kpi.begin() as conn:
        conn.execute(text("DELETE FROM Charges.kpi_evo"))
        insert_stmt = insert(kpi_evo_table)
        for chunk in chunk_records(data.to_dict("records")):
            conn.execute(
                insert_stmt.on_duplicate_key_update(tr=insert_stmt.inserted.tr),
                chunk,
            )

    print(
        f"✅ Table Charges.kpi_evo mise à jour avec {len(data)} lignes (Fin de charge considéré réussi)"
    )


def main() -> None:
    sessions = fetch_sessions()
    sessions = classify_success(sessions)
    aggregates = aggregate_success(sessions)
    upsert_kpi_evo(aggregates)


if __name__ == "__main__":
    main()
