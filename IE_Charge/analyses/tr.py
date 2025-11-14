#!/usr/bin/env python3
"""
Script de calcul du taux de réussite par point de charge et par site.
Ignore la colonne is_ok et utilise des critères personnalisés définis par l'utilisateur.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

# Configuration de la base de données
DB_CONFIG_KPI = {
    "host": "162.19.251.55",
    "port": 3306,
    "user": "nidec",
    "password": "MaV38f5xsGQp83",
    "database": "Charges",
}


def _build_engine(config: dict):
    """Construit un moteur SQLAlchemy à partir de la configuration."""
    return create_engine(
        "mysql+pymysql://{user}:{password}@{host}:{port}/{database}".format(**config)
    )


def get_user_criteria():
    """Demande à l'utilisateur de définir les critères de réussite."""
    print("=" * 80)
    print("DÉFINITION DES CRITÈRES DE RÉUSSITE")
    print("=" * 80)
    print("\nUne charge est considérée comme réussie si :")
    print("  - Energy (kWh) > X")
    print("  - Durée (minutes) > Y")
    print()

    while True:
        try:
            energy_min = float(input("Seuil minimum d'énergie (kWh) : "))
            if energy_min < 0:
                print("⚠️  L'énergie doit être positive. Réessayez.")
                continue
            break
        except ValueError:
            print("⚠️  Veuillez entrer un nombre valide.")

    while True:
        try:
            duration_min = float(input("Seuil minimum de durée (minutes) : "))
            if duration_min < 0:
                print("⚠️  La durée doit être positive. Réessayez.")
                continue
            break
        except ValueError:
            print("⚠️  Veuillez entrer un nombre valide.")

    print()
    print(f"✅ Critères définis : Energy > {energy_min} kWh ET Durée > {duration_min} min")
    print()

    return energy_min, duration_min


def fetch_sessions_data(engine):
    """Récupère les données de la table kpi_sessions."""
    query = """
        SELECT
            Site,
            PDC,
            `Datetime start`,
            `Datetime end`,
            `Energy (Kwh)` as Energy
        FROM kpi_sessions
        WHERE Site IS NOT NULL
          AND PDC IS NOT NULL
          AND `Datetime start` IS NOT NULL
          AND `Datetime end` IS NOT NULL
    """

    print("📥 Récupération des données de kpi_sessions...")
    df = pd.read_sql(query, con=engine)
    print(f"✅ {len(df)} sessions récupérées")

    return df


def calculate_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule la durée de chaque session en minutes."""
    df['Datetime start'] = pd.to_datetime(df['Datetime start'], errors='coerce')
    df['Datetime end'] = pd.to_datetime(df['Datetime end'], errors='coerce')

    # Durée en minutes
    df['Duration_min'] = (df['Datetime end'] - df['Datetime start']).dt.total_seconds() / 60
    df['Duration_min'] = df['Duration_min'].fillna(0).clip(lower=0)

    # Nettoyer l'énergie
    df['Energy'] = pd.to_numeric(df['Energy'], errors='coerce').fillna(0)

    return df


def apply_success_criteria(df: pd.DataFrame, energy_min: float, duration_min: float) -> pd.DataFrame:
    """Applique les critères de réussite définis par l'utilisateur."""
    df['is_success'] = (df['Energy'] > energy_min) & (df['Duration_min'] > duration_min)
    return df


def calculate_success_rate_by_pdc(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule le taux de réussite par point de charge."""
    grouped = df.groupby(['Site', 'PDC']).agg(
        Total=('is_success', 'count'),
        Success=('is_success', 'sum')
    ).reset_index()

    grouped['Success_Rate_%'] = (grouped['Success'] / grouped['Total'] * 100).round(2)
    grouped = grouped.sort_values(['Site', 'PDC'])

    return grouped


def calculate_success_rate_by_site(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule le taux de réussite moyen par site."""
    grouped = df.groupby('Site').agg(
        Total=('is_success', 'count'),
        Success=('is_success', 'sum')
    ).reset_index()

    grouped['Success_Rate_%'] = (grouped['Success'] / grouped['Total'] * 100).round(2)
    grouped = grouped.sort_values('Site')

    return grouped


def display_results(pdc_results: pd.DataFrame, site_results: pd.DataFrame, energy_min: float, duration_min: float):
    """Affiche les résultats dans la console."""
    print("\n" + "=" * 80)
    print("RÉSULTATS - TAUX DE RÉUSSITE")
    print("=" * 80)
    print(f"\nCritères utilisés : Energy > {energy_min} kWh ET Durée > {duration_min} min")

    print("\n" + "-" * 80)
    print("TAUX DE RÉUSSITE PAR SITE (MOYENNE)")
    print("-" * 80)
    print(f"\n{'Site':<30} {'Total':>10} {'Réussies':>10} {'Taux (%)':<10}")
    print("-" * 80)

    for _, row in site_results.iterrows():
        print(f"{row['Site']:<30} {int(row['Total']):>10} {int(row['Success']):>10} {row['Success_Rate_%']:>9.2f}%")

    print("\n" + "-" * 80)
    print("TAUX DE RÉUSSITE PAR POINT DE CHARGE")
    print("-" * 80)
    print(f"\n{'Site':<30} {'PDC':>5} {'Total':>10} {'Réussies':>10} {'Taux (%)':<10}")
    print("-" * 80)

    for _, row in pdc_results.iterrows():
        pdc_str = str(int(row['PDC'])) if pd.notna(row['PDC']) else 'N/A'
        print(f"{row['Site']:<30} {pdc_str:>5} {int(row['Total']):>10} {int(row['Success']):>10} {row['Success_Rate_%']:>9.2f}%")

    print("\n" + "=" * 80)


def main():
    """Fonction principale du script."""
    # Demander les critères à l'utilisateur
    energy_min, duration_min = get_user_criteria()

    # Connexion à la base de données
    engine = _build_engine(DB_CONFIG_KPI)

    # Récupérer les données
    df = fetch_sessions_data(engine)

    if df.empty:
        print("⚠️  Aucune donnée trouvée dans kpi_sessions.")
        return

    # Calculer la durée
    print("🔄 Calcul des durées...")
    df = calculate_duration(df)

    # Appliquer les critères de réussite
    print(f"🔄 Application des critères (Energy > {energy_min} kWh, Durée > {duration_min} min)...")
    df = apply_success_criteria(df, energy_min, duration_min)

    # Calculer les taux de réussite
    print("📊 Calcul des taux de réussite...")
    pdc_results = calculate_success_rate_by_pdc(df)
    site_results = calculate_success_rate_by_site(df)

    # Afficher les résultats
    display_results(pdc_results, site_results, energy_min, duration_min)

    # Option d'export
    export = input("\n💾 Voulez-vous exporter les résultats en CSV ? (o/n) : ").strip().lower()
    if export in ('o', 'oui', 'y', 'yes'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdc_file = f"exports/taux_reussite_pdc_{timestamp}.csv"
        site_file = f"exports/taux_reussite_site_{timestamp}.csv"

        import os
        os.makedirs("exports", exist_ok=True)

        pdc_results.to_csv(pdc_file, index=False, encoding='utf-8-sig')
        site_results.to_csv(site_file, index=False, encoding='utf-8-sig')

        print(f"✅ Résultats exportés :")
        print(f"   - Par PDC : {pdc_file}")
        print(f"   - Par site : {site_file}")


if __name__ == "__main__":
    main()
