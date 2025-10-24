from sqlalchemy import create_engine, inspect, text

engine = create_engine("mysql+pymysql://AdminNidec:u6Ehe987XBSXxa4@141.94.31.144:3306/indicator")

SCHEMA = "indicator"
TABLE_PREFIX = "kpi_"

# Clés uni
UNIQUE_KEYS = {
    "kpi_sessions": ["ID"],
    "kpi_charges_mac": ["ID"],
    "kpi_multi_attempts_hour": ["ID_ref", "Date_heure"],
    "kpi_suspicious_under_1kwh": ["ID"],
    "kpi_durations_site_daily": ["Site", "day"],
    "kpi_durations_pdc_daily": ["PDC", "day"],
    "kpi_charges_monthly": ["month"],
    "kpi_charges_daily": ["day"],
    "kpi_charges_monthly_by_site": ["Site", "month"],
    "kpi_charges_daily_by_site": ["Site", "day"],
    "kpi_charges_daily_by_site_pdc": ["Site", "PDC", "day"],
    "kpi_evi_combo_long": ["PDC", "Datetime start", "moment"],
    "kpi_evi_combo_by_site": ["Site"],
    "kpi_evi_combo_by_site_pdc": ["Site", "PDC"],
}

def create_unique_constraints():
    inspector = inspect(engine)

    with engine.connect() as conn:
        for table, columns in UNIQUE_KEYS.items():
            try:
                existing_columns = {col["name"]: str(col["type"]).lower() for col in inspector.get_columns(table, schema=SCHEMA)}
            except Exception as e:
                print(f"⚠️ Table introuvable ou erreur de structure : {table} → {e}")
                continue

            column_defs = []
            for col in columns:
                if col not in existing_columns:
                    print(f"❌ Colonne absente dans {table} : {col}")
                    break

                col_type = existing_columns[col]
                if any(t in col_type for t in ["text", "char", "varchar", "blob"]):
                    column_defs.append(f"`{col}`(20)")
                else:
                    column_defs.append(f"`{col}`")
            else:
                index_name = f"uniq_{'_'.join(columns).lower()}"
                col_list = ", ".join(column_defs)

                stmt = f"""
                    ALTER TABLE `{SCHEMA}`.`{table}`
                    ADD UNIQUE KEY `{index_name}` ({col_list});
                """
                try:
                    conn.execute(text(stmt))
                    print(f"✅ UNIQUE ajouté : {table} ({', '.join(columns)})")
                except Exception as e:
                    print(f"❌ Erreur ajout UNIQUE sur {table} → {e}")

if __name__ == "__main__":
    print("\n Étape : Ajout des contraintes d’unicité (UNIQUE)...\n")
    create_unique_constraints()