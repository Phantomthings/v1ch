import pandas as pd
from sqlalchemy import create_engine, text

# ---------- Config ----------
DB_CONFIG_SRC = {
    "host": "141.94.31.144",
    "port": 3306,
    "user": "AdminNidec",
    "password": "u6Ehe987XBSXxa4",
    "database": "indicator",    
}

DB_CONFIG_DEST = {
    "host": "162.19.251.55",
    "port": 3306,
    "user": "nidec",
    "password": "MaV38f5xsGQp83",
    "database": "Charges",      
}

TABLE_NAME = "mac_lookup"
CHUNK_SIZE = 5_000
EMPTY_DEST_BEFORE = True  # met à False si tu veux faire un append

# ---------- Connexions ----------
engine_src = create_engine(
    f"mysql+pymysql://{DB_CONFIG_SRC['user']}:{DB_CONFIG_SRC['password']}"
    f"@{DB_CONFIG_SRC['host']}:{DB_CONFIG_SRC['port']}/{DB_CONFIG_SRC['database']}"
)
engine_dest = create_engine(
    f"mysql+pymysql://{DB_CONFIG_DEST['user']}:{DB_CONFIG_DEST['password']}"
    f"@{DB_CONFIG_DEST['host']}:{DB_CONFIG_DEST['port']}/{DB_CONFIG_DEST['database']}"
)

print("📥 Lecture de la table source indicator.mac_lookup ...")
df = pd.read_sql(f"SELECT * FROM `{TABLE_NAME}`", engine_src)
print(f"✅ {len(df)} lignes lues depuis indicator.{TABLE_NAME}.")

# ---------- Vider la destination (optionnel) ----------
if EMPTY_DEST_BEFORE:
    with engine_dest.begin() as conn:
        # SQLAlchemy 2.x => utiliser text()
        conn.execute(text(f"DELETE FROM `{DB_CONFIG_DEST['database']}`.`{TABLE_NAME}`"))
    print(f"🧹 Table {DB_CONFIG_DEST['database']}.{TABLE_NAME} vidée.")

# ---------- Ecriture en chunks ----------
if df.empty:
    print("ℹ️ Aucune ligne à insérer. Fin.")
else:
    # to_sql créera la table si elle n'existe pas (avec un schéma dérivé du DataFrame)
    df.to_sql(
        TABLE_NAME,
        engine_dest,
        if_exists="append",
        index=False,
        chunksize=CHUNK_SIZE,
        method="multi",
    )
    print(f"📤 {len(df)} lignes insérées dans {DB_CONFIG_DEST['database']}.{TABLE_NAME}.")

print("✅ Copie terminée.")
