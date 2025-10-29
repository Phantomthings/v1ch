from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.mysql import insert

# Connexion SQL
DB_CONFIG_KPI = {
    "host": "162.19.251.55",
    "port": 3306,
    "user": "nidec",
    "password": "MaV38f5xsGQp83",
    "database": "Charges",
}

DB_CONFIG_CHARGE = {
    "host": "162.19.251.55",
    "port": 3306,
    "user": "nidec",
    "password": "MaV38f5xsGQp83",
    "database": "nw_borne",
}


def _build_engine(config: dict):
    return create_engine(
        "mysql+pymysql://{user}:{password}@{host}:{port}/{database}".format(**config)
    )


engine_kpi = _build_engine(DB_CONFIG_KPI)
engine_charge = _build_engine(DB_CONFIG_CHARGE)

FINAL_PATH = Path("data/charge.csv")
KPIS_XLSX = Path("data/kpis.xlsx")
KPIS_MAC = Path("data/MAC.xlsx")
TMP_XLSX = KPIS_XLSX.with_name("kpis_tmp.xlsx")

TABLES_TO_SAVE = {
    "sessions",
    "charges_daily_by_site",
    "durations_pdc_daily",
    "durations_site_daily",
    "suspicious_under_1kwh",
    "multi_attempts_hour",
    "charges_mac",
    "evi_combo_by_site_pdc",
    "evi_combo_long",
    "evi_combo_by_site",
}

# ✅ CORRECTION DES CLÉS UNIQUES
# Elles doivent correspondre EXACTEMENT aux colonnes de GROUP BY dans chaque fonction
UNIQUE_KEYS = {
    "kpi_sessions": ["ID"],
    "kpi_charges_mac": ["ID"],
    "kpi_multi_attempts_hour": ["ID_ref", "Date_heure"],
    "kpi_suspicious_under_1kwh": ["ID"],
    "kpi_durations_site_daily": ["Site", "day"],
    "kpi_durations_pdc_daily": ["Site", "PDC", "day"],  # ✅ Ajout de "Site"
    "kpi_charges_daily_by_site": ["Site", "day", "Status"],  # ✅ Ajout de "Status"
    "kpi_evi_combo_long": ["Site", "PDC", "Datetime start", "EVI_Code", "EVI_Step"],  # ✅ Plus précis
    "kpi_evi_combo_by_site": ["Site", "EVI_Code", "EVI_Step"],  # ✅ Ajout dimensions agrégation
    "kpi_evi_combo_by_site_pdc": ["Site", "PDC", "EVI_Code", "EVI_Step"],  # ✅ Ajout dimensions agrégation
}

SITE_CODE_COL = "Name Project"
SITE_COL = "Site"
PDC_COL = "PDC"
DATE_START = "Datetime start"
DATE_END = "Datetime end"

DS_PC = "Downstream Code PC"
EVI_CODE = "EVI Error Code"
EVI_MOMENT = "EVI Status during error"
SOC_COL = "State of charge(0:good, 1:error)"

SITE_MAP = {
    "7571": "Orignolles",
    "7796": "Meru",
    "7797": "Charleval",
    "7798": "Triel",
    "7800": "Saujon",
    "7803": "Cierzac",
    "7804": "Os Marsillon",
    "7809": "St Pere en retz",
    "7812": "Hagetmau",
    "7813": "Biscarosse",
    "7814": "Auriolles",
    "7818": "Verneuil",
    "7819": "Allaire",
    "7825": "Vezin",
    "7828": "Pontchateau",
    "7833": "Pontfaverger",
    "001": "Baud", "003": "Maurs", "050": "Mezidon", "051": "Derval", "054": "Campagne", "057": "Mailly le Chateau", "062": "Winnezeele", "063": "Diges", "065": "Vernouillet", "067": "Orbec", "071": "St Renan", "079": "Molompize", "081": "Carquefou", "083": "Vaupillon", "085": "Pleumartin", "086": "Caumont sur Aure", "087": "Getigne", "088": "Chinon", "091": "La Roche sur Yon", "093": "Aubigne sur Layon", "094": "Bonvillet", "096": "Rambervillers", "099": "Blere", "100": "Plouasne", "108": "Champniers", "112": "Nissan Lez Enserune", "114": "Combourg", "115": "Vimoutiers", "118": "Beaumont de Lomagne", "121": "Sueves", "122": "Maen Roch", "124": "St Leon sur L Isle", "125": "Mirecourt", "128": "La Voge les Bains", "130": "Amanvillers", "131": "Guerlesquin", "134": "Guerande", "135": "Riscle", "139": "Avrille", "142": "Domfront", "149": "Couesmes", "156": "Ste Catherine", "160": "Andel", "161": "Chazey Bons", "163": "Lauzerte", "165": "Trie la ville", "166": "Hambach", "167": "Beaugency", "168": "Carcassonne", "174": "Sable sur Sarthe", "179": "Taden", "184": "Rue", "185": "Quevilloncourt", "187": "St Victor de Morestel", "191": "St Hilaire du Harcouet", "196": "Hémonstoir", "197": "Amily", "199": "Henrichemont", "203": "Couleuvre", "208": "St Pierre le Moutier 2", "209": "Bourbon L Archambaut", "210": "Brou", "211": "Neulise", "214": "St Jean le vieux", "217": "Periers", "218": "Quievrecourt", "221": "Chazelle sur Lyon", "222": "Montverdun", "223": "Dormans", "227": "Glonville 2", "230": "Montalieu Vercieu", "234": "Nesle Normandeuse", "240": "Noyal Pontivy", "246": "Vitre 2", "247": "St Amour", "250": "Dourdan", "254": "Roanne", "259": "Plufur", "266": "Boinville en Mantois", "269": "Loche", "272": "Bonnieres sur Seine", "273": "Piffonds", "274": "St Benin d Azy", "276": "Niort St Florent", "281": "Chauffailles", "282": "St Vincent d Autejac", "283": "Culhat", "289": "Loireauxence", "292": "Reuil", "301": "Coteaux sur Loire", "304": "Le Mans 2", "311": "Chantrigne", "313": "St Thelo", "314": "St Pierre la cour", "317": "Nievroz", "318": "Val Revermont", "320": "Mondoubleau", "321": "Kernoues", "322": "Yvetot Bocage", "324": "Douchy Montcorbon", "328": "Sully sur Loire B", "330": "Vincey", "336": "Ville en Vermois", "337": "Virandeville", "339": "Reims", "340": "Reims B", "342": "Charge", "343": "St Benoit la Foret", "349": "Dombrot le Sec", "352": "Riorges", "362": "Montauban B", "365": "Dogneville 2", "366": "Brieulles sur meuse", "368": "Melesse", "372": "Pujaudran", "374": "Plouye", "376": "Dampierre en Burly", "381": "Dommartin les Remiremont", "382": "St Igny de Roche", "384": "Guengat", "386": "Epeigne sur deme 2", "388": "Maiche", "391": "Wittenheim", "394": "Lacres", "395": "Trelivan", "397": "Vironvay", "399": "Abbeville les Conflans", "401": "Orgeval", "402": "Mantes la Ville", "403": "Liny devant Dun B", "412": "St Leger sur Roanne", "414": "Mairy Mainville",
}


def get_last_update_date(table_name: str) -> datetime:
    """Récupère la dernière date de mise à jour pour une table."""
    schema_name = "Charges"
    full_table = f"{schema_name}.kpi_{table_name.lower()}"
    
    # Mapping des colonnes de date selon la table
    date_columns = {
        "sessions": "Datetime start",
        "charges_daily_by_site": "day",
        "durations_pdc_daily": "day",
        "durations_site_daily": "day",
        "suspicious_under_1kwh": "Datetime start",
        "multi_attempts_hour": "Date_heure",
        "charges_mac": "Datetime start",
        "evi_combo_long": "Datetime start",
        "evi_combo_by_site": None,
        "evi_combo_by_site_pdc": None,
    }
    
    date_col = date_columns.get(table_name)
    
    if date_col is None:
        return datetime(2025, 2, 6)
    
    try:
        query = f"SELECT MAX(`{date_col}`) as max_date FROM {full_table}"
        with engine_kpi.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            if result and result[0]:
                return result[0] - timedelta(days=2)
    except Exception as e:
        print(f"⚠️ Impossible de récupérer la dernière date pour {table_name}: {e}")
    
    return datetime(2025, 2, 6)


def classify_errors(df: pd.DataFrame) -> pd.DataFrame:
    soc_series = pd.to_numeric(df.get(SOC_COL), errors="coerce")
    if isinstance(soc_series, pd.Series):
        soc = soc_series.fillna(0).astype(int)
    else:
        soc = pd.Series(0, index=df.index, dtype=int)
    df["is_ok"] = soc.eq(0)

    fail_mask = soc.eq(1)

    ds_pc_val = pd.to_numeric(df.get(DS_PC), errors="coerce")
    ds_pc_val = (
        ds_pc_val.fillna(0).astype(int)
        if isinstance(ds_pc_val, pd.Series)
        else pd.Series(0, index=df.index, dtype=int)
    )
    evi_code_val = pd.to_numeric(df.get(EVI_CODE), errors="coerce")
    evi_code_val = (
        evi_code_val.fillna(0).astype(int)
        if isinstance(evi_code_val, pd.Series)
        else pd.Series(0, index=df.index, dtype=int)
    )

    df["type_erreur"] = np.select(
        [
            fail_mask
            & ((ds_pc_val.eq(8192)) | ((ds_pc_val.eq(0)) & (evi_code_val.ne(0)))),
            fail_mask & ((ds_pc_val.ne(0)) & (ds_pc_val.ne(8192))),
        ],
        ["Erreur_EVI", "Erreur_DownStream"],
        default="Erreur_Unknow_S",
    )

    def map_moment(val):
        try:
            val = int(val)
        except Exception:
            return "Fin de charge"
        if 1 <= val <= 2:
            return "Init"
        if 4 <= val <= 6:
            return "Lock Connector"
        if val == 7:
            return "CableCheck"
        if val == 8:
            return "Charge"
        if val > 8:
            return "Fin de charge"
        return "Fin de charge"

    evi_moment_val = pd.to_numeric(df.get(EVI_MOMENT), errors="coerce")
    evi_moment_val = (
        evi_moment_val.fillna(0).astype(int)
        if isinstance(evi_moment_val, pd.Series)
        else pd.Series(0, index=df.index, dtype=int)
    )

    def map_moment_general(row):
        if row["type_erreur"] in ("Erreur_EVI", "Erreur_DownStream"):
            try:
                val = int(row[EVI_MOMENT])
            except Exception:
                return "Unknown"
            if val == 0:
                return "Fin de charge"
            return map_moment(val)
        return "Unknown"

    df["moment"] = df.apply(map_moment_general, axis=1)
    return df


def _safe_dt(s):
    return pd.to_datetime(s, errors="coerce")


def _date_str_from_rows(idx, dt_end, dt_start):
    if pd.isna(idx):
        return ""
    try:
        d = dt_end.loc[idx]
        if pd.isna(d):
            d = dt_start.loc[idx]
        return d.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(d) else ""
    except Exception:
        return ""


def build_evi_combo_tables(df: pd.DataFrame) -> dict:
    site_col = "Site" if "Site" in df.columns else "Name Project"
    soc_col = "State of charge(0:good, 1:error)"
    code_col = "EVI Error Code"
    step_col = "EVI Status during error"
    pdc_col = "PDC" if "PDC" in df.columns else None
    start = "Datetime start"

    soc = pd.to_numeric(df.get(soc_col, 0), errors="coerce").fillna(0).astype(int)
    code = pd.to_numeric(df.get(code_col, 0), errors="coerce").fillna(0).astype(int)
    step = pd.to_numeric(df.get(step_col, 0), errors="coerce").fillna(0).astype(int)

    fail = df.loc[soc.eq(1)].copy()
    fail["EVI_Code"] = code.loc[fail.index]
    fail["EVI_Step"] = step.loc[fail.index]
    mask_combo = (fail["EVI_Code"].ne(0)) | (fail["EVI_Step"].ne(0))

    def _map_step_to_moment_int(s: int) -> str:
        if 1 <= s <= 2:
            return "Init"
        if 4 <= s <= 6:
            return "Lock Connector"
        if s == 7:
            return "CableCheck"
        if s == 8:
            return "Charge"
        if s > 8:
            return "Fin de charge"
        return "Unknown"

    evi_long = fail.loc[
        mask_combo,
        [site_col, start, "EVI_Code", "EVI_Step"] + ([pdc_col] if pdc_col else []),
    ].copy()
    evi_long.rename(columns={site_col: "Site"}, inplace=True)
    evi_long["Datetime start"] = pd.to_datetime(evi_long["Datetime start"], errors="coerce")
    evi_long["step_num"] = pd.to_numeric(evi_long["EVI_Step"], errors="coerce").fillna(-1).astype(int)
    evi_long["code_num"] = pd.to_numeric(evi_long["EVI_Code"], errors="coerce").fillna(-1).astype(int)
    evi_long["moment"] = evi_long["step_num"].map(_map_step_to_moment_int)
    
    # ✅ Agrégation par site (avec toutes les dimensions)
    by_site = (
        evi_long.groupby(["Site", "EVI_Code", "EVI_Step"], as_index=False)
        .size()
        .rename(columns={"size": "Occurrences"})
    )
    by_site["%_site"] = (
        by_site["Occurrences"]
        / by_site.groupby("Site")["Occurrences"].transform("sum")
        * 100
    ).round(2)

    if pdc_col:
        # ✅ Agrégation par site/PDC (avec toutes les dimensions)
        by_site_pdc = (
            evi_long.groupby(["Site", "PDC", "EVI_Code", "EVI_Step"], as_index=False)
            .size()
            .rename(columns={"size": "Occurrences"})
        )
        by_site_pdc["%_site_pdc"] = (
            by_site_pdc["Occurrences"]
            / by_site_pdc.groupby(["Site", "PDC"])["Occurrences"].transform("sum")
            * 100
        ).round(2)
    else:
        by_site_pdc = pd.DataFrame(
            columns=["Site", "PDC", "EVI_Code", "EVI_Step", "Occurrences", "%_site_pdc"]
        )

    return {
        "evi_combo_long": evi_long.sort_values(["Site", "Datetime start"]),
        "evi_combo_by_site": by_site.sort_values(["Site", "Occurrences"], ascending=[True, False]),
        "evi_combo_by_site_pdc": by_site_pdc.sort_values(
            ["Site", "PDC", "Occurrences"], ascending=[True, True, False]
        ),
    }


def build_durations_daily(df: pd.DataFrame) -> dict:
    site_col = "Site" if "Site" in df.columns else ("Name Project" if "Name Project" in df.columns else "Site")
    if site_col not in df.columns:
        df[site_col] = "Unknown"

    dt_s = pd.to_datetime(df.get("Datetime start"), errors="coerce")
    dt_e = pd.to_datetime(df.get("Datetime end"), errors="coerce")
    dur_min = (dt_e - dt_s).dt.total_seconds().div(60)
    dur_min = dur_min.mask(dur_min < 0, 0).fillna(0)

    ok_mask = df.get("is_ok", False).astype(bool)
    ok = df[ok_mask].copy()
    ok["_day"] = pd.to_datetime(ok["Datetime start"], errors="coerce").dt.floor("D")
    ok["dur_min"] = dur_min.loc[ok.index].fillna(0)

    base = df.copy()
    base["_day"] = pd.to_datetime(base.get("Datetime start"), errors="coerce").dt.floor("D")
    base = base[base["_day"].notna()].copy()

    # ✅ Agrégation par site/jour
    dur_site_daily = (
        ok.groupby([site_col, "_day"], dropna=False)["dur_min"].sum().reset_index()
        if not ok.empty
        else pd.DataFrame(columns=[site_col, "_day", "dur_min"])
    )

    if not base.empty:
        all_site_days = base[[site_col, "_day"]].drop_duplicates()
        dur_site_daily = all_site_days.merge(
            dur_site_daily,
            on=[site_col, "_day"],
            how="left",
        )
    dur_site_daily = dur_site_daily.rename(columns={site_col: "Site", "_day": "day"})
    dur_site_daily["dur_min"] = dur_site_daily["dur_min"].fillna(0)

    if "PDC" in ok.columns:
        # ✅ Agrégation par site/PDC/jour
        dur_pdc_daily = (
            ok.groupby([site_col, "PDC", "_day"], dropna=False)["dur_min"].sum().reset_index()
            if not ok.empty
            else pd.DataFrame(columns=[site_col, "PDC", "_day", "dur_min"])
        )
        if "PDC" in base.columns:
            all_site_pdc_days = (
                base[[site_col, "PDC", "_day"]]
                .dropna(subset=["_day"])
                .drop_duplicates()
            )
            dur_pdc_daily = all_site_pdc_days.merge(
                dur_pdc_daily,
                on=[site_col, "PDC", "_day"],
                how="left",
            )
        dur_pdc_daily = dur_pdc_daily.rename(columns={site_col: "Site", "_day": "day"})
        dur_pdc_daily["dur_min"] = dur_pdc_daily["dur_min"].fillna(0)
    else:
        dur_pdc_daily = pd.DataFrame(columns=["Site", "PDC", "day", "dur_min"])

    return {
        "durations_site_daily": dur_site_daily.sort_values(["Site", "day"]),
        "durations_pdc_daily": dur_pdc_daily.sort_values(["Site", "PDC", "day"]),
    }


def _norm_mac_full(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    for ch in [":", "-", " "]:
        s = s.replace(ch, "")
    s = "".join(ch for ch in s if ch in "0123456789abcdef")
    return "" if (s == "" or all(ch == "0" for ch in s)) else s


def _norm_hex_frag(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    s = "".join(ch for ch in s if ch in "0123456789abcdef")
    if s == "" or all(ch == "0" for ch in s):
        return ""
    return s


def _compose_full_mac(
    row,
    c1_candidates=("mac_adress_1", "mac_address_1", "ac_adress_", "mac1"),
    c2_candidates=("mac_adress_2", "mac_address_2", "mac2"),
    c_single=("mac", "mac_address", "mac_adress"),
):
    def _get_first(colnames):
        for c in colnames:
            if c in row and pd.notna(row[c]):
                return row[c]
        return ""

    m1 = _norm_hex_frag(_get_first(c1_candidates))
    m2 = _norm_hex_frag(_get_first(c2_candidates))
    if m1 and m2:
        return f"{m1}{m2}"
    return _norm_mac_full(_get_first(c_single))


def _fmt_mac(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    for ch in (":", "-", " "):
        s = s.replace(ch, "")
    s = "".join(ch for ch in s if ch in "0123456789abcdef")
    if s == "":
        return ""
    if set(s) == {"0"}:
        return "00"
    if len(s) % 2 == 1:
        s = "0" + s
    pairs = [s[i : i + 2].upper() for i in range(0, len(s), 2)]
    return ":".join(pairs)


def build_charges_mac(df: pd.DataFrame, mac_lookup: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["mac"] = work.apply(_compose_full_mac, axis=1)
    if "mac_adress_1" in work.columns:
        work["mac_adress_1"] = work["mac_adress_1"].map(_norm_hex_frag)
    if "mac_adress_2" in work.columns:
        work["mac_adress_2"] = work["mac_adress_2"].map(_norm_hex_frag)
    work = work[work["mac"] != ""].copy()
    work["MAC Address"] = work["mac"].map(_fmt_mac)
    
    if not mac_lookup.empty:
        m = mac_lookup.copy()
        m["mac"] = (
            m["mac"].astype(str)
            .str.lower()
            .str.replace("0x", "", regex=False)
            .str.replace("[:\\- ]", "", regex=True)
        )
        exact = m[["mac", "Vehicle"]].drop_duplicates(subset=["mac"])
        work = work.merge(exact, on="mac", how="left", suffixes=("", "_from_exact"))
        m["prefix6"] = m["mac"].str[:6]
        p6 = m[["prefix6", "Vehicle"]].drop_duplicates(subset=["prefix6"])
        work["prefix6"] = work["mac"].str[:6]
        work = work.merge(p6, on="prefix6", how="left", suffixes=("", "_from_p6"))
        m["prefix4"] = m["mac"].str[:4]
        p4 = m[["prefix4", "Vehicle"]].drop_duplicates(subset=["prefix4"])
        work["prefix4"] = work["mac"].str[:4]
        work = work.merge(p4, on="prefix4", how="left", suffixes=("", "_from_p4"))
        work["Vehicle"] = (
            work["Vehicle"].fillna(work.pop("Vehicle_from_p6")).fillna(work.pop("Vehicle_from_p4"))
        )
        for tmp in ("prefix6", "prefix4"):
            if tmp in work.columns:
                work.drop(columns=[tmp], inplace=True)
    else:
        work["Vehicle"] = ""
    
    id_col = next((c for c in ["ID", "Id", "session_id", "Session ID"] if c in work.columns), None)
    if id_col is None:
        raise ValueError("Aucune colonne ID trouvée dans le DataFrame")
    elif id_col != "ID":
        work.rename(columns={id_col: "ID"}, inplace=True)
    
    site_col = "Site" if "Site" in work.columns else ("Name Project" if "Name Project" in work.columns else "Site")
    if site_col not in work.columns:
        work[site_col] = "Unknown"
    
    keep = [
        "ID",
        site_col,
        "Datetime start",
        "is_ok",
        "SOC Start",
        "SOC End",
        "mac_adress_1",
        "mac_adress_2",
        "mac",
        "MAC Address",
        "Vehicle",
    ]
    keep = [c for c in keep if c in work.columns]
    out = (
        work[keep]
        .rename(columns={site_col: "Site"})
        .sort_values(
            ["Site", "Datetime start", "ID"] if "Datetime start" in keep else ["Site", "ID"],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    return out


def resolve_session_id(df: pd.DataFrame) -> str:
    if "id" not in df.columns:
        raise ValueError("charge.csv doit contenir la colonne 'id'.")
    df.rename(columns={"id": "ID"}, inplace=True)
    df["ID"] = df["ID"].astype(str).str.strip()
    return "ID"


def build_suspicious_under_1kwh(df: pd.DataFrame) -> pd.DataFrame:
    site_col = "Site" if "Site" in df.columns else ("Name Project" if "Name Project" in df.columns else None)
    need = [
        "ID",
        site_col,
        "PDC" if "PDC" in df.columns else None,
        "Datetime start",
        "Datetime end",
        "Energy (Kwh)" if "Energy (Kwh)" in df.columns else None,
        "SOC Start" if "SOC Start" in df.columns else None,
        "SOC End" if "SOC End" in df.columns else None,
        "is_ok" if "is_ok" in df.columns else None,
    ]
    need = [c for c in need if c]
    out = df[need].copy()

    if site_col and site_col != "Site":
        out.rename(columns={site_col: "Site"}, inplace=True)

    out["ID"] = out["ID"].astype(str).str.strip()
    out["Datetime start"] = pd.to_datetime(out.get("Datetime start"), errors="coerce")
    out["Datetime end"] = pd.to_datetime(out.get("Datetime end"), errors="coerce")
    if "Energy (Kwh)" in out.columns:
        out["Energy (Kwh)"] = pd.to_numeric(out["Energy (Kwh)"], errors="coerce")

    mask_ok = out["is_ok"] if "is_ok" in out.columns else False
    mask_e = out["Energy (Kwh)"].lt(1) if "Energy (Kwh)" in out.columns else False
    out = out[mask_ok & mask_e].copy()

    for c in ("Datetime start", "Datetime end"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    keep = [
        "ID",
        "Site",
        "PDC",
        "Datetime start",
        "Datetime end",
        "Energy (Kwh)",
        "SOC Start",
        "SOC End",
    ]
    keep = [c for c in keep if c in out.columns]
    return (
        out[keep]
        .sort_values(["Site", "Datetime start", "ID"], na_position="last")
        .reset_index(drop=True)
    )


def build_multi_attempts_hour(df: pd.DataFrame) -> pd.DataFrame:
    site_col = "Site" if "Site" in df.columns else ("Name Project" if "Name Project" in df.columns else None)
    required_cols = ["Datetime start", "ID", "MAC Address"]
    if site_col is None or not all(col in df.columns for col in required_cols):
        return pd.DataFrame(
            columns=[
                "Site",
                "Heure",
                "Date_heure",
                "MAC Address",
                "tentatives",
                "PDC(s)",
                "1ère tentative",
                "Dernière tentative",
                "ID(s)",
                "ID_ref",
                "SOC start min",
                "SOC start max",
                "SOC end min",
                "SOC end max",
            ]
        )

    work = df.copy()
    work["MAC Address"] = work["MAC Address"].astype(str).str.strip().str.lower()
    work["MAC Address"] = work["MAC Address"].replace(["", "none", "nan", "nat"], np.nan)
    work = work.dropna(subset=["MAC Address"])
    work = work[work["MAC Address"].str.contains(r"[0-9a-f]{4,}", regex=True, na=False)]
    if work.empty:
        return pd.DataFrame(
            columns=[
                "Site",
                "Heure",
                "Date_heure",
                "MAC Address",
                "tentatives",
                "PDC(s)",
                "1ère tentative",
                "Dernière tentative",
                "ID(s)",
                "ID_ref",
            ]
        )
    work["Datetime start"] = pd.to_datetime(work["Datetime start"], errors="coerce")
    work = work.dropna(subset=["Datetime start"])
    work["Date_heure"] = work["Datetime start"].dt.floor("h")
    grp_keys = [site_col, "Date_heure", "MAC Address"]
    agg_dict = {
        "tentatives": ("PDC", "count"),
        "PDC(s)": ("PDC", lambda s: ", ".join(sorted({str(x) for x in s.dropna().astype(str)}))),
        "1ère tentative": ("Datetime start", "min"),
        "Dernière tentative": ("Datetime start", "max"),
    }
    if "SOC Start" in work.columns:
        agg_dict["SOC start min"] = ("SOC Start", "min")
        agg_dict["SOC start max"] = ("SOC Start", "max")
    if "SOC End" in work.columns:
        agg_dict["SOC end min"] = ("SOC End", "min")
        agg_dict["SOC end max"] = ("SOC End", "max")

    agg = work.groupby(grp_keys).agg(**agg_dict).reset_index()
    sorted_work = work.sort_values(["Date_heure", "Datetime start"])

    ids_list = (
        sorted_work.groupby(grp_keys)["ID"].apply(lambda s: ", ".join([str(x).strip() for x in s.astype(str)])).reset_index(name="ID(s)")
    )

    last_id = (
        sorted_work.groupby(grp_keys)["ID"].last().reset_index(name="ID_ref")
    )
    merge_keys = [site_col, "Date_heure", "MAC Address"]
    out = agg.merge(ids_list, on=merge_keys, how="left").merge(last_id, on=merge_keys, how="left")

    out = out.rename(columns={site_col: "Site"})
    out = out[out["tentatives"] >= 2].copy()
    if out.empty:
        return out
    out["Heure"] = out["Date_heure"].dt.strftime("%Y-%m-%d %H:00")
    base_cols = [
        "Site",
        "Heure",
        "Date_heure",
        "MAC Address",
        "tentatives",
        "PDC(s)",
        "1ère tentative",
        "Dernière tentative",
        "ID(s)",
        "ID_ref",
    ]
    soc_cols = [c for c in ["SOC start min", "SOC start max", "SOC end min", "SOC end max"] if c in out.columns]
    out = (
        out[base_cols + soc_cols]
        .sort_values(["Date_heure", "Site", "tentatives"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return out


def build_charges_daily_by_site(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df.get("Datetime start"), errors="coerce")
    ok = df.get("is_ok", False).astype(bool)

    site_col = "Site" if "Site" in df.columns else ("Name Project" if "Name Project" in df.columns else None)
    site = df.get(site_col, "Unknown")
    pdc = df["PDC"].astype(str) if "PDC" in df.columns else pd.Series(["—"] * len(df), index=df.index)

    base = pd.DataFrame(
        {
            "Site": site,
            "PDC": pdc,
            "dt": dt,
            "Status": np.where(ok, "OK", "NOK"),
        }
    ).dropna(subset=["dt"]).copy()

    base["day"] = base["dt"].dt.strftime("%Y-%m-%d")

    # ✅ Agrégation par site/jour/statut
    d_site = base.groupby(["Site", "day", "Status"], as_index=False).size().rename(columns={"size": "Nb"})

    return d_site.sort_values(["Site", "day"])


def fetch_charge_data(start_date: datetime = None) -> pd.DataFrame:
    """Récupère les données de charge depuis la dernière mise à jour."""
    if start_date is None:
        start_date = datetime(2025, 2, 6)
    
    query = f"""
        SELECT *
        FROM charge_info
        WHERE start_time >= '{start_date.strftime('%Y-%m-%d')}' AND Tri_charge = 1
        ORDER BY start_time ASC
    """
    df = pd.read_sql(query, engine_charge)

    rename_map = {
        "start_time": "Datetime start",
        "end_time": "Datetime end",
        "start_time_utc": "Datetime start utc",
        "end_time_utc": "Datetime end utc",
        "borne_id": "PDC",
        "energy": "Energy (Kwh)",
        "soc_debut": "SOC Start",
        "soc_fin": "SOC End",
        "duration": "Duration",
        "Etat": "State of charge(0:good, 1:error)",
        "mean_power": "Mean Power (Kw)",
        "max_power": "Max Power (Kw)",
        "status_upstream": "Status Upstream",
        "status_downstream": "Status Downstream",
        "upstream_ic": "Upstream Code IC",
        "upstream_pc": "Upstream Code PC",
        "EVI_error": "EVI Status",
        "EVi_status_at_error": "EVI Status during error",
        "Evi_error_code": "EVI Error Code",
        "downstream_ic": "Downstream Code IC",
        "downstream_pc": "Downstream Code PC",
        "project_num": "Id Project",
        "project_name": "Name Project",
        "mac": "MAC Address",
    }

    df.rename(columns=rename_map, inplace=True)
    return df


def delete_old_data(table_name: str, start_date: datetime):
    """Supprime les anciennes données depuis start_date pour éviter les doublons."""
    schema_name = "Charges"
    full_table = f"{schema_name}.kpi_{table_name.lower()}"
    
    date_columns = {
        "sessions": "Datetime start",
        "charges_daily_by_site": "day",
        "durations_pdc_daily": "day",
        "durations_site_daily": "day",
        "suspicious_under_1kwh": "Datetime start",
        "multi_attempts_hour": "Date_heure",
        "charges_mac": "Datetime start",
        "evi_combo_long": "Datetime start",
    }
    
    date_col = date_columns.get(table_name)
    
    if date_col is None:
        try:
            with engine_kpi.begin() as conn:
                conn.execute(text(f"DELETE FROM {full_table}"))
                print(f"🗑️  Toutes les données supprimées de {full_table} (table agrégée)")
        except Exception as e:
            print(f"⚠️ Erreur lors de la suppression des données de {table_name}: {e}")
    else:
        try:
            with engine_kpi.begin() as conn:
                delete_query = f"DELETE FROM {full_table} WHERE `{date_col}` >= '{start_date.strftime('%Y-%m-%d')}'"
                result = conn.execute(text(delete_query))
                print(f"🗑️  {result.rowcount} lignes supprimées de {full_table} depuis {start_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"⚠️ Erreur lors de la suppression des données de {table_name}: {e}")


def save_to_indicator(table_dict: dict, incremental: bool = True):
    """Sauvegarde les tables dans la base de données."""
    metadata = MetaData()
    
    for name, df in table_dict.items():
        if name not in TABLES_TO_SAVE:
            print(f"⚠️ Table ignorée (hors périmètre) : {name}")
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"⚠️ Table ignorée (vide) : {name}")
            continue

        table_name = f"kpi_{name.lower()}"
        schema_name = "Charges"

        try:
            table = Table(
                table_name, metadata, autoload_with=engine_kpi, schema=schema_name
            )
        except Exception as e:
            print(f"❌ Table non trouvée ou erreur chargement : {table_name} → {e}")
            continue

        df_cleaned = df.where(pd.notna(df), None)

        unique_cols = UNIQUE_KEYS.get(table_name)
        if unique_cols:
            missing_cols = [col for col in unique_cols if col not in df_cleaned.columns]
            if missing_cols:
                print(
                    "⚠️ Colonnes manquantes pour la déduplication de "
                    f"{schema_name}.{table_name} : {', '.join(missing_cols)}"
                )
            else:
                before = len(df_cleaned)

                complete_keys_mask = df_cleaned[unique_cols].notna().all(axis=1)

                # ✅ Vérifier s'il y a de VRAIS doublons
                true_duplicates = df_cleaned.loc[complete_keys_mask].duplicated(
                    subset=unique_cols, keep=False
                ).sum()

                dedup_part = (
                    df_cleaned.loc[complete_keys_mask]
                    .sort_values(unique_cols)
                    .drop_duplicates(subset=unique_cols, keep="last")
                )

                df_cleaned = pd.concat(
                    [dedup_part, df_cleaned.loc[~complete_keys_mask]],
                    ignore_index=True,
                )

                dropped = before - len(df_cleaned)
                if dropped > 0:
                    print(
                        f"ℹ️  {dropped} doublons supprimés avant insertion dans {schema_name}.{table_name} "
                        f"(clé unique : {', '.join(unique_cols)}) | Vrais doublons : {true_duplicates}"
                    )

        attempted_rows = len(df_cleaned)

        with engine_kpi.begin() as conn:
            try:
                if incremental:
                    stmt = insert(table).prefix_with("IGNORE")
                else:
                    stmt = insert(table)
                
                result = conn.execute(stmt, df_cleaned.to_dict(orient="records"))
                affected_rows = getattr(result, "rowcount", None)

                if affected_rows is None or affected_rows < 0:
                    affected_rows = attempted_rows

                message = (
                    f"✅ Table {'mise à jour' if incremental else 'insérée'} : "
                    f"{schema_name}.{table_name} ({affected_rows} lignes)"
                )

                if incremental and affected_rows != attempted_rows:
                    message += f" sur {attempted_rows} tentatives"

                print(message)
            except Exception as e:
                print(f"❌ Erreur insertion pour {table_name} → {e}")


def main():
    print("=" * 80)
    print("DÉMARRAGE DU PROCESSUS DE MISE À JOUR INCRÉMENTALE")
    print("=" * 80)
    
    last_update = get_last_update_date("sessions")
    print(f"\n📅 Dernière mise à jour détectée : {last_update.strftime('%Y-%m-%d')}")
    print(f"📥 Récupération des données depuis : {last_update.strftime('%Y-%m-%d')}")
    
    df = fetch_charge_data(start_date=last_update)
    print(f"✅ {len(df)} nouvelles charges récupérées")
    
    if df.empty:
        print("ℹ️  Aucune nouvelle donnée à traiter. Arrêt du processus.")
        return
    
    print(f"\n🗑️  Suppression des données existantes depuis {last_update.strftime('%Y-%m-%d')}...")
    for table in TABLES_TO_SAVE:
        delete_old_data(table, last_update)
    
    print("\n🔄 Traitement des données...")
    resolve_session_id(df)
    df = classify_errors(df)
    
    if "Site" not in df.columns and "Name Project" in df.columns:
        df["Site"] = df["Name Project"]
    df["Site"] = df["Site"].astype(str).str.strip().replace({"": "Unknown"})
    df["moment_avancee"] = df["moment"].map(
        {
            "Init": "Avant charge",
            "Lock Connector": "Avant charge",
            "CableCheck": "Avant charge",
            "Charge": "Charge",
            "Fin de charge": "Fin de charge",
        }
    ).fillna("Unknown")

    print("📊 Construction des tables KPI...")
    evi = build_evi_combo_tables(df)

    mac_lookup = pd.read_sql("SELECT * FROM Charges.mac_lookup", con=engine_kpi)
    mac_lookup.columns = [c.lower().strip() for c in mac_lookup.columns]
    mac_lookup = mac_lookup.rename(columns={"mac address": "mac", "vehicle": "Vehicle"})

    charges_mac = build_charges_mac(df, mac_lookup)
    durations = build_durations_daily(df)
    charges_daily_by_site = build_charges_daily_by_site(df)
    multi_attempts = build_multi_attempts_hour(df)
    suspicious_under_1kwh = build_suspicious_under_1kwh(df)

    if not charges_mac.empty and not multi_attempts.empty:
        cm_min = charges_mac[["ID", "MAC Address", "Vehicle"]].drop_duplicates("ID", keep="last")
        multi_attempts = multi_attempts.merge(
            cm_min, left_on="ID_ref", right_on="ID", how="left"
        ).drop(columns="ID", errors="ignore")
        multi_attempts["MAC Address"] = multi_attempts["MAC Address_y"].fillna("").map(_fmt_mac)
        multi_attempts["Vehicle"] = multi_attempts["Vehicle"].fillna("Unknown")
        multi_attempts["MAC"] = multi_attempts["MAC Address"]
        multi_attempts = multi_attempts.drop(
            columns=["MAC Address_x", "MAC Address_y"], errors="ignore"
        )
    else:
        multi_attempts["MAC Address"] = ""
        multi_attempts["Vehicle"] = "Unknown"
        multi_attempts["MAC"] = ""

    if not charges_mac.empty and not suspicious_under_1kwh.empty:
        cm_min = charges_mac[["ID", "MAC Address", "Vehicle"]].drop_duplicates("ID", keep="last")
        suspicious_under_1kwh = suspicious_under_1kwh.merge(cm_min, on="ID", how="left")
        suspicious_under_1kwh["MAC Address"] = (
            suspicious_under_1kwh["MAC Address"].fillna("").map(_fmt_mac)
        )
        suspicious_under_1kwh["Vehicle"] = suspicious_under_1kwh["Vehicle"].fillna("Unknown")
    else:
        suspicious_under_1kwh["MAC Address"] = ""
        suspicious_under_1kwh["Vehicle"] = "Unknown"

    sessions_cols = [
        "Site",
        "Name Project",
        "PDC",
        "Datetime start",
        "Datetime end",
        "State of charge(0:good, 1:error)",
        "ID",
        "is_ok",
        "type_erreur",
        "moment",
        "moment_avancee",
        "Energy (Kwh)",
        "Mean Power (Kw)",
        "Max Power (Kw)",
        "SOC Start",
        "SOC End",
        "EVI Error Code",
        "Downstream Code PC",
        "EVI Status during error",
        "MAC Address",
        "charge_900V",
    ]
    sessions_cols = [c for c in sessions_cols if c in df.columns]
    sessions = df[sessions_cols].copy()

    all_tables = {
        "evi_combo_long": evi["evi_combo_long"],
        "evi_combo_by_site": evi["evi_combo_by_site"],
        "evi_combo_by_site_pdc": evi["evi_combo_by_site_pdc"],
        "charges_mac": charges_mac,
        "multi_attempts_hour": multi_attempts,
        "suspicious_under_1kwh": suspicious_under_1kwh,
        "durations_site_daily": durations["durations_site_daily"],
        "durations_pdc_daily": durations["durations_pdc_daily"],
        "charges_daily_by_site": charges_daily_by_site,
        "sessions": sessions,
    }
    
    print("\n💾 Sauvegarde dans la base de données...")
    save_to_indicator(all_tables, incremental=True)
    
    print("\n" + "=" * 80)
    print("✅ PROCESSUS TERMINÉ AVEC SUCCÈS")
    print("=" * 80)


if __name__ == "__main__":
    main()
