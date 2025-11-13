from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine


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

DB_CONFIG_CHARGE = {
    "host": "162.19.251.55",
    "port": 3306,
    "user": "nidec",
    "password": "MaV38f5xsGQp83",
    "database": "nw_borne",
}

def env(name: str, default: str) -> str:
    """Return the value of an environment variable with a default fallback."""

    return os.environ.get(name, default)


INFLUX_HOST = env("INFLUX_HOST", "tsdbe.nidec-asi-online.com")
INFLUX_PORT = env("INFLUX_PORT", "443")
INFLUX_USER = env("INFLUX_USER", "nw")
INFLUX_PW = env("INFLUX_PW", "at3Dd94Yp8BT4Sh!")
INFLUX_DB = env("INFLUX_DB", "signals")
INFLUX_MEAS = env("INFLUX_MEAS", "fastcharge")
INFLUX_TAG_PROJECT = env("INFLUX_TAG_PROJECT", "project")


SIGNAL_MAP = {
    1: "EVI_P1.ILI.EVSE_OutVoltage",
    2: "EVI_P2.ILI.EVSE_OutVoltage",
    3: "EVI_P3.ILI.EVSE_OutVoltage",
    4: "EVI_P4.ILI.EVSE_OutVoltage",
}


ERROR_CODE = 84
ERROR_STEP = 7
NUM_WORKERS = 10  # Number of parallel workers for processing charges

# Heuristics used to qualify the signal behaviour.
FLAT_ABS_TOLERANCE = 1e-6
PEAK_THRESHOLD = 100.0
BETWEEN_LOW = 30.0
BETWEEN_HIGH = 70.0


class InfluxClient:
    """Minimal HTTP client for InfluxDB 1.x queries."""

    def __init__(self) -> None:
        self._base_url = f"https://{INFLUX_HOST}:{INFLUX_PORT}/query"
        self._auth = (INFLUX_USER, INFLUX_PW)

    def query(self, query: str) -> pd.DataFrame:
        params = {"db": INFLUX_DB, "q": query, "epoch": "s"}
        response = requests.get(self._base_url, params=params, auth=self._auth, timeout=30)
        response.raise_for_status()
        payload = response.json()

        results = payload.get("results", [])
        if not results:
            return pd.DataFrame(columns=["time", "value"])

        series = results[0].get("series")
        if not series:
            return pd.DataFrame(columns=["time", "value"])

        values = series[0].get("values", [])
        if not values:
            return pd.DataFrame(columns=["time", "value"])

        return pd.DataFrame(values, columns=series[0].get("columns", ["time", "value"]))


def _build_engine(config: dict):
    """Build a SQLAlchemy engine from the existing configuration."""

    return create_engine(
        "mysql+pymysql://{user}:{password}@{host}:{port}/{database}".format(**config)
    )


def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iter_project_candidates(row: pd.Series) -> Iterator[str]:
    site = str(row.get("Site", "")).strip()
    if site:
        yield site

    name_project = str(row.get("Name Project", "")).strip()
    if name_project and name_project != site:
        yield name_project

    project_id = row.get("Id Project")
    if pd.notna(project_id):
        project_str = str(project_id).strip()
        if project_str:
            yield project_str
            yield project_str.zfill(3)
            mapped = SITE_MAP.get(project_str)
            if mapped:
                yield mapped


def fetch_charges(engine, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame:
    """Retrieve all charges matching the target error."""

    start_clause = f" AND start_time >= '{start.strftime('%Y-%m-%d')}'" if start else ""
    end_clause = f" AND start_time < '{(end + timedelta(days=1)).strftime('%Y-%m-%d')}'" if end else ""

    query = f"""
        SELECT *
        FROM charge_info
        WHERE Tri_charge = 1
          AND Evi_error_code = {ERROR_CODE}
          AND EVi_status_at_error = {ERROR_STEP}
          {start_clause}
          {end_clause}
        ORDER BY start_time ASC
    """

    df = pd.read_sql(query, con=engine)

    rename_map = {
        "start_time": "Datetime start",
        "end_time": "Datetime end",
        "borne_id": "PDC",
        "project_name": "Name Project",
        "project_num": "Id Project",
    }

    df = df.rename(columns=rename_map)

    for col in ("Datetime start", "Datetime end"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["PDC"] = pd.to_numeric(df.get("PDC"), errors="coerce").astype("Int64")

    if "Site" not in df.columns:
        df["Site"] = df.get("Name Project", "")

    missing_site = df["Site"].astype(str).str.strip().eq("")
    if missing_site.any():
        candidates = df.loc[missing_site, "Id Project"].astype(str).str.strip()
        df.loc[missing_site, "Site"] = candidates.map(SITE_MAP).fillna(candidates)

    return df


def describe_signal(values: pd.Series) -> str:
    if values.empty:
        return "Aucune donnée Influx"

    arr = values.astype(float).to_numpy()
    if np.all(np.abs(arr) <= FLAT_ABS_TOLERANCE):
        return "Lecture EVI"

    segments: List[tuple[int, int]] = []
    current: Optional[List[int]] = None
    for idx, val in enumerate(arr):
        if val >= PEAK_THRESHOLD:
            if current is None:
                current = [idx, idx]
            else:
                current[1] = idx
        else:
            if current is not None:
                segments.append((current[0], current[1]))
                current = None

    if current is not None:
        segments.append((current[0], current[1]))

    for first, second in zip(segments, segments[1:]):
        between = arr[first[1] + 1 : second[0]]
        if between.size == 0:
            continue
        if np.any((between >= BETWEEN_LOW) & (between <= BETWEEN_HIGH)):
            return "Réglage Variateur"

    return "Profil indéterminé"


def load_signal(
    client: InfluxClient,
    field: str,
    start: datetime,
    end: datetime,
    project_candidates: Iterable[str],
) -> pd.DataFrame:
    utc_start = ensure_utc(start)
    utc_end = ensure_utc(end)
    base_conditions = f"time >= '{utc_start.isoformat()}' AND time <= '{utc_end.isoformat()}'"

    for candidate in project_candidates:
        # Escape single quotes in the candidate name for InfluxDB
        escaped_candidate = candidate.replace("'", "\\'")
        tag_condition = f'"{INFLUX_TAG_PROJECT}" = \'{escaped_candidate}\''
        query = f'SELECT "{field}" FROM "{INFLUX_MEAS}" WHERE {tag_condition} AND {base_conditions}'
        result = client.query(query)
        if not result.empty:
            return result

    return pd.DataFrame(columns=["time", field])


def build_report(df: pd.DataFrame, output: Path) -> Path:
    client = InfluxClient()

    records = []

    for _, row in df.iterrows():
        pdc = row.get("PDC")
        if pd.isna(pdc) or int(pdc) not in SIGNAL_MAP:
            continue

        field = SIGNAL_MAP[int(pdc)]
        project_candidates = list(dict.fromkeys(iter_project_candidates(row)))
        signal_df = load_signal(client, field, row["Datetime start"], row["Datetime end"], project_candidates)

        if signal_df.empty:
            comment = "Aucune donnée Influx"
            min_val = np.nan
            max_val = np.nan
        else:
            value_col = field if field in signal_df.columns else signal_df.columns[-1]
            series = pd.to_numeric(signal_df[value_col], errors="coerce").dropna()
            min_val = series.min() if not series.empty else np.nan
            max_val = series.max() if not series.empty else np.nan
            comment = describe_signal(series)

        records.append(
            {
                "Site": row.get("Site") or row.get("Name Project"),
                "Name Project": row.get("Name Project"),
                "Id Project": row.get("Id Project"),
                "PDC": int(pdc),
                "Datetime start": row.get("Datetime start"),
                "Datetime end": row.get("Datetime end"),
                "Signal": field,
                "Min Voltage": min_val,
                "Max Voltage": max_val,
                "Commentaire": comment,
            }
        )

    report_df = pd.DataFrame(records)
    report_df.sort_values(["Site", "Datetime start", "PDC"], inplace=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_excel(output, index=False)
    return output


def main(argv: Optional[Sequence[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description="Analyse des tensions EVI pour l'erreur 84 / moment 7")
    parser.add_argument("--start", type=parse_date, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", type=parse_date, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports/evi_voltage_report.xlsx"),
        help="Chemin du fichier Excel de sortie",
    )

    args = parser.parse_args(argv)

    engine = _build_engine(DB_CONFIG_CHARGE)
    charges = fetch_charges(engine, args.start, args.end)

    if charges.empty:
        print("Aucune charge avec l'erreur EVI 84 / step 7 trouvée dans l'intervalle spécifié.")
        return args.output

    output_path = build_report(charges, args.output)
    print(f"Rapport généré : {output_path}")
    return output_path


if __name__ == "__main__":
    main()
