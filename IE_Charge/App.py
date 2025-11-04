#Dashboard.py
import uuid
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine
import plotly.graph_objects as go
import datetime
import calendar
from tabs.context import get_context
from tabs import (
    tab1_general,
    tab2_comparaison,
    tab3_details_pdc,
    tab4_stats,
    tab5_projection,
    tab6_tentatives,
    tab7_suspectes,
    tab8_erreur_moment,
    tab9_erreur_specifique,
    tab10_alertes,
    tab11_evolution,
)

# COULEURS 
COLORS_PDC = ["#FF7F0E", "#FF0000", "#2CA02C", "#D62728", "#9467BD", "#8C564B"]
MOMENT_ORDER = ["Init", "Lock Connector", "CableCheck", "Charge", "Fin de charge", "Unknown"]
MOMENT_PALETTE = {
    "Init": "#636EFA",
    "Lock Connector": "#EF553B",
    "CableCheck": "#00CC96",
    "Charge": "#AB63FA",
    "Fin de charge": "#38AC21",
    "Unknown": "#19D3F3",
}
SITE_PALETTE = {
    "Saint-Jean-de-Maurienne": "#636EFA",  
    "La Rochelle": "#EF553B",             
    "Pouilly-en-Auxois": "#00CC96",        
    "Carvin": "#AB63FA",                   
    "Pau - Novotel": "#38AC21",           
    "Unknown": "#19D3F3",                  
}
BASE_CHARGE_URL = "https://elto.nidec-asi-online.com/Charge/detail?id="

# PAGE / HEADER 
st.set_page_config(
    page_title="Dashboard Erreurs de Charge",
    page_icon="assets/elto.png",
    layout="wide",
)
context = get_context()
import base64
def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

elto_logo  = encode_image("assets/elto.png")
main_logo  = encode_image("assets/Logo.png")
nidec_logo = encode_image("assets/nidec.png")

# Layout en colonnes : Elto | Logo principal (large) | Nidec
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    st.markdown(
        f"<img src='data:image/png;base64,{elto_logo}' style='height:80px;'>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"<div style='text-align:center; margin: 10px 0;'><img src='data:image/png;base64,{main_logo}' style='width:20%; max-width:800px;'></div>",
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"<img src='data:image/png;base64,{nidec_logo}' style='height:80px;'>",
        unsafe_allow_html=True
    )

st.markdown("---")

# HELPERS
def _get_list_safe(pivot, colname, default=0):
    if colname in pivot:
        col = pivot[colname]
        if isinstance(col, pd.Series):
            return col.tolist()
    return [default] * len(pivot)

def gen_key(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4()}"

def plot(fig, key_prefix: str):
    st.plotly_chart(fig, use_container_width=True, key=gen_key(key_prefix))

@st.cache_data(show_spinner=False)
def load_kpis_from_sql():
    engine = create_engine("mysql+pymysql://nidec:MaV38f5xsGQp83@162.19.251.55:3306/Charges")
    query = """
        SELECT TABLE_NAME
        FROM information_schema.tables
        WHERE table_schema = 'Charges' AND table_name LIKE 'kpi_%%'
    """
    df_tables = pd.read_sql(query, con=engine)
    table_names = df_tables["TABLE_NAME"].tolist()  
    kpis = {}
    for name in table_names:
        df = pd.read_sql(f"SELECT * FROM Charges.{name}", con=engine)
        kpis[name.replace("kpi_", "")] = df
    return kpis

def evi_counts_pivot(df):
    tmp = df.copy()
    tmp["EVI_Code"] = pd.to_numeric(tmp["EVI_Code"], errors="coerce").fillna(0).astype(int)
    tmp["EVI_Step"] = pd.to_numeric(tmp["EVI_Step"], errors="coerce").fillna(0).astype(int)

    tmp = tmp[(tmp["EVI_Code"] != 0) | (tmp["EVI_Step"] != 0)]

    # Groupby et pivot 
    gb = tmp.groupby(["Site", "EVI_Step", "EVI_Code"], as_index=False).size()
    pv = gb.pivot_table(
        index="Site",
        columns=["EVI_Step", "EVI_Code"], 
        values="size",
        aggfunc="sum",
        fill_value=0
    ).sort_index(axis=1, level=[0, 1])

    # Ajout des totaux
    pv["Total"] = pv.sum(axis=1)
    total_row = pv.sum(axis=0)
    total_row.name = "Total"
    pv = pd.concat([pv, total_row.to_frame().T])

    return pv

def hide_zero_labels(fig):
    import numpy as np
    for tr in fig.data:
        vals = np.array(tr.y if getattr(tr, "orientation", "v") != "h" else tr.x, dtype=float)
        txt = []
        for v in vals:
            if np.isnan(v) or v == 0:
                txt.append("")
            else:
                txt.append(str(int(v)) if float(v).is_integer() else f"{v:g}")
        tr.text = txt
        tr.texttemplate = "%{text}"
        tr.textposition = "outside"
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
    fig.update_traces(cliponaxis=False)
    return fig

def _fmt_mac(s: str) -> str:
    if not s: return ""
    if len(s) % 2: s = "0" + s
    pairs = [s[i:i+2] for i in range(0, min(12, len(s)), 2)]
    return ":".join(pairs).upper()

def with_charge_link(df: pd.DataFrame, id_col: str = "ID", link_col: str = "Lien") -> pd.DataFrame:
    if id_col not in df.columns:
        return df
    out = df.copy()
    out[id_col] = out[id_col].astype(str).str.strip()
    out[link_col] = BASE_CHARGE_URL + out[id_col]
    return out

# CHARGEMENT KPI DEPUIS SQL
tables = load_kpis_from_sql()

evi_long      = tables.get("evi_combo_long", pd.DataFrame())
evi_by_site   = tables.get("evi_combo_by_site", pd.DataFrame())
evi_by_site_p = tables.get("evi_combo_by_site_pdc", pd.DataFrame())

sessions = tables.get("sessions", pd.DataFrame())

if sessions.empty:
    st.error("Aucune donnée dans `sessions` — lancer la mise à jour.")
    st.stop()

# FILTRES ROBUSTES
SITE_COL = "Site" if "Site" in sessions.columns else "Name Project"
sites = sorted(sessions[SITE_COL].dropna().unique().tolist())

# État initial
if "site_sel" not in st.session_state:
    st.session_state.site_sel = sites[:] 

# Flag pour limiter les sites
if "limit_sites_to_20" not in st.session_state:
    st.session_state.limit_sites_to_20 = False

if st.session_state.limit_sites_to_20:
    top_sites = []
    if not sessions.empty and SITE_COL in sessions.columns:
        counts = (
            sessions[SITE_COL]
            .dropna()
            .value_counts()
        )
        top_sites = [site for site in counts.index if site in sites][:20]

    if not top_sites and len(st.session_state.site_sel) > 20:
        top_sites = st.session_state.site_sel[:20]

    if top_sites:
        st.session_state.site_sel = top_sites
    st.session_state.limit_sites_to_20 = False

# Préconversion des dates 
dt_start = pd.to_datetime(sessions["Datetime start"], errors="coerce")

# Initialisation des paramètres de date
today = datetime.date.today()
if "date_mode" not in st.session_state:
    st.session_state.date_mode = "mois_complet"
if "focus_year" not in st.session_state:
    st.session_state.focus_year = today.year
if "focus_month" not in st.session_state:
    st.session_state.focus_month = today.month
if "focus_day" not in st.session_state:
    st.session_state.focus_day = today

TYPE_MEMORY_KEY = "__type_sel_memory__"
MOMENT_MEMORY_KEY = "__moment_sel_memory__"
LAST_MOMENT_OPTIONS_KEY = "__moment_options_snapshot__"

# ========== FILTRES ==========
st.markdown("### 🎯 Filtres")

# Ligne 1: Sites
c1, c2 = st.columns([1, 5], gap="small")

with c1:
    if st.button("✅ Tous les sites", key="btn_all_sites", use_container_width=True):
        st.session_state.site_sel = sites[:]   
        st.rerun()
if "site_sel" not in st.session_state:
    st.session_state.site_sel = sites[:]
with c2:
    st.multiselect(
        "Sites",
        options=sites,
        key="site_sel",   
        label_visibility="collapsed",
        help="Choisissez un ou plusieurs sites",
    )


# Ligne 2: Mode de période (4 boutons)
st.markdown("#### 📅 Période d'analyse")
col_mode_day, col_mode_full, col_mode_j1, col_mode_week, col_mode_all = st.columns(5)

with col_mode_day:
    if st.button("📅 Focus Jour", key="btn_focus_jour", use_container_width=True, type="primary" if st.session_state.date_mode == "focus_jour" else "secondary"):
        st.session_state.date_mode = "focus_jour"
        st.rerun()

with col_mode_full:
    if st.button("📅 Focus Mois", key="btn_mois_complet", use_container_width=True, type="primary" if st.session_state.date_mode == "mois_complet" else "secondary"):
        st.session_state.date_mode = "mois_complet"
        st.rerun()

with col_mode_j1:
    if st.button("📅 J-1 (Hier)", key="btn_j_minus_1", use_container_width=True, type="primary" if st.session_state.date_mode == "j_minus_1" else "secondary"):
        st.session_state.date_mode = "j_minus_1"
        st.rerun()

with col_mode_week:
    if st.button("📅 Semaine -1", key="btn_semaine_minus_1", use_container_width=True, type="primary" if st.session_state.date_mode == "semaine_minus_1" else "secondary"):
        st.session_state.date_mode = "semaine_minus_1"
        st.rerun()

with col_mode_all:
    if st.button("📅 Toute la période", key="btn_all_period", use_container_width=True, type="primary" if st.session_state.date_mode == "toute_periode" else "secondary"):
        st.session_state.date_mode = "toute_periode"
        # Activer le flag pour limiter à 20 sites au prochain rerun
        st.session_state.limit_sites_to_20 = True
        st.rerun()

# Ligne 3: Sélection du mois
if st.session_state.date_mode == "focus_jour":
    prev_day = st.session_state.get("focus_day", today)
    st.session_state.focus_day = st.date_input(
        "📅 Sélectionner une date",
        value=prev_day,
        min_value=today - datetime.timedelta(days=365 * 5),
        max_value=today,
        key="focus_day_input"
    )

if st.session_state.date_mode == "mois_complet":
    prev_year = st.session_state.focus_year
    prev_month = st.session_state.focus_month
    col_year, col_month = st.columns([1, 3])

    with col_year:
        report_year = st.selectbox(
            "Année",
            options=range(today.year, today.year - 5, -1),
            index=0,
            key="focus_year"
        )

    with col_month:
        month_abbr = list(calendar.month_abbr[1:])
        current_month_idx = st.session_state.focus_month - 1

        report_month_str = st.radio(
            "Mois",
            options=month_abbr,
            index=current_month_idx,
            horizontal=True,
            key="focus_month_radio"
        )

        report_month = month_abbr.index(report_month_str) + 1
        st.session_state.focus_month = report_month

    if prev_year != st.session_state.focus_year or prev_month != st.session_state.focus_month:
        st.rerun()

# ========== CALCUL DES DATES SELON LE MODE ==========
date_mode = st.session_state.date_mode

# Aujourd'hui
yesterday = today - datetime.timedelta(days=1)

if date_mode == "j_minus_1":
    # J-1 = hier uniquement
    d1 = yesterday
    d2 = yesterday
    mode_label = f"📅 J-1 (hier) : {yesterday.strftime('%d/%m/%Y')}"

elif date_mode == "semaine_minus_1":
    # Semaine -1 = les 7 derniers jours jusqu'à hier
    d1 = yesterday - datetime.timedelta(days=6)
    d2 = yesterday
    mode_label = f"📅 Semaine -1 : du {d1.strftime('%d/%m/%Y')} au {d2.strftime('%d/%m/%Y')}"

elif date_mode == "toute_periode":
    # Toute la période = du min au max des données disponibles
    min_dt = dt_start.min()
    max_dt = dt_start.max()
    if pd.notna(min_dt) and pd.notna(max_dt):
        d1 = min_dt.date()
        d2 = max_dt.date()
        mode_label = f"📅 Toute la période : du {d1.strftime('%d/%m/%Y')} au {d2.strftime('%d/%m/%Y')}"
    else:
        # Fallback si pas de données
        d1 = today
        d2 = today
        mode_label = "📅 Toute la période : Aucune donnée disponible"
elif date_mode == "focus_jour":
    focus_day = st.session_state.get("focus_day", today)
    d1 = focus_day
    d2 = focus_day
    mode_label = f"📅 Focus Jour : {focus_day.strftime('%d/%m/%Y')}"

else:  # mois_complet
    year = st.session_state.focus_year
    month = st.session_state.focus_month
    _, last_day = calendar.monthrange(year, month)
    d1 = datetime.date(year, month, 1)
    d2 = datetime.date(year, month, last_day)
    mode_label = f"📅 Mois complet : {calendar.month_name[month]} {year}"

# Affichage de la période active avec avertissement si limite de sites atteinte
if date_mode == "toute_periode" and len(st.session_state.site_sel) == 20:
    st.info(mode_label)
    st.warning("⚠️ Par défaut, seuls les 20 sites avec l’activité la plus élevée sont pré-sélectionnés pour optimiser les performances. Vous pouvez étendre la sélection pour inclure l’ensemble des sites si nécessaire.")
else:
    st.info(mode_label)

# ========== FILTRAGE DES DONNÉES ==========
d1_ts = pd.Timestamp(d1)
d2_ts = pd.Timestamp(d2) + pd.Timedelta(days=1) 

site_mask = sessions[SITE_COL].isin(st.session_state.site_sel)
mask = site_mask & dt_start.ge(d1_ts) & dt_start.lt(d2_ts)
sess = sessions.loc[mask].copy()

# is_ok
if "State of charge(0:good, 1:error)" in sess.columns:
    soc = pd.to_numeric(sess["State of charge(0:good, 1:error)"], errors="coerce").fillna(0).astype(int)
    sess["is_ok"] = soc.eq(0)
else:
    sess["is_ok"] = False

# RÉSUMÉ 
nb_sites = len(st.session_state.site_sel)
nb_pdc_tot = sess["PDC"].nunique() if "PDC" in sess.columns else 0

st.caption(
    f"**Période**: {d1} → {d2} · "
    f"**Sites**: {nb_sites} · "
    f"**PDC**: {nb_pdc_tot}"
)

if sess.empty:
    st.warning("Aucune donnée pour ces filtres. Essayez de modifier la période ou les sites sélectionnés.")

# ========== FILTRES SECONDAIRES (Type/Moment d'erreur) ==========
st.markdown("---")
row_type, row_moment, row_avant = st.columns([1, 1, 0.7])

type_options, moment_options = [], []
if "type_erreur" in sess.columns:
    type_options = sorted(sess["type_erreur"].dropna().unique().tolist())

if "moment" in sess.columns:
    opts = sorted(sess["moment"].dropna().unique().tolist())
    moment_options = [m for m in MOMENT_ORDER if m in opts] + [m for m in opts if m not in MOMENT_ORDER]


def _sync_multiselect_state(key: str, options: list[str], memory_key: str) -> None:
    """Ensure a multiselect keeps previous choices even if options vary."""

    last_options_key = f"{memory_key}__last_options__"
    select_all_key = f"{memory_key}__select_all__"
    full_memory_key = f"{memory_key}__full_memory__"

    if not options:
        st.session_state.setdefault(memory_key, [])
        st.session_state[key] = []
        st.session_state[last_options_key] = []
        st.session_state[select_all_key] = False
        st.session_state.setdefault(full_memory_key, st.session_state.get(memory_key, [])[:])
        return

    previous_options = st.session_state.get(last_options_key, options[:])

    if memory_key not in st.session_state:
        st.session_state[memory_key] = options[:]

    if select_all_key not in st.session_state:
        st.session_state[select_all_key] = True

    full_memory = st.session_state.get(full_memory_key)
    if full_memory is None:
        default_pref = st.session_state.get(memory_key, options[:])
        full_memory = default_pref[:]
    else:
        full_memory = list(full_memory)

    memory = st.session_state[memory_key]

    if key in st.session_state:
        current = st.session_state[key]
    else:
        current = memory

    filtered_current = [val for val in current if val in options]
    filtered_memory = [val for val in full_memory if val in options]

    if not filtered_memory and options:
        filtered_memory = options[:]

    select_all_prev = st.session_state.get(select_all_key, False)
    previous_set = set(previous_options)
    options_set = set(options)

    if select_all_prev and options_set.issuperset(previous_set):
        new_selection = options[:]
    elif key in st.session_state and not st.session_state[key]:
        new_selection = []
    elif filtered_current:
        new_selection = filtered_current[:]
    elif filtered_memory:
        new_selection = filtered_memory[:]
    else:
        new_selection = options[:]

    st.session_state[key] = new_selection[:]
    st.session_state[memory_key] = new_selection[:]
    st.session_state[last_options_key] = options[:]
    st.session_state[select_all_key] = len(new_selection) == len(options) and bool(options)
    st.session_state[f"{memory_key}__options__"] = options[:]

    unavailable_preserved = [val for val in full_memory if val not in options]
    new_full_memory = unavailable_preserved[:]
    for val in new_selection:
        if val not in new_full_memory:
            new_full_memory.append(val)

    st.session_state[full_memory_key] = new_full_memory


def _make_memory_updater(key: str, memory_key: str):
    def _update():
        selection = st.session_state.get(key, [])[:]
        st.session_state[memory_key] = selection
        options = st.session_state.get(f"{memory_key}__last_options__", [])
        st.session_state[f"{memory_key}__select_all__"] = len(selection) == len(options) and bool(options)
        full_memory_key = f"{memory_key}__full_memory__"
        prev_full_memory = list(st.session_state.get(full_memory_key, []))
        unavailable_preserved = [val for val in prev_full_memory if val not in options]
        new_full_memory = unavailable_preserved[:]
        for val in selection:
            if val not in new_full_memory:
                new_full_memory.append(val)

        st.session_state[full_memory_key] = new_full_memory

    return _update


_sync_multiselect_state("type_sel", type_options, TYPE_MEMORY_KEY)
_sync_multiselect_state("moment_sel", moment_options, MOMENT_MEMORY_KEY)

if moment_options:
    st.session_state["__moment_order__"] = moment_options[:]

_update_type_memory = _make_memory_updater("type_sel", TYPE_MEMORY_KEY)
_update_moment_memory = _make_memory_updater("moment_sel", MOMENT_MEMORY_KEY)

GROUPS = {
    "avant_charge_toggle": {"Init", "Lock Connector", "CableCheck"},
    "charge_toggle": {"Charge"},
    "fin_charge_toggle": {"Fin de charge"},
}


def _sync_toggle_state(options: list[str]) -> None:
    """Aligne l'état des toggles sur les valeurs réellement disponibles."""

    previous_options = st.session_state.get(LAST_MOMENT_OPTIONS_KEY)
    if previous_options != options:
        current_moments = set(st.session_state.get("moment_sel", []))
        for toggle_key, group in GROUPS.items():
            st.session_state[toggle_key] = group.issubset(current_moments)
    st.session_state[LAST_MOMENT_OPTIONS_KEY] = options[:]


_sync_toggle_state(moment_options)

def _on_toggle(key):
    group = GROUPS[key]
    active = st.session_state.get(key, False)
    cur = set(st.session_state.get("moment_sel", []))
    if active:
        cur |= group
    else:
        cur -= group
    order = st.session_state.get("__moment_order__", list(cur))
    st.session_state["moment_sel"] = [m for m in order if m in cur]
    _update_moment_memory()

with row_type:
    st.multiselect(
        "Type d'erreur (global)",
        options=type_options,
        key="type_sel",
        on_change=_update_type_memory,
        help="Filtre global sur le type d'erreur (ex: Erreur_EVI, Erreur_DownStream)."
    )

with row_moment:
    st.multiselect(
        "Moment d'erreur",
        options=moment_options,
        key="moment_sel",
        on_change=_update_moment_memory,
        help="S'applique seulement aux erreurs EVI et DS"
    )

with row_avant:
    st.toggle(
        "⚡ Avant charge",
        value=GROUPS["avant_charge_toggle"].issubset(set(st.session_state.get("moment_sel", []))),
        key="avant_charge_toggle",
        help="Ajoute/retire Init, Lock Connector et CableCheck",
        on_change=lambda: _on_toggle("avant_charge_toggle"),
    )

    st.toggle(
        "🔋 Charge",
        value=GROUPS["charge_toggle"].issubset(set(st.session_state.get("moment_sel", []))),
        key="charge_toggle",
        help="Ajoute/retire Charge",
        on_change=lambda: _on_toggle("charge_toggle"),
    )

    st.toggle(
        "⚡ Fin de charge",
        value=GROUPS["fin_charge_toggle"].issubset(set(st.session_state.get("moment_sel", []))),
        key="fin_charge_toggle",
        help="Ajoute/retire Fin de charge",
        on_change=lambda: _on_toggle("fin_charge_toggle"),
    )

# APPLICATION DES FILTRES SUR LES KPI GLOBAUX 
mask_nok   = ~sess["is_ok"]
mask_type  = True
mask_moment = True

if "type_erreur" in sess.columns and st.session_state.type_sel:
    mask_type = sess["type_erreur"].isin(st.session_state.type_sel)

if {"type_erreur", "moment"}.issubset(sess.columns) and st.session_state.moment_sel:
    mask_moment = sess["moment"].isin(st.session_state.moment_sel)

mask_nok_keep = mask_nok & mask_type & mask_moment
sess_kpi = sess.copy()
sess_kpi["is_ok_filt"] = np.where(mask_nok_keep, False, True)

total = len(sess_kpi)
ok    = int(sess_kpi["is_ok_filt"].sum())
nok   = total - ok
taux_reussite = round(ok / total * 100, 2) if total else 0.0
taux_echec    = round(nok / total * 100, 2) if total else 0.0

# Préparation des variables pour le contexte
site_sel = st.session_state.site_sel

# Mise à jour du session state pour compatibilité avec les tabs
st.session_state.d1 = d1
st.session_state.d2 = d2

context.__dict__.clear()
context.__dict__.update({k: v for k, v in locals().items() if k != "context"})

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "📋 Générale",
    "🏢 Comparaison par site (Activité)",
    "🔌 Détails Site (par PDC)",
    "📈 Statistiques",
    "📑 Projection pivot",
    "⚠️ Analyse tentatives multiples",
    "⚠️ Transactions suspectes",
    "🔍 Analyse Erreur Moment",
    "🔍 Analyse Erreur Spécifique",
    "⚠️ Alertes",
    "📈 Evolution",
])

stats_all = tables.get("stats_global_all", pd.DataFrame())
stats_ok  = tables.get("stats_global_ok",  pd.DataFrame())
context.__dict__.update({"stats_all": stats_all, "stats_ok": stats_ok})

# Tab 1 
with tab1:
    tab1_general.render()
with tab2:
    tab2_comparaison.render()
with tab3:
    tab3_details_pdc.render()
with tab4:
    tab4_stats.render()
with tab5:
    tab5_projection.render()
with tab6:
    tab6_tentatives.render()
with tab7:
    tab7_suspectes.render()
with tab8:
    tab8_erreur_moment.render()
with tab9:
    tab9_erreur_specifique.render()
with tab10:
    tab10_alertes.render()
with tab11:
    tab11_evolution.render()
