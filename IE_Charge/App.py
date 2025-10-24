#Dashboard.py
import uuid
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine
import plotly.graph_objects as go

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

# état initial
if "site_sel" not in st.session_state:
    st.session_state.site_sel = sites[:] 
if "d1" not in st.session_state:
    st.session_state.d1 = None
if "d2" not in st.session_state:
    st.session_state.d2 = None

# Préconversion des dates 
dt_start = pd.to_datetime(sessions["Datetime start"], errors="coerce")

# BARRE DE FILTRES 
c1, c2, c3, c4 = st.columns([2, 5, 2, 5], gap="small")

with c1:
    if st.button("✅ Tous les sites", key="btn_all_sites", use_container_width=True):
        st.session_state.site_sel = sites[:]   
        st.rerun()
st.session_state.setdefault("site_sel", sites[:])
with c2:
    st.multiselect(
        "Sites",
        options=sites,
        key="site_sel",   
        label_visibility="collapsed",
        help="Choisissez un ou plusieurs sites",
    )

if len(st.session_state.site_sel) == 0:
    st.session_state.site_sel = sites[:]

# Bornes dates
site_mask = sessions[SITE_COL].isin(st.session_state.site_sel)
dt_site = dt_start[site_mask]
min_dt = dt_site.min()
max_dt = dt_site.max()
if pd.isna(min_dt) or pd.isna(max_dt):
    st.error("Aucune date disponible pour les sites sélectionnés.")
    st.stop()

min_date = min_dt.date()
max_date = max_dt.date()

# clamp utilitaire
def _clamp(d):
    if d is None: return None
    if d < min_date: return min_date
    if d > max_date: return max_date
    return d

# init/clamp d1/d2
d1 = _clamp(st.session_state.d1) or min_date
d2 = _clamp(st.session_state.d2) or max_date
if d1 > d2:
    d1, d2 = d2, d1
    st.session_state.d1, st.session_state.d2 = d1, d2

# Initialise le compteur si besoin
if "widget_key" not in st.session_state:
    st.session_state.widget_key = 0

with c3:
    col_btn_all, col_btn_jmoins1 = st.columns(2)
    col_btn_week, col_btn_month = st.columns(2)

    if col_btn_week.button("📅 Semaine -1", key="btn_week_minus1", use_container_width=True):
        d2 = st.session_state.get("d2", max_date)
        new_d1 = max(min_date, d2 - pd.Timedelta(days=7))
        st.session_state.d1 = new_d1
        st.session_state.d2 = d2
        st.session_state.widget_key += 1
        st.rerun()

    if col_btn_month.button("📅 Mois -1", key="btn_month_minus1", use_container_width=True):
        d2 = st.session_state.get("d2", max_date)
        try:
            new_d1 = (pd.Timestamp(d2) - pd.DateOffset(months=1)).date()
        except Exception:
            new_d1 = max(min_date, d2 - pd.Timedelta(days=30))
        new_d1 = max(min_date, new_d1)
        st.session_state.d1 = new_d1
        st.session_state.d2 = d2
        st.session_state.widget_key += 1
        st.rerun()

    if col_btn_all.button("📅 Toute la période", key="btn_all_dates", use_container_width=True):
        st.session_state.d1 = min_date
        st.session_state.d2 = max_date
        st.session_state.widget_key += 1
        st.rerun()

    if col_btn_jmoins1.button("📅 J-1", key="btn_jmoins1", use_container_width=True):
        d2 = st.session_state.get("d2", max_date)
        new_d1 = max(min_date, d2 - pd.Timedelta(days=1))
        st.session_state.d1 = new_d1
        st.session_state.d2 = d2
        st.session_state.widget_key += 1
        st.rerun()

with c4:
    # Ensure d1 and d2 are proper date objects
    d1_clean = pd.Timestamp(st.session_state.d1).date() if st.session_state.d1 else min_date
    d2_clean = pd.Timestamp(st.session_state.d2).date() if st.session_state.d2 else max_date
    
    dates = st.date_input(
        "Intervalle",
        value=(d1_clean, d2_clean),
        min_value=min_date,
        max_value=max_date,
        key=f"date_range_{st.session_state.widget_key}",
        format="YYYY-MM-DD",
        label_visibility="collapsed",
    )
    
    if dates and len(dates) == 2:
        if dates[0] != st.session_state.d1 or dates[1] != st.session_state.d2:
            st.session_state.d1 = dates[0]
            st.session_state.d2 = dates[1]
            st.rerun()
# sécurisation de la sélection de dates
if isinstance(dates, (list, tuple)) and len(dates) == 2:
    d1, d2 = dates
    if d1 and d2:
        if d1 > d2: d1, d2 = d2, d1
        d1 = max(min_date, min(d1, max_date))
        d2 = max(min_date, min(d2, max_date))
        st.session_state.d1, st.session_state.d2 = d1, d2

# FILTRAGE 
d1_ts = pd.Timestamp(st.session_state.d1)
d2_ts = pd.Timestamp(st.session_state.d2) + pd.Timedelta(days=1) 
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
site_sel = st.session_state.site_sel
st.caption(
    f"**Période**: {d1_ts.date()} → {(d2_ts - pd.Timedelta(seconds=1)).date()} · "
    f"**Sites**: {nb_sites}"
)

if sess.empty:
    st.warning("Aucune donnée pour ces filtres. Essayez d’élargir l’intervalle ou de (re)sélectionner des sites.")
#Filtre2
row_type, row_moment, row_avant = st.columns([1, 1, 0.7])
type_options, moment_options = [], []
if "type_erreur" in sess.columns:
    type_options = sorted(sess["type_erreur"].dropna().unique().tolist())

if "moment" in sess.columns:
    opts = sorted(sess["moment"].dropna().unique().tolist())
    moment_options = [m for m in MOMENT_ORDER if m in opts] + [m for m in opts if m not in MOMENT_ORDER]

if "type_sel" not in st.session_state or not st.session_state["type_sel"]:
    if type_options:
        st.session_state["type_sel"] = type_options[:]
if "moment_sel" not in st.session_state or not st.session_state["moment_sel"]:
    if moment_options:
        st.session_state["moment_sel"] = moment_options[:]
if moment_options:
    st.session_state["__moment_order__"] = moment_options[:]

GROUPS = {
    "avant_charge_toggle": {"Init", "Lock Connector", "CableCheck"},
    "charge_toggle": {"Charge"},
    "fin_charge_toggle": {"Fin de charge"},
}

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

if "type_sel" in st.session_state:
    st.session_state["type_sel"] = [
        val for val in st.session_state["type_sel"]
        if val in type_options
    ]

if "moment_sel" in st.session_state:
    st.session_state["moment_sel"] = [
        val for val in st.session_state["moment_sel"]
        if val in moment_options
    ]

with row_type:
    st.multiselect(
        "Type d'erreur (global)",
        options=type_options,
        key="type_sel",
        help="Filtre global sur le type d’erreur (ex: Erreur_EVI, Erreur_DownStream)."
    )

with row_moment:
    st.multiselect(
        "Moment d'erreur",
        options=moment_options,
        key="moment_sel",
        help="S’applique seulement aux erreurs EVI et DS"
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

context.__dict__.clear()
context.__dict__.update({k: v for k, v in locals().items() if k != "context"})

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10= st.tabs(["📋 Générale", "🏢 Comparaison par site (Activité)", "🔌 Détails Site (par PDC)", "📈 Statistiques","📑 Projection pivot", "⚠️ Analyse tentatives multiples", "⚠️ Transactions suspectes", "🔍 Analyse Erreur Moment", "🔍 Analyse Erreur Spécifique", "⚠️Alertes"])
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
