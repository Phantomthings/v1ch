#Dashboard.py
import uuid
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine
import plotly.graph_objects as go

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

# TABS 
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10= st.tabs(["📋 Générale", "🏢 Comparaison par site (Activité)", "🔌 Détails Site (par PDC)", "📈 Statistiques","📑 Projection pivot", "⚠️ Analyse tentatives multiples", "⚠️ Transactions suspectes", "🔍 Analyse Erreur Moment", "🔍 Analyse Erreur Spécifique", "⚠️Alertes"])
stats_all = tables.get("stats_global_all", pd.DataFrame())
stats_ok  = tables.get("stats_global_ok",  pd.DataFrame())

# Tab 1 
with tab1:
    st.subheader("Indicateurs globaux")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total charges", total)
    c2.metric("Réussite", ok)
    c3.metric("Échec", nok)
    c4.metric("Taux de réussite", f"{taux_reussite:.2f}%")
    c5.metric("Taux d’échec", f"{taux_echec:.2f}%")

    st.divider()
    by_site_kpi = (
        sess_kpi.groupby(SITE_COL, as_index=False)
                .agg(total=("is_ok_filt", "count"),
                    ok=("is_ok_filt", "sum"))
    )

    if not by_site_kpi.empty:
        by_site_kpi["nok"] = by_site_kpi["total"] - by_site_kpi["ok"]
        by_site_kpi["taux_ok"] = (by_site_kpi["ok"] / by_site_kpi["total"] * 100).round(1)

        sites_list = by_site_kpi[SITE_COL].tolist()
        for i in range(0, len(sites_list), 6):
            row_sites = sites_list[i:i+6]
            cols = st.columns(len(row_sites))
            for col, site in zip(cols, row_sites):
                row = by_site_kpi.loc[by_site_kpi[SITE_COL] == site].iloc[0]
                col.metric(
                    label=site,
                    value=f"{int(row['ok'])}/{int(row['total'])} OK",
                    delta=f"{row['taux_ok']:.1f}% réussite"
                )
    else:
        st.info("Aucun site pour ces filtres.")
    st.divider()
    # Graphique % réussite par site
    if not by_site_kpi.empty:
        st.subheader("Taux de réussite par site %")
        fig = px.bar(
            by_site_kpi.sort_values("taux_ok"),
            x="taux_ok",
            y=SITE_COL,
            orientation="h",
            text="taux_ok",
            color=SITE_COL,
            color_discrete_map=SITE_PALETTE
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        plot(fig, "tab1_site_success")

    err = sess_kpi[~sess_kpi["is_ok_filt"]].copy()
    # Vérif minimum
    if not err.empty and "moment_avancee" in err.columns and SITE_COL in err.columns:
        # Total erreurs par projet
        total_err_per_site = err.groupby(SITE_COL).size().reset_index(name="Total_NOK")

        # Erreurs par moment avancé
        err_grouped = (
            err.groupby([SITE_COL, "moment_avancee"])
            .size()
            .reset_index(name="Nb")
            .pivot(index=SITE_COL, columns="moment_avancee", values="Nb")
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        # Stat global : total / ok / nok
        stat_global = (
            sess_kpi.groupby(SITE_COL)
            .agg(
                Total=("is_ok_filt", "count"),
                OK=("is_ok_filt", "sum"),
            )
            .reset_index()
        )
        stat_global["NOK"] = stat_global["Total"] - stat_global["OK"]
        stat_global["% OK"] = (stat_global["OK"] / stat_global["Total"] * 100).round(2)
        stat_global["% NOK"] = (stat_global["NOK"] / stat_global["Total"] * 100).round(2)

        # Fusionner les données
        recap = (
            total_err_per_site
            .merge(err_grouped, on=SITE_COL, how="left")
            .merge(stat_global[[SITE_COL, "% OK", "% NOK"]], on=SITE_COL, how="left")
            .fillna(0)
            .sort_values("Total_NOK", ascending=False)
            .reset_index(drop=True)
        )
    st.subheader("Récapitulatif des erreurs par site/moment")
    if 'recap' in locals():
        st.dataframe(recap, use_container_width=True)
    else:
        st.info("Aucune donnée récapitulative disponible pour ce périmètre.")

    if not err.empty and "moment" in err.columns:
        counts_moment = (
            err.groupby("moment")
            .size()
            .reindex(MOMENT_ORDER, fill_value=0)
            .reset_index(name="Somme de Charge_NOK")
        )
        counts_moment = counts_moment[counts_moment["Somme de Charge_NOK"] > 0]
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.pie(
                counts_moment,
                names="moment",
                values="Somme de Charge_NOK",
                title="Moment d'erreurs (EVI et DownStream) (%)",
                hole=0.25,
                color="moment",
                color_discrete_map=MOMENT_PALETTE,
                category_orders={"moment": MOMENT_ORDER},
            )
            fig.update_traces(textinfo="label+percent")
            plot(fig, "pie_erreurs_par_moment")
        with col2:
            total_row = pd.DataFrame({
                "moment": ["Total"],
                "Somme de Charge_NOK": [counts_moment["Somme de Charge_NOK"].sum()]
            })
            full_table = pd.concat([counts_moment, total_row], ignore_index=True)
            st.dataframe(full_table, use_container_width=True)
# Tab 2 
with tab2:
    st.subheader("Statistiques par site")
    by_site_f = (
        sess_kpi.groupby(SITE_COL, as_index=False)
                .agg(
                    Total_Charges=("is_ok_filt", "count"),
                    Charges_OK=("is_ok_filt", "sum")
                )
    )
    by_site_f["Charges_NOK"] = by_site_f["Total_Charges"] - by_site_f["Charges_OK"]
    by_site_f["% Réussite"] = np.where(
        by_site_f["Total_Charges"].gt(0),
        (by_site_f["Charges_OK"] / by_site_f["Total_Charges"] * 100).round(2),
        0.0
    )
    by_site_f["% Échec"] = np.where(
        by_site_f["Total_Charges"].gt(0),
        (by_site_f["Charges_NOK"] / by_site_f["Total_Charges"] * 100).round(2),
        0.0
    )
    by_site_f = by_site_f.reset_index(drop=True)
    st.dataframe(by_site_f, use_container_width=True, hide_index=True)
    # Barres 
    by_site_sorted = by_site_f.sort_values("Total_Charges", ascending=True)
    sites = by_site_sorted[SITE_COL].tolist()
    ok = by_site_sorted["Charges_OK"]
    nok = by_site_sorted["Charges_NOK"]
    ok_pct = by_site_sorted["% Réussite"]
    nok_pct = by_site_sorted["% Échec"]

    fig = go.Figure()

    # Traces en Nombre
    fig.add_bar(name="Charges OK", x=sites, y=ok, text=ok, marker_color="royalblue", visible=True)
    fig.add_bar(name="Charges NOK", x=sites, y=nok, text=nok, marker_color="orangered", visible=True)

    # Traces en %
    fig.add_bar(name="Charges OK (%)", x=sites, y=ok_pct, 
                text=[f"{v:.1f}%" for v in ok_pct], marker_color="royalblue", visible=False)
    fig.add_bar(name="Charges NOK (%)", x=sites, y=nok_pct,
                text=[f"{v:.1f}%" for v in nok_pct], marker_color="orangered", visible=False)

    # Boutons toggle
    fig.update_layout(
        barmode="group",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1.1, y=1.15,
                buttons=[
                    dict(label="Nombre", method="update",
                        args=[{"visible": [True, True, False, False]},
                            {"yaxis": {"title": "Nombre de charges"}}]),
                    dict(label="%", method="update",
                        args=[{"visible": [False, False, True, True]},
                            {"yaxis": {"title": "% de charges"}}])
                ]
            )
        ]
    )

    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True, key="chart_ok_nok_toggle")
    st.divider()
    st.subheader("Analyse temporelle")
    if "Datetime start" not in sess_kpi.columns:
        st.info("Colonne 'Datetime start' absente.")
    else:
        base = sess_kpi.copy()
        if base.empty:
            st.warning("Aucune charge sur ce périmètre")
        else:
            base["hour"] = pd.to_datetime(base["Datetime start"], errors="coerce").dt.hour

            g = (
                base.dropna(subset=["hour"])
                    .groupby([SITE_COL, "hour"])
                    .size()
                    .reset_index(name="Nb")
            )

            if g.empty:
                st.info("Pas d'heures valides pour les charges.")
            else:
                peak = g.loc[g.groupby(SITE_COL)["Nb"].idxmax()][[SITE_COL, "hour", "Nb"]] \
                        .rename(columns={"hour": "Heure de pic", "Nb": "Nb au pic"})

                def _w_median_hours(dfh: pd.DataFrame) -> int:
                    s = dfh.sort_values("hour")
                    c = s["Nb"].cumsum()
                    half = s["Nb"].sum() / 2.0
                    return int(s.loc[c >= half, "hour"].iloc[0])

                med = g.groupby(SITE_COL).apply(_w_median_hours).reset_index(name="Heure médiane")

                summ = peak.merge(med, on=SITE_COL, how="left")
                for col in ["Heure de pic", "Heure médiane"]:
                    summ[col] = summ[col].astype(int).apply(lambda x: f"{x:02d}:00")
                st.dataframe(
                    summ[[SITE_COL, "Heure de pic", "Nb au pic", "Heure médiane"]].sort_values(SITE_COL),
                    use_container_width=True,
                    hide_index=True
                )

                # HEATMAP 
                pivot = g.pivot(index=SITE_COL, columns="hour", values="Nb").fillna(0)

                fig = px.imshow(
                    pivot,
                    labels=dict(x="Heure", y="Site", color="Nb charges"),
                    x=[f"{h:02d}:00" for h in pivot.columns],
                    y=pivot.index,
                    color_continuous_scale="Blues",
                    aspect="auto"
                )

                fig.update_layout(
                    xaxis=dict(side="top")
                )

                st.plotly_chart(fig, use_container_width=True, key="heatmap_all")
                st.divider()
                site_options = summ[SITE_COL].tolist() 
                if len(site_options) == 0:
                    st.info("Aucun site disponible après filtres pour le zoom horaire.")
                else:
                    site_focus_both = st.selectbox(
                        "📊 Zoom sur un site",
                        options=site_options
                    )

                    base_site = base[base[SITE_COL] == site_focus_both].copy()
                    ok_focus_all  = base_site[base_site["is_ok_filt"]].copy()
                    nok_focus_all = base_site[~base_site["is_ok_filt"]].copy()

                    ok_focus_all["month"]  = pd.to_datetime(ok_focus_all["Datetime start"], errors="coerce").dt.to_period("M").astype(str)
                    nok_focus_all["month"] = pd.to_datetime(nok_focus_all["Datetime start"], errors="coerce").dt.to_period("M").astype(str)

                    g_ok_m  = ok_focus_all.groupby("month").size().reset_index(name="Nb").assign(Status="OK")
                    g_nok_m = nok_focus_all.groupby("month").size().reset_index(name="Nb").assign(Status="NOK")

                    g_both_m = pd.concat([g_ok_m, g_nok_m], ignore_index=True)
                    g_both_m["month"] = pd.to_datetime(g_both_m["month"], errors="coerce")
                    g_both_m = g_both_m.dropna(subset=["month"]).sort_values("month")
                    g_both_m["month"] = g_both_m["month"].dt.strftime("%Y-%m")

                    piv_m = (
                        g_both_m.pivot(index="month", columns="Status", values="Nb")
                                .fillna(0)
                                .sort_index()
                    )
                    months = piv_m.index.tolist()

                    ok_num = piv_m["OK"].tolist() if "OK" in piv_m else []
                    nok_num = piv_m["NOK"].tolist() if "NOK" in piv_m else []

                    tot_m = piv_m.sum(axis=1).replace(0, np.nan)
                    ok_pct = (piv_m["OK"] / tot_m * 100).fillna(0).round(1).tolist() if "OK" in piv_m else [0] * len(piv_m)
                    nok_pct = (piv_m["NOK"] / tot_m * 100).fillna(0).round(1).tolist() if "NOK" in piv_m else [0] * len(piv_m)

                    fig_both_m = go.Figure()
                    # Nombre
                    fig_both_m.add_bar(name="OK (Nb)",  x=months, y=ok_num,  text=ok_num,  marker_color="#38AC21", visible=True)
                    fig_both_m.add_bar(name="NOK (Nb)", x=months, y=nok_num, text=nok_num, marker_color="#EF553B", visible=True)
                    # %
                    fig_both_m.add_bar(name="OK (%)",  x=months, y=ok_pct,  text=[f"{v:.1f}%" for v in ok_pct],  marker_color="#38AC21", visible=False)
                    fig_both_m.add_bar(name="NOK (%)", x=months, y=nok_pct, text=[f"{v:.1f}%" for v in nok_pct], marker_color="#EF553B", visible=False)

                    fig_both_m.update_layout(
                        title=f"Distribution mensuelle OK vs NOK — {site_focus_both}",
                        barmode="group",
                        xaxis=dict(type="category"),
                        updatemenus=[
                            dict(
                                type="buttons", direction="right", x=1.1, y=1.15,
                                buttons=[
                                    dict(label="Nombre", method="update",
                                        args=[{"visible": [True, True, False, False]},
                                            {"yaxis": {"title": "Nombre"}}]),
                                    dict(label="%", method="update",
                                        args=[{"visible": [False, False, True, True]},
                                            {"yaxis": {"title": "%"}}]),
                                ]
                            )
                        ]
                    )
                    fig_both_m.update_traces(textposition="outside")
                    plot(fig_both_m, f"tab2_ok_nok_month_distribution_{site_focus_both}")
                    months_all = months 
                    if months_all:
                        month_focus = st.selectbox(
                            "📅 Focus: afficher le détail par jour pour un mois",
                            options=months_all,
                            index=len(months_all)-1,
                            key=f"month_focus_days_{site_focus_both}"
                        )

                        ok_mo  = ok_focus_all[ok_focus_all["month"]  == month_focus].copy()
                        nok_mo = nok_focus_all[nok_focus_all["month"] == month_focus].copy()

                        ok_mo["day"]  = pd.to_datetime(ok_mo["Datetime start"],  errors="coerce").dt.strftime("%Y-%m-%d")
                        nok_mo["day"] = pd.to_datetime(nok_mo["Datetime start"], errors="coerce").dt.strftime("%Y-%m-%d")

                        per = pd.Period(month_focus, freq="M")
                        m_start = per.to_timestamp(how="start")
                        m_end   = per.to_timestamp(how="end")
                        days = pd.date_range(m_start, m_end, freq="D").strftime("%Y-%m-%d")

                        g_ok_d  = ok_mo.groupby("day").size().reindex(days, fill_value=0).reset_index()
                        g_ok_d.columns = ["day", "Nb"];  g_ok_d["Status"] = "OK"
                        g_nok_d = nok_mo.groupby("day").size().reindex(days, fill_value=0).reset_index()
                        g_nok_d.columns = ["day", "Nb"]; g_nok_d["Status"] = "NOK"

                        g_both_d = pd.concat([g_ok_d, g_nok_d], ignore_index=True)

                        piv_d = g_both_d.pivot(index="day", columns="Status", values="Nb").fillna(0)
                        days_ord = piv_d.index.tolist()
                        ok_num_d  = (piv_d["OK"]  if "OK"  in piv_d else 0).tolist()
                        nok_num_d = (piv_d["NOK"] if "NOK" in piv_d else 0).tolist()

                        tot_d = (piv_d.sum(axis=1)).replace(0, np.nan)
                        ok_pct_d  = (piv_d["OK"]  / tot_d * 100).fillna(0).round(1).tolist() if "OK"  in piv_d else [0]*len(days_ord)
                        nok_pct_d = (piv_d["NOK"] / tot_d * 100).fillna(0).round(1).tolist() if "NOK" in piv_d else [0]*len(days_ord)

                        fig_both_d = go.Figure()
                        # Nombre
                        fig_both_d.add_bar(name="OK (Nb)",  x=days_ord, y=ok_num_d,  text=ok_num_d,  marker_color="#38AC21", visible=True)
                        fig_both_d.add_bar(name="NOK (Nb)", x=days_ord, y=nok_num_d, text=nok_num_d, marker_color="#EF553B", visible=True)
                        # %
                        fig_both_d.add_bar(name="OK (%)",  x=days_ord, y=ok_pct_d,  text=[f"{v:.1f}%" for v in ok_pct_d],  marker_color="#38AC21", visible=False)
                        fig_both_d.add_bar(name="NOK (%)", x=days_ord, y=nok_pct_d, text=[f"{v:.1f}%" for v in nok_pct_d], marker_color="#EF553B", visible=False)

                        fig_both_d.update_layout(
                            title=f"OK vs NOK par jour — {site_focus_both} — {month_focus}",
                            barmode="group",
                            xaxis=dict(type="category", tickangle=-45),
                            updatemenus=[
                                dict(
                                    type="buttons", direction="right", x=1.1, y=1.15,
                                    buttons=[
                                        dict(label="Nombre", method="update",
                                            args=[{"visible": [True, True, False, False]},
                                                {"yaxis": {"title": "Nombre"}}]),
                                        dict(label="%", method="update",
                                            args=[{"visible": [False, False, True, True]},
                                                {"yaxis": {"title": "%"}}]),
                                    ]
                                )
                            ]
                        )
                        fig_both_d.update_traces(textposition="outside")
                        plot(fig_both_d, f"tab2_ok_nok_day_distribution_{site_focus_both}_{month_focus}")
                    else:
                        st.info("Aucun mois disponible pour le focus journalier.")
with tab3:
    st.subheader("Statistiques par PDC")
    base = sess_kpi.copy() 
    # Sites dispos dans la base filtrée & avec PDC
    if (SITE_COL not in base.columns) or ("PDC" not in base.columns) or base.empty:
        st.info("Pas de colonne PDC/site ou pas de données après filtres.")
        st.stop()

    sites_avail = sorted(base[SITE_COL].dropna().unique().tolist())
    if not sites_avail:
        st.info("Aucun site disponible après filtres.")
        st.stop()

    # Sélecteur site 
    st.session_state.setdefault("tab3_site", sites_avail[0])
    site_unique = st.selectbox("🏢 Sélectionner un site", options=sites_avail, key="tab3_site")

    # Sous-ensemble du site choisi
    sess_site = base[base[SITE_COL] == site_unique].copy()
    if sess_site.empty:
        st.info("Aucune donnée pour ce site après filtres.")
        st.stop()
        
    # PDC disponibles
    sess_site["PDC"] = sess_site["PDC"].astype(str)
    pdc_order = sorted(sess_site["PDC"].dropna().unique().tolist())

    # Ajout de l’option spéciale
    options = ["✅ Tous"] + pdc_order

    # Sélection courante (state brut)
    default_raw = st.session_state.get("tab3_pdc_sel_raw", ["✅ Tous"])

    sel_raw = st.multiselect(
        f"🔌 Sélection PDC — {site_unique}",
        options=options,
        default=[o for o in default_raw if o in options],
        key="tab3_pdc_sel_raw",
        help="Choisis un ou plusieurs PDC, ou coche '✅ Tous' pour tout sélectionner."
    )

    if "✅ Tous" in sel_raw:
        selected_pdc = pdc_order[:]
    else:
        selected_pdc = [p for p in sel_raw if p in pdc_order]

    # Filtre
    sess_pdc = sess_site[sess_site["PDC"].isin(selected_pdc)].copy()
    if sess_pdc.empty:
        st.info("Aucun PDC dans la sélection.")
        st.stop()
    # Récap
    BASE_CHARGE_URL = "https://elto.nidec-asi-online.com/Charge/detail?id="
    df_src = globals().get("sess_pdc", None)
    if df_src is None or not isinstance(df_src, pd.DataFrame) or df_src.empty:
        df_src = sess  
    if df_src.empty:
        st.info("Aucune donnée disponible pour ce périmètre.")
    else:
        if "is_ok" not in df_src.columns:
            st.warning("Colonne 'is_ok' absente dans les sessions.")
        else:
            mask_type   = True
            mask_moment = True
            if "type_erreur" in df_src.columns and st.session_state.get("type_sel"):
                mask_type = df_src["type_erreur"].isin(st.session_state.type_sel)
            if {"type_erreur", "moment"}.issubset(df_src.columns) and st.session_state.get("moment_sel"):
                mask_moment = df_src["moment"].isin(st.session_state.moment_sel)
            if isinstance(mask_type, bool):
                mask_type = pd.Series([mask_type] * len(df_src), index=df_src.index)
            if isinstance(mask_moment, bool):
                mask_moment = pd.Series([mask_moment] * len(df_src), index=df_src.index)
            df_src_f = df_src[mask_type & mask_moment].copy()
            err_sum = df_src_f.loc[~df_src_f["is_ok"]].copy()
            if err_sum.empty:
                st.info("Aucune charge en erreur pour le périmètre/filtre sélectionné.")
            else:
                for c in ("Datetime start", "Datetime end"):
                    if c in err_sum.columns:
                        err_sum[c] = pd.to_datetime(err_sum[c], errors="coerce")
                if "Energy (Kwh)" in err_sum.columns:
                    err_sum["Energy (Kwh)"] = pd.to_numeric(err_sum["Energy (Kwh)"], errors="coerce")

                for c in ("SOC Start", "SOC End"):
                    if c in err_sum.columns:
                        err_sum[c] = pd.to_numeric(err_sum[c], errors="coerce")
                if "MAC Address" in err_sum.columns:
                    err_sum["MAC Address"] = err_sum["MAC Address"].apply(_fmt_mac)
                def _etiquette(row):
                    t = str(row.get("type_erreur", "") or "")
                    m = str(row.get("moment", "") or "")
                    return f"{t} — {m}" if m else t
                err_sum["Erreur"] = err_sum.apply(_etiquette, axis=1)
                def _soc_evo(row):
                    s0 = row.get("SOC Start", pd.NA)
                    s1 = row.get("SOC End", pd.NA)
                    if pd.notna(s0) and pd.notna(s1):
                        try:
                            return f"{int(round(s0))}% → {int(round(s1))}%"
                        except Exception:
                            return ""
                    return ""
                err_sum["Évolution SOC"] = err_sum.apply(_soc_evo, axis=1)
                if "ID" not in err_sum.columns:
                    st.warning("Colonne 'ID' absente : les liens ELTO ne seront pas affichés.")
                    err_sum["ELTO"] = ""
                else:
                    err_sum["ELTO"] = BASE_CHARGE_URL + err_sum["ID"].astype(str).str.strip()
                cols_aff = ["ID", "Datetime start", "Datetime end", "PDC",
                            "Energy (Kwh)", "MAC Address", "Erreur", "Évolution SOC", "ELTO"]
                cols_aff = [c for c in cols_aff if c in err_sum.columns]

                out = err_sum[cols_aff].copy()
                if "Datetime start" in out.columns:
                    out = out.sort_values("Datetime start", ascending=False)
                out.insert(0, "#", range(1, len(out) + 1))
                st.data_editor(
                    out,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ELTO": st.column_config.LinkColumn(
                            "Lien ELTO",
                            help="Ouvrir la session dans ELTO",
                            display_text="🔗 Ouvrir"
                        ),
                        "Datetime start": st.column_config.DatetimeColumn("Start time", format="YYYY-MM-DD HH:mm:ss"),
                        "Datetime end":   st.column_config.DatetimeColumn("End time",   format="YYYY-MM-DD HH:mm:ss"),
                        "Energy (Kwh)":   st.column_config.NumberColumn("Energy (kWh)", format="%.3f"),
                        "MAC Address":    st.column_config.TextColumn("MacAdress"),
                        "Erreur":         st.column_config.TextColumn("Error etiquette"),
                        "Évolution SOC":  st.column_config.TextColumn("Evolution SOC"),
                    }
                )

    # KPI PDC 
    by_pdc_f = (
        sess_pdc.groupby("PDC", as_index=False)
                .agg(Total_Charges=("is_ok_filt", "count"),
                    Charges_OK=("is_ok_filt", "sum"))
    )
    by_pdc_f["Charges_NOK"] = by_pdc_f["Total_Charges"] - by_pdc_f["Charges_OK"]
    by_pdc_f["% Réussite"]  = np.where(
        by_pdc_f["Total_Charges"].gt(0),
        (by_pdc_f["Charges_OK"] / by_pdc_f["Total_Charges"] * 100).round(2),
        0.0
    )

    st.divider()
    st.subheader("Récapitulatif des charges par PDC")
    st.dataframe(
        by_pdc_f.sort_values(["% Réussite","PDC"], ascending=[True, True]),
        use_container_width=True
    )

    if not by_pdc_f.empty:
        by_pdc_sorted = by_pdc_f.sort_values("% Réussite", ascending=True).reset_index(drop=True)
        pdc_order = by_pdc_sorted["PDC"].astype(str).tolist()
        import plotly.express as px
        palette = px.colors.qualitative.D3 + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
        color_map = {p: palette[i % len(palette)] for i, p in enumerate(pdc_order)}
        vmin, vmax = float(by_pdc_sorted["% Réussite"].min()), float(by_pdc_sorted["% Réussite"].max())
        texts = [f"{v:.1f}% {'🔻' if v==vmin else ('🔺' if v==vmax else '')}" for v in by_pdc_sorted["% Réussite"]]

        title_site = f" — {site_unique}" if "site_unique" in locals() else ""
        fig = px.bar(
            by_pdc_sorted,
            x="% Réussite", y="PDC",
            orientation="h",
            color="PDC", color_discrete_map=color_map,
            text=texts,
            title=f"Taux de réussite par PDC{title_site} (%)",
            labels={"% Réussite": "% Réussite", "PDC": "PDC"}
        )
        fig.update_traces(
            textposition="outside",
            marker_line_width=0  
        )
        fig.update_layout(
            xaxis=dict(range=[0, 100]),
            showlegend=True,
            yaxis=dict(categoryorder="array", categoryarray=pdc_order)  
        )
        try:
            plot(fig, f"tab3_pdc_success_{site_unique}")
        except Exception:
            st.plotly_chart(fig, use_container_width=True)

    err_site = sess_pdc[~sess_pdc["is_ok_filt"]].copy()
    col1, col2 = st.columns(2)

    # Bar % moments EVI
    with col1:
        err_site_evi = err_site[err_site["type_erreur"] == "Erreur_EVI"].copy()
        if not err_site_evi.empty and "moment" in err_site_evi.columns:
            counts = (
                err_site_evi.groupby("moment")
                            .size()
                            .reindex(MOMENT_ORDER, fill_value=0)
                            .reset_index(name="Nb")
            )
            total_m = counts["Nb"].sum()
            if total_m > 0:
                counts["%"] = (counts["Nb"] / total_m * 100).round(2)
                fig = px.bar(
                    counts,
                    x="moment", y="%",
                    text="%",
                    color="moment",
                    category_orders={"moment": MOMENT_ORDER},
                    color_discrete_map=MOMENT_PALETTE,
                    title=f"Erreurs EVI par moment — {site_unique} (%)",
                    labels={"moment": "Moment", "%": "% Erreurs"}
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(yaxis=dict(range=[0, 100]))
                plot(fig, f"tab3_moment_pct_{site_unique}")
            else:
                st.info("Aucune erreur EVI sur ce périmètre.")
        else:
            st.info("Aucune erreur EVI sur ce périmètre.")

    # Pie moments EVI
    with col2:
        if not err_site_evi.empty and "moment" in err_site_evi.columns:
            counts_m = (
                err_site_evi.groupby("moment")
                            .size()
                            .reindex(MOMENT_ORDER, fill_value=0)
                            .reset_index(name="Nb")
            )
            counts_m = counts_m[counts_m["Nb"] > 0]
            if not counts_m.empty:
                fig = px.pie(
                    counts_m,
                    names="moment",
                    values="Nb",
                    title=f"Moments d’erreur EVI — {site_unique} (%)",
                    hole=0.25,
                    color="moment",
                    color_discrete_map=MOMENT_PALETTE,
                    category_orders={"moment": MOMENT_ORDER},
                )
                fig.update_traces(textinfo="label+percent")
                plot(fig, f"tab3_moments_pie_{site_unique}")
            else:
                st.info("Aucun moment d’erreur EVI sur ce périmètre.")
        else:
            st.info("Aucune erreur EVI sur ce périmètre.")

    if not err_site_evi.empty and "moment" in err_site_evi.columns:
        counts_grouped = (
            err_site_evi.groupby("moment")
                        .size()
                        .reset_index(name="Nb")
        )
        mapping = {
            "Init": "Avant charge",
            "Lock Connector": "Avant charge",
            "CableCheck": "Avant charge",
            "Charge": "Charge",
            "Fin de charge": "Fin de charge",
            "Unknown": "Unknown"
        }
        counts_grouped["Moment_grp"] = counts_grouped["moment"].map(mapping)

        counts_grouped = (
            counts_grouped.groupby("Moment_grp", as_index=False)["Nb"].sum()
                        .sort_values("Nb", ascending=False)
        )

        if not counts_grouped.empty:
            fig = px.pie(
                counts_grouped,
                names="Moment_grp",
                values="Nb",
                title=f"Moments d’erreur EVI (Avancé) — {site_unique} (%)",
                hole=0.25,
                color="Moment_grp",
                color_discrete_map={
                    "Avant charge": "#636EFA",
                    "Charge": "#00CC96",
                    "Fin de charge": "#AB63FA",
                    "Unknown": "#FFA15A"
                }
            )
            fig.update_traces(textinfo="label+percent")
            plot(fig, f"tab3_moments_grouped_pie_{site_unique}")
        else:
            st.info("Aucun moment d’erreur regroupé sur ce périmètre.")

    st.divider()
    st.subheader(f"Occurrences des erreurs par code — {site_unique}")

    if not err_site.empty:
        cols_occ = st.columns(2)
        with cols_occ[0]:
            st.markdown("**Downstream Code PC × Moment**")
            need_cols = {"Downstream Code PC", "moment"}
            if need_cols.issubset(err_site.columns):
                ds_num = pd.to_numeric(err_site["Downstream Code PC"], errors="coerce").fillna(0).astype(int)
                evi_code = pd.to_numeric(err_site.get("EVI Error Code", 0), errors="coerce").fillna(0).astype(int)

                # Appliquer la vraie logique "Downstream"
                mask_downstream = (ds_num != 0) & (ds_num != 8192)
                sub = err_site.loc[mask_downstream, ["Downstream Code PC", "moment"]].copy()

                if sub.empty:
                    st.info("Aucun Downstream Code PC non nul sur ce périmètre.")
                else:
                    sub["Code_PC"] = pd.to_numeric(sub["Downstream Code PC"], errors="coerce").astype(int)
                    tmp = (sub.groupby(["Code_PC", "moment"])
                                .size()
                                .reset_index(name="Occurrences"))
                    moment_order = MOMENT_ORDER if "MOMENT_ORDER" in globals() else sorted(tmp["moment"].unique())

                    table = (tmp.pivot(index="Code_PC", columns="moment", values="Occurrences")
                                .reindex(columns=moment_order, fill_value=0)
                                .reset_index())

                    table["Total"] = table[moment_order].sum(axis=1)
                    table = table.sort_values("Total", ascending=False).reset_index(drop=True)

                    total_all = int(table["Total"].sum())
                    table["%"] = (table["Total"] / total_all * 100).round(2).astype(str) + " %"

                    table.insert(0, "#", range(1, len(table) + 1))
                    total_row = {"#": "", "Code_PC": "Total", **{m: int(table[m].sum()) for m in moment_order}}
                    total_row["Total"] = int(table["Total"].sum())
                    total_row["%"] = "100 %"

                    st.dataframe(pd.concat([table, pd.DataFrame([total_row])], ignore_index=True),
                                use_container_width=True)
            else:
                st.info("Colonnes requises absentes : 'Downstream Code PC' et/ou 'moment'.")
        with cols_occ[1]:
            st.markdown("**EVI Error Code × Moment**")
            need_cols = {"EVI Error Code", "moment"}
            if need_cols.issubset(err_site.columns):
                ds_num = pd.to_numeric(err_site.get("Downstream Code PC", 0), errors="coerce").fillna(0).astype(int)
                evi_code = pd.to_numeric(err_site["EVI Error Code"], errors="coerce").fillna(0).astype(int)

                # Appliquer la vraie logique "EVI"
                mask_evi = (ds_num == 8192) | ((ds_num == 0) & (evi_code != 0))
                sub = err_site.loc[mask_evi, ["EVI Error Code", "moment"]].copy()
                if sub.empty:
                    st.info("Aucun EVI Error Code non nul sur ce périmètre.")
                else:
                    sub["EVI_Code"] = pd.to_numeric(sub["EVI Error Code"], errors="coerce").astype(int)
                    tmp = (sub.groupby(["EVI_Code", "moment"])
                                .size()
                                .reset_index(name="Occurrences"))
                    moment_order = MOMENT_ORDER if "MOMENT_ORDER" in globals() else sorted(tmp["moment"].unique())

                    table = (tmp.pivot(index="EVI_Code", columns="moment", values="Occurrences")
                                .reindex(columns=moment_order, fill_value=0)
                                .reset_index())

                    table["Total"] = table[moment_order].sum(axis=1)
                    table = table.sort_values("Total", ascending=False).reset_index(drop=True)

                    total_all = int(table["Total"].sum())
                    table["%"] = (table["Total"] / total_all * 100).round(2).astype(str) + " %"

                    table.insert(0, "#", range(1, len(table) + 1))

                    total_row = {"#": "", "EVI_Code": "Total", **{m: int(table[m].sum()) for m in moment_order}}
                    total_row["Total"] = int(table["Total"].sum())
                    total_row["%"] = "100 %"

                    st.dataframe(pd.concat([table, pd.DataFrame([total_row])], ignore_index=True),
                                use_container_width=True)
            else:
                st.info("Colonnes requises absentes : 'EVI Error Code' et/ou 'moment'.")
    else:
        st.info("Aucune erreur (site + PDC sélectionnés) pour afficher les occurrences par code.")
    
with tab4:
    st.subheader("Statistiques générales")
    # CSS
    st.markdown("""
        <style>
        .kpi-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        .kpi-card {
          background: #0f172a; border: 1px solid #1f2937; border-radius: 14px;
          padding: 14px 16px; box-shadow: 0 2px 10px rgba(0,0,0,.15);
        }
        .kpi-title{ font-size:13px; color:#cbd5e1; margin:0 0 6px 0; }
        .kpi-value{ font-size:24px; font-weight:700; color:#f8fafc; margin:0; }
        .kpi-sub{ font-size:12px; color:#94a3b8; margin-top:4px; }
        .kpi-tag{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; margin-left:6px; background:#111827; border:1px solid #334155; color:#93c5fd;}
        .sec { font-weight:600; margin:18px 0 8px 0; }
        </style>
        """, unsafe_allow_html=True)

    def card(title, value, sub=""):
        st.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>""",
            unsafe_allow_html=True
        )

    if "is_ok" not in sess.columns:
        st.info("Colonne is_ok absente.")
    else:
        ok = sess[sess["is_ok"]].copy()
        if ok.empty:
            st.warning("Aucune charge OK dans ce périmètre.")
        else:
            dt_s   = pd.to_datetime(ok.get("Datetime start"), errors="coerce")
            dt_e   = pd.to_datetime(ok.get("Datetime end"), errors="coerce")
            energy = pd.to_numeric(ok.get("Energy (Kwh)"), errors="coerce")
            pmean  = pd.to_numeric(ok.get("Mean Power (Kw)"), errors="coerce")
            pmax   = pd.to_numeric(ok.get("Max Power (Kw)"), errors="coerce")
            soc_s  = pd.to_numeric(ok.get("SOC Start"), errors="coerce")
            soc_e  = pd.to_numeric(ok.get("SOC End"), errors="coerce")
            dur_min = (dt_e - dt_s).dt.total_seconds() / 60

            def date_of(idx):
                if pd.isna(idx) or idx not in ok.index: return "—"
                d = dt_e.loc[idx]
                if pd.isna(d): d = dt_s.loc[idx]
                return d.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(d) else "—"

            def lieu_of(idx):
                if pd.isna(idx) or idx not in ok.index: return "—"
                site = str(ok.loc[idx].get("Site", ok.loc[idx].get("Name Project", ""))) or "—"
                pdc  = str(ok.loc[idx].get("PDC", "—"))
                return f"{site} — PDC {pdc}"

            def idxmin_thresh(series: pd.Series, thr: float):
                s = series.where(series >= thr)
                return s.idxmin() if s.notna().any() else np.nan

            # seuils minima à ignorer
            THR_ENERGY = 4.0
            THR_PMEAN  = 4.0
            THR_PMAX   = 4.0

            # ÉNERGIE
            e_total = round(float(energy.sum(skipna=True)), 3) if energy.notna().any() else 0
            e_mean  = round(float(energy.mean(skipna=True)), 3) if energy.notna().any() else 0
            e_min_i = idxmin_thresh(energy, THR_ENERGY)
            e_max_i = energy.idxmax() if energy.notna().any() else np.nan
            e_min_v = (round(float(energy.loc[e_min_i]), 3) if e_min_i==e_min_i else "—")
            e_max_v = (round(float(energy.loc[e_max_i]), 3) if e_max_i==e_max_i else "—")

            # Pmean
            pm_mean = round(float(pmean.mean(skipna=True)), 3) if pmean.notna().any() else 0
            pm_min_i = idxmin_thresh(pmean, THR_PMEAN)
            pm_max_i = pmean.idxmax() if pmean.notna().any() else np.nan
            pm_min_v = (round(float(pmean.loc[pm_min_i]), 3) if pm_min_i==pm_min_i else "—")
            pm_max_v = (round(float(pmean.loc[pm_max_i]), 3) if pm_max_i==pm_max_i else "—")

            # Pmax
            px_mean = round(float(pmax.mean(skipna=True)), 3) if pmax.notna().any() else 0
            px_min_i = idxmin_thresh(pmax, THR_PMAX)
            px_max_i = pmax.idxmax() if pmax.notna().any() else np.nan
            px_min_v = (round(float(pmax.loc[px_min_i]), 3) if px_min_i==px_min_i else "—")
            px_max_v = (round(float(pmax.loc[px_max_i]), 3) if px_max_i==px_max_i else "—")

            # SOC
            soc_start_mean = round(float(soc_s.mean(skipna=True)), 2) if soc_s.notna().any() else 0
            soc_end_mean   = round(float(soc_e.mean(skipna=True)), 2) if soc_e.notna().any() else 0

            # Durées
            d_mean = round(float(dur_min.mean(skipna=True)), 1) if dur_min.notna().any() else 0
            d_min_i = idxmin_thresh(dur_min, 1.0) 
            d_max_i = dur_min.idxmax() if dur_min.notna().any() else np.nan
            d_min_v = (round(float(dur_min.loc[d_min_i]), 1) if d_min_i==d_min_i else "—")
            d_max_v = (round(float(dur_min.loc[d_max_i]), 1) if d_max_i==d_max_i else "—")
            st.divider()
            # ÉNERGIE
            st.markdown('#### ⚡ Énergie <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: card("Total (kWh)", f"{e_total}")
            with c2: card("Moyenne (kWh)", f"{e_mean}")
            with c3: card("Min (kWh) (≥4)", f"{e_min_v}", f"{date_of(e_min_i)} — {lieu_of(e_min_i)}")
            c4, c5, _ = st.columns(3)
            with c4: card("Max (kWh)", f"{e_max_v}", f"{date_of(e_max_i)} — {lieu_of(e_max_i)}")
            st.divider()    
            # Pmean
            st.markdown('#### 🔌 Puissance moyenne (kW) <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: card("Moyenne (kW)", f"{pm_mean}")
            with c2: card("Min (kW) (≥4)", f"{pm_min_v}", f"{date_of(pm_min_i)} — {lieu_of(pm_min_i)}")
            with c3: card("Max (kW)", f"{pm_max_v}", f"{date_of(pm_max_i)} — {lieu_of(pm_max_i)}")
            st.divider()
            # Pmax
            st.markdown('#### 🚀 Puissance maximale (kW) <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: card("Moyenne (kW)", f"{px_mean}")
            with c2: card("Min (kW) (≥4)", f"{px_min_v}", f"{date_of(px_min_i)} — {lieu_of(px_min_i)}")
            with c3: card("Max (kW)", f"{px_max_v}", f"{date_of(px_max_i)} — {lieu_of(px_max_i)}")
            st.divider()
            # SOC
            st.markdown('#### 🔋 SOC <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)
            if soc_s.notna().any() and soc_e.notna().any():
                soc_gain = soc_e - soc_s
                soc_gain_mean = round(float(soc_gain.mean(skipna=True)), 2)
            else:
                soc_gain_mean = "—"
            c1, c2, c3 = st.columns(3)
            with c1: card("SOC début moyen (%)", f"{soc_start_mean}")
            with c2: card("SOC fin moyen (%)", f"{soc_end_mean}")
            with c3: card("SOC moyen de recharge (%)", f"{soc_gain_mean}")
            st.divider()

            st.markdown('#### 🔋 Charges 900V <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)
            c900 = pd.to_numeric(sess["charge_900V"], errors="coerce").fillna(0).astype(int)

            total_900 = int(c900.sum())
            total_all = len(sess)
            pct_900   = round(total_900 / total_all * 100, 2) if total_all > 0 else 0.0

            c1, c2, c3 = st.columns(3)
            with c2: card("Total charges 900V", f"{total_900}")
            with c1: card("Total charges", f"{total_all}")
            with c3: card("% en 900V", f"{pct_900}%")
            st.divider()
            # Durées
            st.markdown('#### ⏱️ Durées de charge (min) <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: card("Moyenne (min)", f"{d_mean}")
            with c2: card("Min (min)", f"{d_min_v}", f"{date_of(d_min_i)} — {lieu_of(d_min_i)}")
            with c3: card("Max (min)", f"{d_max_v}", f"{date_of(d_max_i)} — {lieu_of(d_max_i)}")
            
            # Charge par jour
            st.divider()
            st.markdown('#### 📅 Charges par jour <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)

            d_site = tables.get("charges_daily_by_site", pd.DataFrame()).copy()

            if d_site.empty:
                st.info("Feuille 'charges_daily_by_site' absente (relance kpi_cal).")
            else:
                # garder seulement le périmètre de sites déjà filtré dans l'app (si présent)
                if "Site" in d_site.columns and site_sel:
                    d_site = d_site[d_site["Site"].isin(site_sel)]

                # OK only + filtre dates
                d_site = d_site[d_site["Status"] == "OK"].copy()
                d_site["day_dt"] = pd.to_datetime(d_site["day"], errors="coerce")
                d_site = d_site.dropna(subset=["day_dt"])

                d1_day = pd.to_datetime(d1_ts).floor("D")
                d2_day = (pd.to_datetime(d2_ts) - pd.Timedelta(seconds=1)).floor("D")
                d_site = d_site[(d_site["day_dt"] >= d1_day) & (d_site["day_dt"] <= d2_day)]

                if d_site.empty:
                    st.info("Aucune charge OK sur la période (après filtres).")
                else:
                    # Total journalier global (tous sites confondus)
                    daily_tot = (
                        d_site.groupby("day_dt", as_index=False)["Nb"].sum()
                            .sort_values("day_dt")
                    )

                    nb_days  = int(daily_tot["day_dt"].nunique())
                    mean_day = round(float(daily_tot["Nb"].mean()), 2) if nb_days else 0.0
                    med_day  = round(float(daily_tot["Nb"].median()), 2) if nb_days else 0.0

                    c1, c2, c3 = st.columns(3)
                    with c1: card("Jours couverts", f"{nb_days}")
                    with c2: card("Moyenne / jour (OK)", f"{mean_day}")
                    with c3: card("Médiane / jour (OK)", f"{med_day}")

                    # Min / Max global 
                    max_row = d_site.loc[d_site["Nb"].idxmax()]
                    max_date = max_row["day_dt"]
                    max_site = str(max_row["Site"])
                    max_v = int(max_row["Nb"])
                    min_row = d_site.loc[d_site["Nb"].idxmin()]
                    min_date = min_row["day_dt"]
                    min_site = str(min_row["Site"])
                    min_v = int(min_row["Nb"])

                    c4, c5, _ = st.columns(3)
                    with c4:
                        card(
                            "Min / jour (OK)",
                            f"{int(min_v)}",  # ❌ min_v n'existe pas encore ici
                            f"{min_date.strftime('%Y-%m-%d')} — site: {min_site} ({max_site})"
                        )
                    with c5:
                        card(
                            "Max / jour (OK)",
                            f"{max_v}",
                            f"{max_date.strftime('%Y-%m-%d')} — site: {max_site}"
                        )

            st.divider()
            st.subheader("Taux de réussite/échec par type de véhicule")
            charges_mac = tables.get("charges_mac", pd.DataFrame())
            if charges_mac.empty:
                st.info("Feuille 'charges_mac' absente.")
            else:
                dfv = charges_mac.copy()

                if "Datetime start" in dfv.columns:
                    dfv["Datetime start"] = pd.to_datetime(dfv["Datetime start"], errors="coerce")

                site_col_v = "Site" if "Site" in dfv.columns else ("Name Project" if "Name Project" in dfv.columns else None)

                if site_col_v is None or "Datetime start" not in dfv.columns or "is_ok" not in dfv.columns:
                    st.info("Colonnes requises manquantes dans 'charges_mac'.")
                else:
                    dfv["is_ok"] = dfv["is_ok"].map(
                        lambda x: True if str(x).strip().lower() in {"1", "true", "vrai", "yes", "y"} else False
                    )
                    veh = dfv["Vehicle"].astype(str) if "Vehicle" in dfv.columns else pd.Series("", index=dfv.index, dtype="object")
                    veh = veh.str.strip()
                    veh = veh.replace({"": np.nan, "nan": np.nan, "none": np.nan, "NULL": np.nan}, regex=False)
                    dfv["Vehicle"] = veh.fillna("Unknown")
                    mask_v = (
                        dfv[site_col_v].isin(st.session_state.site_sel)
                        & dfv["Datetime start"].ge(pd.Timestamp(st.session_state.d1))
                        & dfv["Datetime start"].lt(pd.Timestamp(st.session_state.d2) + pd.Timedelta(days=1))
                    )
                    dfv = dfv.loc[mask_v].copy()
                    dfv = dfv[dfv["Vehicle"] != "Unknown"]

                    if dfv.empty:
                        st.info("Aucune donnée Vehicle (hors Unknown) sur ce périmètre.")
                    else:
                        g = (
                            dfv.groupby("Vehicle", dropna=False)["is_ok"]
                            .agg(total="size", ok="sum")
                            .reset_index()
                        )
                        g["nok"] = g["total"] - g["ok"]
                        g["% Réussite"] = np.where(g["total"].gt(0), (g["ok"] / g["total"] * 100).round(2), 0.0)
                        g["% Échec"] = 100 - g["% Réussite"]
                        g = g.sort_values(["total", "% Réussite"], ascending=[False, False]).reset_index(drop=True)
                        st.dataframe(g, use_container_width=True)
                        fig2 = px.bar(
                            g, x="Vehicle", y="% Réussite",
                            labels={"% Réussite": "% Réussite", "Vehicle": "Vehicle"},
                            title="Taux de réussite par type de Vehicle (%)",
                            color="% Réussite", color_continuous_scale="GnBu"
                        )
                        fig2.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
                        fig2.update_layout(coloraxis_showscale=False, xaxis=dict(type="category"), yaxis=dict(range=[0, 100]))
                        plot(fig2, "tab4_Vehicle_success_no_unknown")
            st.divider()
            st.subheader("⏱️ Durée de fonctionnement totale (heures)")

            dsd = tables.get("durations_site_daily", pd.DataFrame()).copy()
            dpd = tables.get("durations_pdc_daily",  pd.DataFrame()).copy()

            if dsd.empty and dpd.empty:
                st.info("Tables pré-calculées absentes : 'durations_site_daily' / 'durations_pdc_daily'.")
            else:
                d1_day = pd.to_datetime(d1_ts).floor("D")
                d2_day = (pd.to_datetime(d2_ts) - pd.Timedelta(seconds=1)).floor("D")
                if "day" in dsd.columns: dsd["day"] = pd.to_datetime(dsd["day"], errors="coerce")
                if "day" in dpd.columns: dpd["day"] = pd.to_datetime(dpd["day"], errors="coerce")

                # PAR SITE
                if not dsd.empty:
                    m_site = dsd["Site"].isin(site_sel) & dsd["day"].ge(d1_day) & dsd["day"].le(d2_day)
                    by_site = (
                        dsd.loc[m_site]
                        .groupby("Site", as_index=False)["dur_min"].sum()
                        .assign(Heures=lambda d: (d["dur_min"] / 60).round(1))
                        [["Site", "Heures"]]
                        .sort_values("Heures", ascending=False)  # plus élevé → plus bas
                        .reset_index(drop=True)
                    )
                    st.dataframe(by_site, use_container_width=True)

                    # couleurs distinctes
                    import plotly.express as px
                    palette = px.colors.qualitative.D3 + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
                    cats_sites = by_site["Site"].tolist()
                    color_map_sites = {s: palette[i % len(palette)] for i, s in enumerate(cats_sites)}

                    h_min, h_max = float(by_site["Heures"].min()), float(by_site["Heures"].max())
                    site_text = [f"{h} {'🔺' if h==h_max else ('🔻' if h==h_min else '')}" for h in by_site["Heures"]]

                    fig_site = px.bar(
                        by_site, x="Heures", y="Site",
                        orientation="h", text=site_text,
                        color="Site", color_discrete_map=color_map_sites,
                        title="Durée totale par site (heures)",
                    )
                    fig_site.update_traces(textposition="outside")
                    fig_site.update_layout(
                        showlegend=False,
                        yaxis=dict(categoryorder="array", categoryarray=cats_sites), 
                    )
                    plot(fig_site, "dur_site_precalc")

                if not dpd.empty:
                    m_pdc_all = dpd["Site"].isin(site_sel) & dpd["day"].ge(d1_day) & dpd["day"].le(d2_day)
                    by_pdc_all = (
                        dpd.loc[m_pdc_all]
                        .groupby(["Site", "PDC"], as_index=False)["dur_min"].sum()
                        .assign(Heures=lambda d: (d["dur_min"] / 60).round(1))
                        [["Site", "PDC", "Heures"]]
                    )
            st.divider()
            if not dpd.empty:
                m_pdc_all = dpd["Site"].isin(site_sel) & dpd["day"].ge(d1_day) & dpd["day"].le(d2_day)
                by_pdc_all = (
                    dpd.loc[m_pdc_all]
                    .groupby(["Site", "PDC"], as_index=False)["dur_min"].sum()
                    .assign(Heures=lambda d: (d["dur_min"] / 60).round(1))
                    [["Site", "PDC", "Heures"]]
                )
                sites_opts = sorted(by_pdc_all["Site"].dropna().unique().tolist())
                if sites_opts:
                    site_focus = st.selectbox("🏢 Site", options=sites_opts, index=0, key="dur_site_sel_tab4_precalc")

                    bp = (
                        by_pdc_all[by_pdc_all["Site"] == site_focus]
                        .drop(columns=["Site"])
                        .sort_values("Heures", ascending=True) 
                        .reset_index(drop=True)
                    )
                    st.dataframe(bp, use_container_width=True)

                    if not bp.empty:
                        import plotly.graph_objects as go
                        import plotly.express as px

                        # palette franche (pas de fade)
                        palette = px.colors.qualitative.D3 + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly

                        h_min, h_max = float(bp["Heures"].min()), float(bp["Heures"].max())

                        fig_pdc = go.Figure()
                        for i, row in bp.iterrows():
                            pdc = str(row["PDC"])
                            h   = float(row["Heures"])
                            txt = f"{h} {'🔺' if h==h_max else ('🔻' if h==h_min else '')}"
                            fig_pdc.add_bar(
                                x=[h], y=[pdc], orientation="h",
                                marker=dict(color=palette[i % len(palette)]),  # pas de line => pas de contour
                                text=[txt], textposition="outside",
                                name=pdc, showlegend=False
                            )

                        fig_pdc.update_traces(marker_line_width=0)

                        cats_pdc = bp["PDC"].astype(str).tolist()
                        fig_pdc.update_layout(
                            title=f"Durée totale par PDC — {site_focus} (heures)",
                            yaxis=dict(categoryorder="array", categoryarray=cats_pdc),
                            plot_bgcolor="#ffffff",
                            paper_bgcolor="#ffffff",
                            bargap=0.15
                        )
                        if hasattr(fig_pdc.layout, "coloraxis"):
                            fig_pdc.layout.coloraxis = None
                        plot(fig_pdc, f"dur_pdc_precalc_{site_focus}") 

                        
         
with tab5: 
    st.subheader("Projection pivot — Moments (ligne 1) × Codes (ligne 2)")

    if "evi_combo_long" not in tables:
        st.info("Feuille 'evi_combo_long' absente. Cliquer sur Mettre à jour.")
    else: 
        # filtre périmètre
        evi_long = tables["evi_combo_long"].copy()
        dt_evi = pd.to_datetime(evi_long["Datetime start"], errors="coerce")
        mask_evi = evi_long["Site"].isin(site_sel) & dt_evi.ge(d1_ts) & dt_evi.lt(d2_ts)

        cols_base = ["Site", "Datetime start", "EVI_Code", "EVI_Step", "step_num", "code_num", "moment"]
        if "PDC" in evi_long.columns:
            cols_base.append("PDC")
        evi_f = evi_long.loc[mask_evi, cols_base].copy()

        # filtre moments 
        if "moment_sel" in st.session_state and st.session_state.moment_sel:
            evi_f = evi_f[evi_f["moment"].isin(st.session_state.moment_sel)]
        if evi_f.empty:
            st.info("Aucune combinaison sur ce périmètre (après filtres).")
            
        else:
            site_list = sorted(evi_f["Site"].unique())

        for site in sites_list:
            st.markdown(f"### 📍 {site}")
            hide_zero = st.checkbox("Masquer colonnes vides (0)", key=f"hide_zeros_{site}")

            evi_site = evi_f[evi_f["Site"] == site].copy()
            evi_site["step_num"] = pd.to_numeric(evi_site["step_num"], errors="coerce").fillna(0).astype(int)
            evi_site["code_num"] = pd.to_numeric(evi_site["code_num"], errors="coerce").fillna(0).astype(int)

            if "PDC" in evi_site.columns:
                g_pdc = evi_site.groupby(["PDC", "step_num", "code_num"]).size().rename("Nb").reset_index()
                g_tot = evi_site.groupby(["step_num", "code_num"]).size().rename("Nb").reset_index()
                g_tot["PDC"] = "__TOTAL__"
                full = pd.concat([g_tot, g_pdc], ignore_index=True)

                pv = full.pivot_table(
                    index="PDC",
                    columns=["step_num", "code_num"],
                    values="Nb",
                    fill_value=0,
                    aggfunc="sum",
                )

                steps = sorted({c[0] for c in pv.columns})
                codes = sorted({c[1] for c in pv.columns})
                wanted = pd.MultiIndex.from_product(
                    [steps, codes],
                    names=["Moments (ligne 1)", "Codes (ligne 2)"]
                )
                pv = pv.reindex(columns=wanted, fill_value=0)

                pdcs = sorted(pv.index.tolist(), key=str)
                if "__TOTAL__" in pdcs:
                    pdcs.remove("__TOTAL__")
                    pdcs = ["__TOTAL__"] + pdcs
                pv = pv.reindex(pdcs)

                df_disp = pv.reset_index()
                df_disp["Site / PDC"] = np.where(
                    df_disp["PDC"].eq("__TOTAL__"),
                    f"{site} (TOTAL)",
                    "   " + df_disp["PDC"].astype(str)
                )
                df_disp = df_disp.drop(columns=["PDC"])
            else:
                g_site = evi_site.groupby(["step_num", "code_num"]).size().rename("Nb").reset_index()
                pv = g_site.pivot_table(
                    index=pd.Index([site], name="Site"),
                    columns=["step_num", "code_num"],
                    values="Nb",
                    fill_value=0,
                    aggfunc="sum",
                )

                steps = sorted({c[0] for c in pv.columns})
                codes = sorted({c[1] for c in pv.columns})
                wanted = pd.MultiIndex.from_product(
                    [steps, codes],
                    names=["Moments (ligne 1)", "Codes (ligne 2)"]
                )
                pv = pv.reindex(columns=wanted, fill_value=0)

                df_disp = pv.reset_index()
                df_disp["Site / PDC"] = f"{site} (TOTAL)"

            # Réorganisation colonnes
            cols = df_disp.columns
            disp_col = ("Site / PDC", "") if isinstance(cols, pd.MultiIndex) and ("Site / PDC", "") in cols else "Site / PDC"
            if isinstance(cols, pd.MultiIndex):
                value_cols = [c for c in cols if c != disp_col]
                df_disp = df_disp.loc[:, [disp_col] + value_cols]
            else:
                value_cols = [c for c in cols if c != "Site / PDC"]
                df_disp = df_disp[["Site / PDC"] + value_cols]

            # Calcul total par ligne
            _total_col = ("∑", "Total") if (len(value_cols) and isinstance(value_cols[0], tuple)) else "∑ Total"
            _numeric_all = df_disp[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            df_disp[_total_col] = _numeric_all.sum(axis=1)

            # Masquage des colonnes avec uniquement des zéros
            if hide_zero:
                col_sums = _numeric_all.sum(axis=0)
                value_cols = [c for c in value_cols if col_sums[c] > 0]

            # Calcul total global
            _numeric_base = df_disp[df_disp[disp_col].astype(str).str.startswith("   ")].copy()
            if _numeric_base.empty:
                _numeric_base = df_disp.copy()
            _col_totals = _numeric_base[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()

            _sum_dict = {disp_col: "TOTAL GÉNÉRAL"}
            _sum_dict.update({col: _col_totals[col] for col in value_cols})
            _sum_dict[_total_col] = float(_col_totals.sum())
            _sum_row = pd.DataFrame([_sum_dict], columns=[disp_col] + value_cols + [_total_col])

            df_disp = pd.concat([df_disp, _sum_row], ignore_index=True)

            final_cols = [disp_col] + value_cols + [_total_col]

            def _cell_color(v):
                try:
                    x = float(v)
                except:
                    return ""
                if x == 0: return "background-color: #ffffff;"
                elif x <= 2: return "background-color: #E8F1FB;"
                elif x <= 6: return "background-color: #CFE3F7;"
                elif x <= 15: return "background-color: #A9CFF2;"
                elif x <= 25: return "background-color: #7DB5EA;"
                elif x <= 50: return "background-color: #4F97D9; color: white;"
                elif x <= 100: return "background-color: #2F6FB7; color: white;"
                else: return "background-color: #1F4F8F; color: white;"

            styled = (
                df_disp[final_cols]
                .style
                .applymap(_cell_color, subset=value_cols)
                .format(precision=0, na_rep="")
                .set_table_styles([
                    {"selector": "th.col_heading.level0", "props": [("text-align", "center")]},
                    {"selector": "th.col_heading.level1", "props": [("text-align", "center")]},
                ])
            )
            st.dataframe(styled, use_container_width=True)
        st.markdown("""
        **Légende (occurrences)**  
        <span style="display:inline-block;width:14px;height:14px;background:#ffffff;border:1px solid #ddd;"></span> 0  
        <span style="display:inline-block;width:14px;height:14px;background:#E8F1FB;"></span> 0–2  
        <span style="display:inline-block;width:14px;height:14px;background:#CFE3F7;"></span> 2–6  
        <span style="display:inline-block;width:14px;height:14px;background:#A9CFF2;"></span> 6–15  
        <span style="display:inline-block;width:14px;height:14px;background:#7DB5EA;"></span> 15–25  
        <span style="display:inline-block;width:14px;height:14px;background:#4F97D9;"></span> 25–50  
        <span style="display:inline-block;width:14px;height:14px;background:#2F6FB7;"></span> 50–100  
        <span style="display:inline-block;width:14px;height:14px;background:#1F4F8F;"></span> >100
        <style>
        .stDataFrame table thead th:first-child {min-width: 220px !important;}
        </style>
        """, unsafe_allow_html=True)


with tab6: 
    st.subheader("Tentatives multiples dans la même heure du même utilisateur")
    multi_src = tables.get("multi_attempts_hour", pd.DataFrame())
    if multi_src.empty:
        st.info("Feuille 'multi_attempts_hour' absente (lance la mise à jour).")
    else:
        dfm = multi_src.copy()
        dfm["Date_heure"] = pd.to_datetime(dfm["Date_heure"], errors="coerce")
        mask = dfm["Site"].isin(st.session_state.site_sel) & dfm["Date_heure"].between(d1_ts, d2_ts)
        dfm = dfm.loc[mask].copy().sort_values(["Date_heure","Site","tentatives"], ascending=[True,True,False])
        if dfm.empty:
            st.success("Aucun utilisateur n’a essayé plusieurs fois dans la même heure sur ce périmètre.")
        else:
            BASE_CHARGE_URL = "https://elto.nidec-asi-online.com/Charge/detail?id="

            def _id_links(cell: str) -> str:
                if not isinstance(cell, str) or cell.strip() == "":
                    return ""
                ids = [x.strip() for x in cell.split(",") if x.strip()]
                return " · ".join(f'<a href="{BASE_CHARGE_URL}{iid}" target="_blank">{iid}</a>' for iid in ids)
            dfm["ID(s)"] = dfm["ID(s)"].astype(str).apply(_id_links)
            show_cols = ["Site","Heure","MAC", "Vehicle","tentatives","PDC(s)","1ère tentative","Dernière tentative","ID(s)"]
            soc_cols  = [c for c in ["SOC start min","SOC start max","SOC end min","SOC end max"] if c in dfm.columns]
            show_cols += soc_cols
            out = dfm[show_cols].copy()
            out.insert(0, "#", range(1, len(out)+1))
            st.markdown(out.to_html(index=False, escape=False, border=0), unsafe_allow_html=True)

with tab7:
    st.subheader("Transactions suspectes (<1 kWh)")
    suspicious = tables.get("suspicious_under_1kwh", pd.DataFrame())
    if suspicious.empty:
        st.success("Aucune transaction suspecte détectée (<1 kWh).")
    else:
        df_s = suspicious.copy()
        if "Site" in df_s.columns:
            df_s = df_s[df_s["Site"].isin(st.session_state.site_sel)]

        if "Datetime start" in df_s.columns:
            ds = pd.to_datetime(df_s["Datetime start"], errors="coerce")
            mask = ds.ge(pd.Timestamp(st.session_state.d1)) & ds.lt(pd.Timestamp(st.session_state.d2) + pd.Timedelta(days=1))
            df_s = df_s[mask].copy()

        if "ID" in df_s.columns:
            df_s["ID"] = df_s["ID"].astype(str).str.strip()
            df_s["Lien"] = BASE_CHARGE_URL + df_s["ID"]

        show_cols = [
            "ID","Lien","Site","PDC","MAC Address","Vehicle",
            "Datetime start","Datetime end","Energy (Kwh)","SOC Start","SOC End"
        ]
        show_cols = [c for c in show_cols if c in df_s.columns]

        if not df_s.empty:
            df_s = df_s.sort_values("Datetime start")
            df_s.insert(0, "#", range(1, len(df_s) + 1))
            if "Lien" in df_s.columns:
                st.dataframe(
                    df_s[["#"] + show_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Lien": st.column_config.LinkColumn(
                            label="Lien Vers ELTO",
                            help="Ouvrir la fiche de charge dans ELTO",
                            display_text="Lien Vers ELTO"
                        )
                    }
                )
            else:
                st.dataframe(df_s[["#"] + show_cols], use_container_width=True, hide_index=True)
        else:
            st.success("Aucune transaction suspecte sur ce périmètre.")
with tab8:
    def _map_moment(val: int) -> str:
        try:
            val = int(val)
        except:
            return "Unknown"
        if val == 0:
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
        return "Unknown"
    st.divider()
    st.subheader("🔍 Analyse des codes d’erreur")
    st.divider()
    err = sess_kpi[~sess_kpi["is_ok_filt"]].copy()
    if err.empty:
        st.info("Aucune erreur à afficher.")
    else:
        from analyses.kpi_cal import EVI_MOMENT, EVI_CODE, DS_PC

        evi_step = pd.to_numeric(err[EVI_MOMENT], errors="coerce").fillna(0).astype(int)
        evi_code = pd.to_numeric(err[EVI_CODE], errors="coerce").fillna(0).astype(int)
        ds_pc    = pd.to_numeric(err[DS_PC], errors="coerce").fillna(0).astype(int)
        sub_evi = err.loc[(ds_pc.eq(8192)) | ((ds_pc.eq(0)) & (evi_code.ne(0)))].copy()
        sub_evi["_step"]   = evi_step.loc[sub_evi.index]
        sub_evi["_moment"] = sub_evi["_step"].map(_map_moment)
        sub_evi["_code"]   = evi_code.loc[sub_evi.index]
        sub_evi["_site"]   = err[SITE_COL].loc[sub_evi.index]
        sub_ds = err.loc[ds_pc.ne(0) & ds_pc.ne(8192)].copy()
        sub_ds["_step"]   = evi_step.loc[sub_ds.index]
        sub_ds["_moment"] = sub_ds["_step"].map(_map_moment)
        sub_ds["_code"]   = ds_pc.loc[sub_ds.index]
        sub_ds["_site"]   = err[SITE_COL].loc[sub_ds.index]
        sub_evi["_type"] = "Erreur_EVI"
        sub_ds["_type"]  = "Erreur_DownStream"
        st.markdown("### Top 3 erreurs (EVI + Downstream)")
        all_err = pd.concat([sub_evi, sub_ds], ignore_index=True)
        all_err["_key"] = list(zip(all_err["_moment"], all_err["_step"], all_err["_code"], all_err["_type"]))

        tbl_all = (
            all_err.groupby("_key")
            .size()
            .reset_index(name="Occurrences")
            .sort_values("Occurrences", ascending=False)
        )

        total_err = tbl_all["Occurrences"].sum()
        tbl_all["%"] = (tbl_all["Occurrences"] / total_err * 100).round(2)

        top3_all = tbl_all.head(3)
        top3_keys = top3_all["_key"].tolist()

        col1, col2 = st.columns(2)

        with col1:
            df_top = pd.DataFrame(top3_keys, columns=["Moment", EVI_MOMENT, "Code", "Type d’erreur"])
            df_top["Occurrences"] = top3_all["Occurrences"].values
            df_top["%"] = top3_all["%"].values
            st.dataframe(df_top, use_container_width=True, hide_index=True)

        with col2:
            detail_top = (
                all_err[all_err["_key"].isin(top3_keys)]
                .groupby(["_moment", "_step", "_code", "_type", "_site"])
                .size()
                .reset_index(name="Occurrences")
                .sort_values(["_type", "_moment", "_step", "_code", "Occurrences"], ascending=[True]*5)
            )

            pivot_top = (
                detail_top.pivot(
                    index="_site",
                    columns=["_type", "_moment", "_step", "_code"],
                    values="Occurrences"
                )
                .fillna(0)
                .astype(int)
                .reset_index()
                .rename(columns={"_site": "Site"})
            )

            st.dataframe(pivot_top, use_container_width=True, hide_index=True)
        # EVI
        st.markdown("### Top 3 erreurs **EVI (Moment × Step × Code)**")
        if sub_evi.empty:
            st.info("Aucune erreur EVI trouvée.")
        else:
            tbl_evi = (
                sub_evi.groupby(["_moment", "_step", "_code"])
                    .size()
                    .reset_index(name="Occurrences")
                    .sort_values("Occurrences", ascending=False)
            )

            total_evi = tbl_evi["Occurrences"].sum()
            tbl_evi["%"] = (tbl_evi["Occurrences"] / total_evi * 100).round(2)

            top3_evi = tbl_evi.head(3)

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(
                    top3_evi.rename(columns={
                        "_moment": "Moment",
                        "_step": EVI_MOMENT,
                        "_code": "Code EVI"
                    }),
                    use_container_width=True, hide_index=True
                )

            with col2:
                top_evi_list = top3_evi.set_index(["_moment", "_step", "_code"]).index.tolist()
                detail_evi = (
                    sub_evi[sub_evi.set_index(["_moment", "_step", "_code"]).index.isin(top_evi_list)]
                    .groupby(["_moment", "_step", "_code", "_site"])
                    .size()
                    .reset_index(name="Occurrences")
                    .sort_values(["_moment", "_step", "_code", "Occurrences"], ascending=[True, True, True, False])
                )
                pivot = detail_evi.pivot(
                    index="_site", 
                    columns=["_moment","_step","_code"], 
                    values="Occurrences"
                ).fillna(0).astype(int).reset_index().rename(columns={"_site": "Site"})
                st.dataframe(pivot, use_container_width=True, hide_index=True)


        # Downstream
        st.markdown("### Top 3 erreurs **Downstream (Moment × Step × Code PC)**")
        if sub_ds.empty:
            st.info("Aucune erreur Downstream trouvée.")
        else:
            tbl_ds = (
                sub_ds.groupby(["_moment", "_step", "_code"])
                    .size()
                    .reset_index(name="Occurrences")
                    .sort_values("Occurrences", ascending=False)
            )

            total_ds = tbl_ds["Occurrences"].sum()
            tbl_ds["%"] = (tbl_ds["Occurrences"] / total_ds * 100).round(2)

            top3_ds = tbl_ds.head(3)

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(
                    top3_ds.rename(columns={
                        "_moment": "Moment",
                        "_step": EVI_MOMENT,
                        "_code": "Code PC"
                    }),
                    use_container_width=True, hide_index=True
                )

            with col2:
                top_ds_list = top3_ds.set_index(["_moment", "_step", "_code"]).index.tolist()
                detail_ds = (
                    sub_ds[sub_ds.set_index(["_moment", "_step", "_code"]).index.isin(top_ds_list)]
                    .groupby(["_moment", "_step", "_code", "_site"])
                    .size()
                    .reset_index(name="Occurrences")
                    .sort_values(["_moment", "_step", "_code", "Occurrences"], ascending=[True, True, True, False])
                )
                pivot = detail_ds.pivot(
                    index="_site", 
                    columns=["_moment","_step","_code"], 
                    values="Occurrences"
                ).fillna(0).astype(int).reset_index().rename(columns={"_site": "Site"})
                st.dataframe(pivot, use_container_width=True, hide_index=True)

    err = sess_kpi[~sess_kpi["is_ok_filt"]].copy()
    if err.empty:
        st.info("Aucune erreur à afficher.")
    else:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### EVI — Moment × Code")
            from analyses.kpi_cal import EVI_MOMENT, EVI_CODE, DS_PC
            if {EVI_MOMENT, EVI_CODE, DS_PC, SITE_COL}.issubset(err.columns):
                evi_step = pd.to_numeric(err[EVI_MOMENT], errors="coerce").fillna(0).astype(int)
                evi_code = pd.to_numeric(err[EVI_CODE],   errors="coerce").fillna(0).astype(int)
                ds_pc    = pd.to_numeric(err[DS_PC],      errors="coerce").fillna(0).astype(int)

                mask_evi = (ds_pc.eq(8192)) | ((ds_pc.eq(0)) & (evi_code.ne(0)))
                sub = err.loc[mask_evi].copy()

                sub["_step"] = evi_step.loc[sub.index]
                sub["_code"] = evi_code.loc[sub.index]
                sub["_site"] = err[SITE_COL].loc[sub.index]
                sub["_moment"] = sub["_step"].map(_map_moment)

                if sub.empty:
                    st.info("Aucune erreur correspondant à la logique 'Erreur_EVI' pour ce périmètre.")
                else:
                    tbl = (
                        sub.groupby(["_moment", "_step", "_code"])
                        .size()
                        .reset_index(name="Somme de Charge_NOK")
                        .sort_values("Somme de Charge_NOK", ascending=False)
                    )
                    tbl.rename(columns={
                        "_moment": "Moment",
                        "_step": EVI_MOMENT,
                        "_code": EVI_CODE,
                    }, inplace=True)
                    total = int(tbl["Somme de Charge_NOK"].sum())
                    total_row = pd.DataFrame([{
                        "Moment": "Total",
                        EVI_MOMENT: "",
                        EVI_CODE: "",
                        "Somme de Charge_NOK": total,
                    }])
                    out = pd.concat([tbl, total_row], ignore_index=True)
                    st.dataframe(out, use_container_width=True, hide_index=True)
                    st.markdown("#### EVI — Moment × Code × Site")
                    tbl_site = (
                        sub.groupby(["_site", "_moment", "_step", "_code"])
                        .size()
                        .reset_index(name="Somme de Charge_NOK")
                        .sort_values(["_site", "Somme de Charge_NOK"], ascending=[True, False])
                    )
                    tbl_site.rename(columns={
                        "_site": SITE_COL,
                        "_moment": "Moment",
                        "_step": EVI_MOMENT,
                        "_code": EVI_CODE,
                    }, inplace=True)
                    st.dataframe(tbl_site, use_container_width=True, hide_index=True)
            else:
                st.info("Colonnes EVI ou Downstream manquantes.")
        with c_right:
            st.markdown("#### Downstream — Moment × Code PC")
            if {EVI_MOMENT, DS_PC, SITE_COL}.issubset(err.columns):
                evi_step = pd.to_numeric(err[EVI_MOMENT], errors="coerce").fillna(0).astype(int)
                ds_pc    = pd.to_numeric(err[DS_PC],      errors="coerce").fillna(0).astype(int)

                sub = err.loc[ds_pc.ne(0) & ds_pc.ne(8192)].copy()
                sub["_step"] = evi_step.loc[sub.index]
                sub["_ds"]   = ds_pc.loc[sub.index]
                sub["_site"] = err[SITE_COL].loc[sub.index]
                def _map_moment(val: int) -> str:
                    try:
                        val = int(val)
                    except:
                        return "Unknown"
                    if val == 0:
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
                    return "Unknown"
                sub["_moment"] = sub["_step"].map(_map_moment)

                if sub.empty:
                    st.info("Aucun Downstream Code PC non nul pour ce périmètre.")
                else:
                    tbl = (
                        sub.groupby(["_moment", "_step", "_ds"])
                        .size()
                        .reset_index(name="Somme de Charge_NOK")
                        .sort_values("Somme de Charge_NOK", ascending=False)
                    )
                    tbl.rename(columns={
                        "_moment": "Moment",
                        "_step": EVI_MOMENT,
                        "_ds": DS_PC,
                    }, inplace=True)
                    total = int(tbl["Somme de Charge_NOK"].sum())
                    total_row = pd.DataFrame([{
                        "Moment": "Total",
                        EVI_MOMENT: "",
                        DS_PC: "",
                        "Somme de Charge_NOK": total,
                    }])
                    out = pd.concat([tbl, total_row], ignore_index=True)
                    st.dataframe(out, use_container_width=True, hide_index=True)
                    st.markdown("#### Downstream — Moment × Code PC × Site")
                    tbl_site = (
                        sub.groupby(["_site", "_moment", "_step", "_ds"])
                        .size()
                        .reset_index(name="Somme de Charge_NOK")
                        .sort_values(["_site", "Somme de Charge_NOK"], ascending=[True, False])
                    )
                    tbl_site.rename(columns={
                        "_site": SITE_COL,
                        "_moment": "Moment",
                        "_step": EVI_MOMENT,
                        "_ds": DS_PC,
                    }, inplace=True)
                    st.dataframe(tbl_site, use_container_width=True, hide_index=True)
            else:
                st.info("Colonnes Downstream manquantes.")

    st.divider()
    st.subheader("🔍 Analayse des moments d’erreur")
    st.divider()
    err = sess_kpi[~sess_kpi["is_ok_filt"]].copy()
    PHASE_MAP = {
        "Avant charge": {"Init", "Lock Connector", "CableCheck"},
        "Charge": {"Charge"},
        "Fin de charge": {"Fin de charge"},
        "Unknown": {"Unknown"}
    }
    by_site_f = (
        sess_kpi.groupby(SITE_COL, as_index=False)
                .agg(Total_Charges=("is_ok_filt", "count"),
                    Charges_OK=("is_ok_filt", "sum"))
    )

    by_site_f["Charges_NOK"] = by_site_f["Total_Charges"] - by_site_f["Charges_OK"]
    by_site_f["% Réussite"] = np.where(
        by_site_f["Total_Charges"] > 0,
        (by_site_f["Charges_OK"] / by_site_f["Total_Charges"] * 100).round(2),
        0.0
    )

    nok = sess_kpi.loc[~sess_kpi["is_ok_filt"]].copy()
    nok["moment"] = nok["moment"].fillna("Unknown")

    def map_phase(moment):
        for phase, moments in PHASE_MAP.items():
            if moment in moments:
                return phase
        return "Unknown"

    nok["Phase"] = nok["moment"].map(map_phase)

    err_by_phase = (
        nok.groupby([SITE_COL, "Phase"])
            .size()
            .unstack("Phase", fill_value=0)
            .reset_index()
    )

    err_by_phase.rename(columns={
        "Avant charge": "Nb Avant charge",
        "Charge": "Nb Charge",
        "Fin de charge": "Nb Fin de charge",
        "Unknown": "Nb Unknown"
    }, inplace=True)

    df_final = by_site_f.merge(err_by_phase, on=SITE_COL, how="left").fillna(0)

    df_final["% Erreurs"] = np.where(
        df_final["Total_Charges"] > 0,
        ((df_final.get("Nb Avant charge", 0) +
        df_final.get("Nb Charge", 0) +
        df_final.get("Nb Fin de charge", 0) +
        df_final.get("Nb Unknown", 0)) / df_final["Total_Charges"] * 100).round(2),
        0.0
    )
    expected_cols = [
        "Nb Avant charge",
        "Nb Charge",
        "Nb Fin de charge",
        "Nb Unknown"
    ]
    for col in expected_cols:
        if col not in df_final.columns:
            df_final[col] = 0

    df_final["% Erreurs"] = np.where(
        df_final["Total_Charges"] > 0,
        ((df_final["Nb Avant charge"] + df_final["Nb Charge"] +
        df_final["Nb Fin de charge"] + df_final["Nb Unknown"]) / df_final["Total_Charges"] * 100).round(2),
        0.0
    )
    final_cols = [
        SITE_COL,
        "Total_Charges",
        "Charges_OK",
        "Charges_NOK",
        "% Réussite",
        "% Erreurs",
        "Nb Avant charge",
        "Nb Charge",
        "Nb Fin de charge",
        "Nb Unknown"
    ]
    df_final = df_final[final_cols]
    st.dataframe(df_final, use_container_width=True, hide_index=True)

    err = sess_kpi[~sess_kpi["is_ok_filt"]].copy()
    err_nonempty = err[err["type_erreur"].notna() & (err["type_erreur"] != "")]
    if not err_nonempty.empty:
            counts_t = (
                err_nonempty.groupby("type_erreur")
                .size()
                .reset_index(name="Nb")
                .sort_values("Nb", ascending=False)
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.pie(
                    counts_t,
                    names="type_erreur",
                    values="Nb",
                    title="Types d’erreurs (%)",
                    hole=0.3,
                )
                fig.update_traces(
                    textinfo="label+percent",
                    pull=[0.05] * len(counts_t)
                )
                plot(fig, "tab1_types_pie")

            with col2:
                total_row = pd.DataFrame({
                    "type_erreur": ["Total"],
                    "Nb": [counts_t["Nb"].sum()]
                })
                full_table = pd.concat([counts_t, total_row], ignore_index=True)
                st.dataframe(full_table, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune erreur à afficher pour ce périmètre.")

    err = sess_kpi[~sess_kpi["is_ok_filt"]].copy()
    if not err.empty and "moment" in err.columns:
        counts_moment = (
            err.groupby("moment")
            .size()
            .reindex(MOMENT_ORDER, fill_value=0)
            .reset_index(name="Somme de Charge_NOK")
        )
        counts_moment = counts_moment[counts_moment["Somme de Charge_NOK"] > 0]
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.pie(
                counts_moment,
                names="moment",
                values="Somme de Charge_NOK",
                title="Moment d'erreurs (EVI et DownStream) (%)",
                hole=0.25,
                color="moment",
                color_discrete_map=MOMENT_PALETTE,
                category_orders={"moment": MOMENT_ORDER},
            )
            fig.update_traces(textinfo="label+percent")
            plot(fig, "pie_erreurs_par_moment")
        with col2:
            total_row = pd.DataFrame({
                "moment": ["Total"],
                "Somme de Charge_NOK": [counts_moment["Somme de Charge_NOK"].sum()]
            })
            full_table = pd.concat([counts_moment, total_row], ignore_index=True)
            st.dataframe(full_table, use_container_width=True)
        # Moments (avancé)
        if "moment_avancee" in err.columns:
            counts_av = (
                err.groupby("moment_avancee")
                .size()
                .reset_index(name="Somme de Charge_NOK")
                .sort_values("Somme de Charge_NOK", ascending=False)
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                fig_av = px.pie(
                    counts_av,
                    names="moment_avancee",
                    values="Somme de Charge_NOK",
                    title="Moments d'erreur (EVI et DownStream) (Avancé) (%)",
                    hole=0.25,
                    color="moment_avancee",
                    color_discrete_map={
                        "Avant charge": "#FF7F0E",
                        "Charge": "#1F77B4",
                        "Après charge": "#2CA02C",
                        "Unknown": "#7F7F7F"
                    }
                )
                fig_av.update_traces(textinfo="label+percent")
                plot(fig_av, "pie_moment_agrege")
            with col2:
                total_row = pd.DataFrame({
                    "moment_avancee": ["Total"],
                    "Somme de Charge_NOK": [counts_av["Somme de Charge_NOK"].sum()]
                })
                full_table = pd.concat([counts_av, total_row], ignore_index=True)
                st.dataframe(full_table, use_container_width=True)
    else:
        st.info("Aucune erreur à afficher pour ce périmètre.")
    st.divider()
    # Répartition EVI par moment
    err_evi = err[err["type_erreur"] == "Erreur_EVI"].copy()
    if not err_evi.empty and "moment" in err_evi.columns:
        counts_moment = (
            err_evi.groupby("moment")
                .size()
                .reindex(MOMENT_ORDER, fill_value=0)
                .reset_index(name="Nb")
        )
        total_evi_err = counts_moment["Nb"].sum()
        if total_evi_err > 0:
            counts_moment["%"] = (counts_moment["Nb"] / total_evi_err * 100).round(2)
            fig = px.bar(
                counts_moment,
                x="moment",
                y="%",
                text="%",
                color="moment",
                category_orders={"moment": MOMENT_ORDER},
                color_discrete_map=MOMENT_PALETTE,
                title="Répartition des erreurs EVI par moment (%)"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            plot(fig, "tab1_moment_pct")
        else:
            st.info("Aucune erreur EVI pour les filtres choisis.")
    else:
        st.info("Aucune erreur EVI pour les filtres choisis.")

    # PIE EVI uniquement
    if not err_evi.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Pie EVI - par moment
            if "moment" in err_evi.columns:
                counts_m = (
                    err_evi.groupby("moment")
                        .size()
                        .reindex(MOMENT_ORDER, fill_value=0)
                        .reset_index(name="Nb")
                )
                counts_m = counts_m[counts_m["Nb"] > 0]
                if not counts_m.empty:
                    fig = px.pie(
                        counts_m,
                        names="moment",
                        values="Nb",
                        title="Moments d’erreur EVI (%)",
                        hole=0.25,
                        color="moment",
                        color_discrete_map=MOMENT_PALETTE,
                        category_orders={"moment": MOMENT_ORDER},
                    )
                    fig.update_traces(textinfo="label+percent")
                    plot(fig, "tab1_moments_pie")

                    total_row = pd.DataFrame({
                        "moment": ["Total"],
                        "Nb": [counts_m["Nb"].sum()]
                    })
                    full_table = pd.concat([counts_m, total_row], ignore_index=True)
                    st.dataframe(full_table, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune erreur EVI (moment)")
        with col2:
            # Pie EVI - par moment_avancee
            if "moment_avancee" in err_evi.columns:
                counts_ma = (
                    err_evi.groupby("moment_avancee")
                        .size()
                        .reset_index(name="Nb")
                        .sort_values("Nb", ascending=False)
                )
                if not counts_ma.empty:
                    fig = px.pie(
                        counts_ma,
                        names="moment_avancee",
                        values="Nb",
                        title="Moments d’erreur EVI (Avancé) (%) ",
                        hole=0.25,
                        color="moment_avancee",
                        color_discrete_map={
                            "Avant charge": "#FF7F0E",
                            "Charge": "#1F77B4",
                            "Fin de charge": "#2CA02C",
                            "Unknown": "#7F7F7F"
                        }
                    )
                    fig.update_traces(textinfo="label+percent")
                    plot(fig, "tab1_moments_pie_avancee")

                    total_row = pd.DataFrame({
                        "moment_avancee": ["Total"],
                        "Nb": [counts_ma["Nb"].sum()]
                    })
                    full_table = pd.concat([counts_ma, total_row], ignore_index=True)
                    st.dataframe(full_table, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune erreur EVI (avancé)")
    else:
        st.info("Aucune erreur EVI pour les filtres choisis.")
    st.divider()

    # Répartition Downstream par moment
    err_ds = err[err["type_erreur"] == "Erreur_DownStream"].copy()
    if not err_ds.empty and "moment" in err_ds.columns:
        counts_moment_ds = (
            err_ds.groupby("moment")
                .size()
                .reindex(MOMENT_ORDER, fill_value=0)
                .reset_index(name="Nb")
        )
        total_ds_err = counts_moment_ds["Nb"].sum()
        if total_ds_err > 0:
            counts_moment_ds["%"] = (counts_moment_ds["Nb"] / total_ds_err * 100).round(2)
            fig = px.bar(
                counts_moment_ds,
                x="moment",
                y="%",
                text="%",
                color="moment",
                category_orders={"moment": MOMENT_ORDER},
                color_discrete_map=MOMENT_PALETTE,
                title="Répartition des erreurs DownStream par moment (%)"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            plot(fig, "tab1_ds_moment_pct")
        else:
            st.info("Aucune erreur DownStream pour les filtres choisis.")
    else:
        st.info("Aucune erreur DownStream pour les filtres choisis.")

    # PIE DownStream
    if not err_ds.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Pie DownStream - par moment
            if "moment" in err_ds.columns:
                counts_m_ds = (
                    err_ds.groupby("moment")
                        .size()
                        .reindex(MOMENT_ORDER, fill_value=0)
                        .reset_index(name="Nb")
                )
                counts_m_ds = counts_m_ds[counts_m_ds["Nb"] > 0]
                if not counts_m_ds.empty:
                    fig = px.pie(
                        counts_m_ds,
                        names="moment",
                        values="Nb",
                        title="Moments d’erreur DownStream (%)",
                        hole=0.25,
                        color="moment",
                        color_discrete_map=MOMENT_PALETTE,
                        category_orders={"moment": MOMENT_ORDER},
                    )
                    fig.update_traces(textinfo="label+percent")
                    plot(fig, "tab1_ds_moments_pie")

                    total_row = pd.DataFrame({
                        "moment": ["Total"],
                        "Nb": [counts_m_ds["Nb"].sum()]
                    })
                    full_table = pd.concat([counts_m_ds, total_row], ignore_index=True)
                    st.dataframe(full_table, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune erreur DownStream (moment)")
        with col2:
            # Pie DownStream - par moment_avancee
            if "moment_avancee" in err_ds.columns:
                counts_ma_ds = (
                    err_ds.groupby("moment_avancee")
                        .size()
                        .reset_index(name="Nb")
                        .sort_values("Nb", ascending=False)
                )
                if not counts_ma_ds.empty:
                    fig = px.pie(
                        counts_ma_ds,
                        names="moment_avancee",
                        values="Nb",
                        title="Moments d’erreur DownStream (Avancé) (%) ",
                        hole=0.25,
                        color="moment_avancee",
                        color_discrete_map={
                            "Avant charge": "#FF7F0E",
                            "Charge": "#1F77B4",
                            "Fin de charge": "#2CA02C",
                            "Unknown": "#7F7F7F"
                        }
                    )
                    fig.update_traces(textinfo="label+percent")
                    plot(fig, "tab1_ds_moments_pie_avancee")

                    total_row = pd.DataFrame({
                        "moment_avancee": ["Total"],
                        "Nb": [counts_ma_ds["Nb"].sum()]
                    })
                    full_table = pd.concat([counts_ma_ds, total_row], ignore_index=True)
                    st.dataframe(full_table, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune erreur DownStream (avancé)")
    else:
        st.info("Aucune erreur DownStream pour les filtres choisis.")


with tab9:
    st.markdown("### 🔍 Analyse Erreur Spécifique")
    with st.expander("🔍 Filtrer par code", expanded=False):
        code_raw_tab = st.text_input(
            "N° d’erreur / Code PC",
            value="",
            placeholder="ex : 73, 84, 90",
            key="code_filter_values_tab",
            help="Saisissez un ou plusieurs codes entiers séparés par virgules, espaces ou ;"
        )
        code_type_tab = st.selectbox(
            "Type du code à filtrer",
            options=["Tous", "Erreur_EVI", "Erreur_DownStream"],
            index=0,
            key="code_filter_type_tab"
        )
    def _mask_code_local(df: pd.DataFrame, code_raw: str, code_type: str):
        import re
        if not code_raw.strip():
            return pd.Series(False, index=df.index)

        parts = re.split(r"[,\s;]+", code_raw.strip())
        try:
            codes = [int(p) for p in parts if p.strip() != ""]
        except Exception:
            codes = []

        if not codes:
            return pd.Series(False, index=df.index)

        col_evi = "EVI Error Code" if "EVI Error Code" in df.columns else None
        col_ds  = "Downstream Code PC" if "Downstream Code PC" in df.columns else None

        masks = []
        if code_type == "Erreur_EVI" and col_evi:
            masks.append(pd.to_numeric(df[col_evi], errors="coerce").isin(codes))
        elif code_type == "Erreur_DownStream" and col_ds:
            masks.append(pd.to_numeric(df[col_ds], errors="coerce").isin(codes))
        elif code_type == "Tous":
            if col_evi:
                masks.append(pd.to_numeric(df[col_evi], errors="coerce").isin(codes))
            if col_ds:
                masks.append(pd.to_numeric(df[col_ds], errors="coerce").isin(codes))

        if not masks:
            return pd.Series(False, index=df.index)

        if len(masks) == 1:
            return masks[0]
        return masks[0] | masks[1]
        
    if not code_raw_tab.strip():
        st.info("⏳ Saisissez un ou plusieurs codes dans le filtre")
    else:
        import re
        raw_parts = re.split(r"[,\s;]+", code_raw_tab.strip())
        try:
            selected_codes = [int(p) for p in raw_parts if p.strip() != ""]
        except Exception:
            selected_codes = []

        if not selected_codes:
            st.warning("Aucun code valide reconnu.")
        else:
            def code_match(key):
                return key[2] in selected_codes and (code_type_tab == "Tous" or key[3] == code_type_tab)

            matched_rows = tbl_all[tbl_all["_key"].apply(code_match)]

            if matched_rows.empty:
                st.info("Aucune donnée pour les codes spécifiés.")
            else:
                total_pct = matched_rows["%"].sum().round(2)
                code_list_str = ", ".join(str(c) for c in selected_codes)
                st.markdown(
                    f"**Code {code_list_str} → {total_pct}% des erreurs totales**"
                )
        BASE_CHARGE_URL = "https://elto.nidec-asi-online.com/Charge/detail?id="
        if not isinstance(sess, pd.DataFrame) or sess.empty:
            st.info("Aucune donnée disponible.")
        else:
            df_src = sess.copy()
            if "is_ok" not in df_src.columns:
                st.warning("Colonne 'is_ok' absente dans les sessions.")
            else:
                mask_type   = True
                mask_moment = True
                if "type_erreur" in df_src.columns and st.session_state.get("type_sel"):
                    mask_type = df_src["type_erreur"].isin(st.session_state.type_sel)
                if {"type_erreur", "moment"}.issubset(df_src.columns) and st.session_state.get("moment_sel"):
                    mask_moment = df_src["moment"].isin(st.session_state.moment_sel)

                df_src_f = df_src[mask_type & mask_moment].copy()
                mask_code_tab = _mask_code_local(df_src_f, code_raw_tab, code_type_tab)
                df_src_f = df_src_f[mask_code_tab].copy()
                if code_type_tab == "Erreur_EVI":
                    df_src_f = df_src_f[df_src_f["type_erreur"] == "Erreur_EVI"]
                elif code_type_tab == "Erreur_DownStream":
                    df_src_f = df_src_f[df_src_f["type_erreur"] == "Erreur_DownStream"]
                err_sum = df_src_f.loc[~df_src_f["is_ok"]].copy()

                if err_sum.empty:
                    st.info("Aucune charge en erreur pour le périmètre/filtre sélectionné.")
                else:
                    for c in ("Datetime start", "Datetime end"):
                        if c in err_sum.columns:
                            err_sum[c] = pd.to_datetime(err_sum[c], errors="coerce")
                    if "Energy (Kwh)" in err_sum.columns:
                        err_sum["Energy (Kwh)"] = pd.to_numeric(err_sum["Energy (Kwh)"], errors="coerce")

                    for c in ("SOC Start", "SOC End"):
                        if c in err_sum.columns:
                            err_sum[c] = pd.to_numeric(err_sum[c], errors="coerce")

                    if "MAC Address" in err_sum.columns:
                        err_sum["MAC Address"] = err_sum["MAC Address"].apply(_fmt_mac)

                    def _etiquette(row):
                        t = str(row.get("type_erreur", "") or "")
                        m = str(row.get("moment", "") or "")
                        return f"{t} — {m}" if m else t
                    err_sum["Erreur"] = err_sum.apply(_etiquette, axis=1)

                    def _soc_evo(row):
                        s0 = row.get("SOC Start", pd.NA)
                        s1 = row.get("SOC End", pd.NA)
                        if pd.notna(s0) and pd.notna(s1):
                            try:
                                return f"{int(round(s0))}% → {int(round(s1))}%"
                            except Exception:
                                return ""
                        return ""
                    err_sum["Évolution SOC"] = err_sum.apply(_soc_evo, axis=1)

                    if "ID" not in err_sum.columns:
                        st.warning("Colonne 'ID' absente : les liens ELTO ne seront pas affichés.")
                        err_sum["ELTO"] = ""
                    else:
                        err_sum["ELTO"] = BASE_CHARGE_URL + err_sum["ID"].astype(str).str.strip()

                    if "Vehicle" not in err_sum.columns and "ID" in err_sum.columns:
                        if "charges_mac" in locals(): 
                            veh_map = charges_mac[["ID", "Vehicle"]].drop_duplicates("ID", keep="last")
                            err_sum = err_sum.merge(veh_map, on="ID", how="left")
                    cols_aff = ["Site", "Datetime start", "Datetime end",
                                "Energy (Kwh)", "MAC Address",
                                "Vehicle", "Erreur", "Évolution SOC", "ELTO"]
                    cols_aff = [c for c in cols_aff if c in err_sum.columns]

                    out = err_sum[cols_aff].copy()
                    if "Datetime start" in out.columns:
                        out = out.sort_values("Datetime start", ascending=False)

                    out.insert(0, "#", range(1, len(out) + 1))

                    st.data_editor(
                        out,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "ELTO": st.column_config.LinkColumn(
                                "Lien ELTO",
                                help="Ouvrir la session dans ELTO",
                                display_text="🔗 Ouvrir"
                            ),
                            "Datetime start": st.column_config.DatetimeColumn("Start time", format="YYYY-MM-DD HH:mm:ss"),
                            "Datetime end":   st.column_config.DatetimeColumn("End time",   format="YYYY-MM-DD HH:mm:ss"),
                            "Energy (Kwh)":   st.column_config.NumberColumn("Energy (kWh)", format="%.3f"),
                            "MAC Address":    st.column_config.TextColumn("MacAdress"),
                            "Erreur":         st.column_config.TextColumn("Error etiquette"),
                            "Évolution SOC":  st.column_config.TextColumn("Evolution SOC"),
                        }
                    )
    st.divider()
    # Histogramme des occurrences par véhicule
    if not code_raw_tab.strip():
        st.info("⏳ Saisissez un ou plusieurs codes dans le filtre")
    else:
        if "Vehicle" in err_sum.columns:
            occ_vehicle = (
                err_sum.groupby("Vehicle")
                .size()
                .reset_index(name="Occurrences")
            )

            total_charges = (
                dfv.groupby("Vehicle")
                .size()
                .reset_index(name="Total Charges")
            )

            occ_vehicle = occ_vehicle.merge(
                total_charges,
                on="Vehicle",
                how="left"
            )

            occ_vehicle["Vehicle Label"] = occ_vehicle.apply(
                lambda row: f"{row['Vehicle']} ({int(row['Total Charges'])})"
                if pd.notna(row["Total Charges"]) else f"{row['Vehicle']} (0)",
                axis=1
            )

            occ_vehicle = occ_vehicle.sort_values("Occurrences", ascending=True)

            fig_vehicle = px.bar(
                occ_vehicle,
                x="Occurrences",
                y="Vehicle Label",
                orientation="h",
                title="Occurrences par véhicule (avec total charges)",
                labels={"Occurrences": "Nb d'occurrences", "Vehicle Label": "Véhicule"}
            )
            st.plotly_chart(fig_vehicle, use_container_width=True)
        else:
            st.info("Colonne 'Vehicle' absente pour compter les occurrences par véhicule.")

    st.divider()

    # Histogramme temporel par mois
    if not code_raw_tab.strip():
        st.info("⏳ Saisissez un ou plusieurs codes dans le filtre")
    else:
        if "Datetime start" in err_sum.columns and "Site" in err_sum.columns:
            err_sum["Datetime start"] = pd.to_datetime(err_sum["Datetime start"], errors="coerce")
            err_sum["month"] = err_sum["Datetime start"].dt.to_period("M").astype(str)
            fig_month = px.histogram(
                err_sum,
                x="month",
                color="Site",
                barmode="group",
                labels={"month": "Mois", "count": "Occurrences"},
                title="Histogramme mensuel des occurrences par site"
            )
            fig_month.update_layout(
                plot_bgcolor="#f9f9f9",
                bargap=0.2,
                xaxis=dict(type="category", tickangle=-45),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
            )
            st.plotly_chart(fig_month, use_container_width=True)
        else:
            st.info("Colonnes 'Datetime start' ou 'Site' manquantes pour tracer l’histogramme.")
    # Nombre d’occurrences par site et PDC
    if not code_raw_tab.strip():
        st.info("⏳ Saisissez un ou plusieurs codes dans le filtre")
    else:
        if {"Site", "PDC"}.issubset(err_sum.columns):
            occ = (
                err_sum.groupby(["Site", "PDC"])
                    .size()
                    .reset_index(name="Occurrences")
                    .sort_values("Occurrences", ascending=False)
            )
            st.markdown("### Nombre d’occurrences par site et PDC")
            st.dataframe(occ, use_container_width=True, hide_index=True)
        else:
            st.info("Colonnes 'Site' ou 'PDC' absentes pour compter les occurrences.")
    st.divider()
    from datetime import date, timedelta

    if not code_raw_tab.strip():
        st.info("⏳ Saisissez un ou plusieurs codes dans le filtre")
    else:
        st.markdown("### Zoom site/mois/jour")

        if err_sum.empty or not {"Site", "Datetime start", "PDC"}.issubset(err_sum.columns):
            st.info("Données insuffisantes.")
        else:
            err_sum["Datetime start"] = pd.to_datetime(err_sum["Datetime start"], errors="coerce")
            err_sum = err_sum.dropna(subset=["Datetime start"])

            site_focus = st.selectbox("Site", sorted(err_sum["Site"].dropna().unique()), key="site_focus_tab9")
            df_site = err_sum[err_sum["Site"] == site_focus].copy()
            df_site["month"] = df_site["Datetime start"].dt.to_period("M").astype(str)

            months = sorted(df_site["month"].unique().tolist())
            if months:
                month_focus = st.selectbox("Mois", months, key="month_focus_tab9")
                df_month = df_site[df_site["month"] == month_focus].copy()
                df_month["day"] = df_month["Datetime start"].dt.date
                pdc_unique = sorted(df_month["PDC"].dropna().unique())
                palette = px.colors.qualitative.Plotly
                if not pdc_unique:
                    st.info("Aucun PDC disponible pour le filtre sélectionné.")
                else:
                    color_map = {pdc: palette[i % len(palette)] for i, pdc in enumerate(pdc_unique)}
                    year, month = map(int, month_focus.split("-"))
                    first_day = date(year, month, 1)
                    last_day = (first_day.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                    all_days = pd.date_range(first_day, last_day)

                    df_day_count = df_month.groupby(["day", "PDC"]).size().reset_index(name="Occurrences")
                    df_pivot = df_day_count.pivot(index="day", columns="PDC", values="Occurrences").reindex(all_days, fill_value=0)
                    df_full = df_pivot.stack().reset_index()
                    df_full.columns = ["day", "PDC", "Occurrences"]
                    df_full["day"] = pd.to_datetime(df_full["day"])

                    pdc_unique = sorted(df_full["PDC"].dropna().unique())

                    fig_jours = go.Figure()
                    for i, pdc in enumerate(pdc_unique):
                        df_sub = df_full[df_full["PDC"] == pdc]
                        df_sub = df_sub.sort_values("day")
                        fig_jours.add_trace(go.Bar(
                            x=df_sub["day"],
                            y=df_sub["Occurrences"],
                            name=str(pdc),
                            marker_color=color_map.get(pdc, palette[i % len(palette)])
                        ))

                    fig_jours.update_layout(
                        barmode='group',
                        title=f"{site_focus} — {month_focus} : Occurrences par jour et PDC",
                        xaxis=dict(title="Jour", tickangle=-45, type="date", tickformat="%Y-%m-%d"),
                        yaxis=dict(title="Occurrences"),
                        legend_title_text="PDC",
                        bargap=0.15,
                        showlegend=True
                    )

                    st.plotly_chart(fig_jours, use_container_width=True)
                    jours = sorted(df_month["day"].unique())
                    if jours:
                        jour_focus = st.selectbox("Jour", options=jours, key="day_focus_tab9")
                        df_jour = df_month[df_month["day"] == jour_focus].copy()
                        df_jour["hour"] = df_jour["Datetime start"].dt.hour

                        all_hours = list(range(24))
                        df_hour_count = df_jour.groupby(["hour", "PDC"]).size().reset_index(name="Occurrences")
                        df_pivot_hour = df_hour_count.pivot(index="hour", columns="PDC", values="Occurrences").reindex(all_hours, fill_value=0)
                        df_full_hour = df_pivot_hour.stack().reset_index()
                        df_full_hour.columns = ["hour", "PDC", "Occurrences"]

                        pdc_unique_hour = sorted(df_full_hour["PDC"].dropna().unique())

                        fig_heures = go.Figure()
                        for i, pdc in enumerate(pdc_unique_hour):
                            df_sub = df_full_hour[df_full_hour["PDC"] == pdc].sort_values("hour")
                            fig_heures.add_trace(go.Bar(
                                x=df_sub["hour"],
                                y=df_sub["Occurrences"],
                                name=str(pdc),
                                marker_color=color_map.get(pdc, palette[i % len(palette)])
                            ))

                        fig_heures.update_layout(
                            barmode='group',
                            title=f"{site_focus} — {jour_focus} : Occurrences par heure et PDC",
                            xaxis=dict(title="Heure", dtick=1, tickmode="linear"),
                            yaxis=dict(title="Occurrences"),
                            legend_title_text="PDC",
                            bargap=0.15,
                            showlegend=True
                        )
                        st.plotly_chart(fig_heures, use_container_width=True)

with tab10:
    st.markdown("### ⚠️ Alertes : erreurs récurrentes par PDC")
    errors_only = sess_kpi[~sess_kpi["is_ok_filt"]].copy()
    if errors_only.empty:
        st.info("Aucune erreur dans le périmètre.")
    else:
        errors_only["Datetime start"] = pd.to_datetime(errors_only["Datetime start"], errors="coerce")
        errors_only = errors_only.dropna(subset=["Datetime start", "PDC", "type_erreur"])
        errors_only = errors_only.sort_values(["PDC", "type_erreur", "Datetime start"]).reset_index()

        alert_rows = []

        for (pdc, err_type), group in errors_only.groupby(["PDC", "type_erreur"]):
            times = group["Datetime start"]
            idxs = group["index"]

            for i in range(len(times)):
                t0 = times.iloc[i]
                t1 = t0 + pd.Timedelta(hours=12)
                window = times[(times >= t0) & (times <= t1)]
                if len(window) >= 3:
                    idx3 = idxs.iloc[i] 
                    row = sess_kpi.loc[idx3]

                    alert_rows.append({
                        "Site": row.get(SITE_COL, "—"),
                        "PDC": pdc,
                        "Type d'erreur": err_type,
                        "Détection": t0,
                        "Occurrences sur 12h": len(window),
                        "Moment": row.get("moment", "—"),
                        "EVI Code": row.get("EVI Error Code", "—"),
                        "Downstream Code PC": row.get("Downstream Code PC", "—")
                    })
                    break  

        if not alert_rows:
            st.success("✅ Aucune alerte détectée.")
        else:
            df_alertes = pd.DataFrame(alert_rows).sort_values("Détection", ascending=False)
            st.dataframe(df_alertes, use_container_width=True)