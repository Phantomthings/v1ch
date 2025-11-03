import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabs.context import get_context

TAB_CODE = """
st.markdown("### ⚠️ Alertes : erreurs récurrentes par PDC")

df_alertes = tables.get("alertes", pd.DataFrame()).copy()

if df_alertes.empty:
    st.info("Aucune erreur dans le périmètre.")
else:
    df_alertes = df_alertes.rename(columns={
        "type_erreur": "Type d'erreur",
        "detection": "Détection",
        "occurrences_12h": "Occurrences sur 12h",
        "moment": "Moment",
        "evi_code": "EVI Code",
        "downstream_code_pc": "Downstream Code PC",
    })

    if "Détection" in df_alertes.columns:
        df_alertes["Détection"] = pd.to_datetime(df_alertes["Détection"], errors="coerce")

        if "d1" in locals() and "d2" in locals():
            start_dt = pd.to_datetime(locals().get("d1"))
            end_dt = pd.to_datetime(locals().get("d2")) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            if pd.notna(start_dt) and pd.notna(end_dt):
                df_alertes = df_alertes[df_alertes["Détection"].between(start_dt, end_dt)]

    active_sites = locals().get("site_sel")
    if active_sites and "Site" in df_alertes.columns:
        df_alertes = df_alertes[df_alertes["Site"].isin(active_sites)]

    active_types = locals().get("type_sel")
    if active_types and "Type d'erreur" in df_alertes.columns:
        df_alertes = df_alertes[df_alertes["Type d'erreur"].isin(active_types)]

    active_moments = locals().get("moment_sel")
    if active_moments and "Moment" in df_alertes.columns:
        df_alertes = df_alertes[df_alertes["Moment"].isin(active_moments)]

    display_cols = [
        col for col in [
            "Site",
            "PDC",
            "Type d'erreur",
            "Détection",
            "Occurrences sur 12h",
            "Moment",
            "EVI Code",
            "Downstream Code PC",
        ]
        if col in df_alertes.columns
    ]

    if "Détection" in df_alertes.columns:
        df_alertes = df_alertes.dropna(subset=["Détection"]).sort_values("Détection", ascending=False)

    if df_alertes.empty:
        st.success("✅ Aucune alerte détectée.")
    else:
        st.dataframe(
            df_alertes[display_cols],
            use_container_width=True,
            hide_index=True,
        )
"""

def render():
    ctx = get_context()
    globals_dict = {"np": np, "pd": pd, "px": px, "go": go, "st": st}
    local_vars = dict(ctx.__dict__)
    local_vars.setdefault('plot', getattr(ctx, 'plot', None))
    local_vars.setdefault('hide_zero_labels', getattr(ctx, 'hide_zero_labels', None))
    local_vars.setdefault('with_charge_link', getattr(ctx, 'with_charge_link', None))
    local_vars.setdefault('evi_counts_pivot', getattr(ctx, 'evi_counts_pivot', None))
    # remove None entries
    local_vars = {k: v for k, v in local_vars.items() if v is not None}
    exec(TAB_CODE, globals_dict, local_vars)

