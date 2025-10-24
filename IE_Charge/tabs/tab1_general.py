import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from textwrap import dedent

from tabs.context import get_context

TAB_CODE = dedent("""
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
""")

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

