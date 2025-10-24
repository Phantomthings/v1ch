import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabs.context import get_context

TAB_CODE = """
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

