import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabs.context import get_context

TAB_CODE = """
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

