import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Optional

from tabs.context import get_context

TAB_CODE = """
st.markdown("### 🔍 Analyse Erreur Spécifique")
with st.expander("🔍 Filtrer par code", expanded=False):
    code_raw_tab = st.text_input(
        "N° d'erreur / Code PC",
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

# Initialize selected_codes outside the conditional block
selected_codes = []

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
        def code_match(key, codes=selected_codes, type_filter=code_type_tab):
            return key[2] in codes and (type_filter == "Tous" or key[3] == type_filter)

        if "tbl_all" not in locals() or not isinstance(tbl_all, pd.DataFrame):
            tbl_all = pd.DataFrame()
            if "sess_kpi" in locals():
                from analyses.kpi_cal import EVI_MOMENT, EVI_CODE, DS_PC

                err_ctx = sess_kpi.loc[~sess_kpi["is_ok_filt"]].copy()
                if not err_ctx.empty:
                    evi_step = pd.to_numeric(err_ctx[EVI_MOMENT], errors="coerce").fillna(0).astype(int)
                    evi_code = pd.to_numeric(err_ctx[EVI_CODE], errors="coerce").fillna(0).astype(int)
                    ds_pc    = pd.to_numeric(err_ctx[DS_PC], errors="coerce").fillna(0).astype(int)

                    def _local_map(step_val):
                        try:
                            step_val = int(step_val)
                        except Exception:
                            return "Unknown"
                        if step_val == 0:
                            return "Fin de charge"
                        if 1 <= step_val <= 2:
                            return "Init"
                        if 4 <= step_val <= 6:
                            return "Lock Connector"
                        if step_val == 7:
                            return "CableCheck"
                        if step_val == 8:
                            return "Charge"
                        if step_val > 8:
                            return "Fin de charge"
                        return "Unknown"

                    sub_evi = err_ctx.loc[(ds_pc.eq(8192)) | ((ds_pc.eq(0)) & (evi_code.ne(0)))].copy()
                    sub_evi["_step"]   = evi_step.loc[sub_evi.index]
                    sub_evi["_moment"] = sub_evi["_step"].map(_local_map)
                    sub_evi["_code"]   = evi_code.loc[sub_evi.index]
                    sub_evi["_type"]   = "Erreur_EVI"

                    sub_ds = err_ctx.loc[ds_pc.ne(0) & ds_pc.ne(8192)].copy()
                    sub_ds["_step"]   = evi_step.loc[sub_ds.index]
                    sub_ds["_moment"] = sub_ds["_step"].map(_local_map)
                    sub_ds["_code"]   = ds_pc.loc[sub_ds.index]
                    sub_ds["_type"]   = "Erreur_DownStream"

                    all_err = pd.concat([sub_evi, sub_ds], ignore_index=True)
                    if not all_err.empty:
                        tbl_all = (
                            all_err.assign(
                                _key=lambda d: list(zip(
                                    d.get("_moment", [""] * len(d)),
                                    d.get("_step",   [0]  * len(d)),
                                    d.get("_code",   [0]  * len(d)),
                                    d.get("_type",   [""] * len(d)),
                                ))
                            )
                            .groupby("_key")
                            .size()
                            .reset_index(name="Occurrences")
                            .sort_values("Occurrences", ascending=False)
                        )
                        total_err = tbl_all["Occurrences"].sum()
                        if total_err:
                            tbl_all["%"] = (tbl_all["Occurrences"] / total_err * 100).round(2)
                        else:
                            tbl_all["%"] = 0.0

        matched_rows = tbl_all[tbl_all.get("_key", pd.Series(dtype=object)).apply(code_match)] if not tbl_all.empty else pd.DataFrame()

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
            def _prepare_charge_details(df: pd.DataFrame, lookup: Optional[pd.DataFrame] = None) -> pd.DataFrame:
                work = df.copy()

                if work.empty:
                    return work

                for c in ("Datetime start", "Datetime end"):
                    if c in work.columns:
                        work[c] = pd.to_datetime(work[c], errors="coerce")

                if "Energy (Kwh)" in work.columns:
                    work["Energy (Kwh)"] = pd.to_numeric(work["Energy (Kwh)"], errors="coerce")

                for c in ("SOC Start", "SOC End"):
                    if c in work.columns:
                        work[c] = pd.to_numeric(work[c], errors="coerce")

                if "MAC Address" in work.columns:
                    work["MAC Address"] = work["MAC Address"].apply(_fmt_mac)

                def _etiquette(row):
                    t = str(row.get("type_erreur", "") or "")
                    m = str(row.get("moment", "") or "")
                    return f"{t} — {m}" if m else t

                work["Erreur"] = work.apply(_etiquette, axis=1)

                def _soc_evo(row):
                    s0 = row.get("SOC Start", pd.NA)
                    s1 = row.get("SOC End", pd.NA)
                    if pd.notna(s0) and pd.notna(s1):
                        try:
                            return f"{int(round(s0))}% → {int(round(s1))}%"
                        except Exception:
                            return ""
                    return ""

                work["Évolution SOC"] = work.apply(_soc_evo, axis=1)

                if "ID" not in work.columns:
                    st.warning("Colonne 'ID' absente : les liens ELTO ne seront pas affichés.")
                    work["ELTO"] = ""
                else:
                    work["ELTO"] = BASE_CHARGE_URL + work["ID"].astype(str).str.strip()

                if "Vehicle" not in work.columns and "ID" in work.columns:
                    if (
                        isinstance(lookup, pd.DataFrame)
                        and not lookup.empty
                        and {"ID", "Vehicle"}.issubset(lookup.columns)
                    ):
                        veh_map = lookup[["ID", "Vehicle"]].drop_duplicates("ID", keep="last")
                        work = work.merge(veh_map, on="ID", how="left")

                return work

            def _build_charge_table(df: pd.DataFrame) -> pd.DataFrame:
                cols_aff = [
                    "Site",
                    "Datetime start",
                    "Datetime end",
                    "Energy (Kwh)",
                    "MAC Address",
                    "Vehicle",
                    "Erreur",
                    "Évolution SOC",
                    "ELTO",
                ]
                cols_aff = [c for c in cols_aff if c in df.columns]

                out_df = df[cols_aff].copy()
                if "Datetime start" in out_df.columns:
                    out_df = out_df.sort_values("Datetime start", ascending=False)

                out_df.insert(0, "#", range(1, len(out_df) + 1))
                return out_df

            charges_lookup = None
            if (
                "charges_mac" in locals()
                and isinstance(charges_mac, pd.DataFrame)
            ):
                charges_lookup = charges_mac

            column_config_common = {
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

            err_sum = df_src_f.loc[~df_src_f["is_ok"]].copy()
            err_sum = _prepare_charge_details(err_sum, charges_lookup)

            if err_sum.empty:
                st.info("Aucune charge en erreur pour le périmètre/filtre sélectionné.")
            else:
                st.markdown("#### Charges en erreur")
                out = _build_charge_table(err_sum)
                st.data_editor(
                    out,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config_common,
                )

            ok_sum = df_src_f.loc[df_src_f["is_ok"]].copy()
            ok_sum = _prepare_charge_details(ok_sum, charges_lookup)

            if ok_sum.empty:
                st.info("Aucune charge OK pour le périmètre/filtre sélectionné.")
            else:
                st.markdown("#### Charges OK")
                out_ok = _build_charge_table(ok_sum)
                st.data_editor(
                    out_ok,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config_common,
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

        dfv_local = None
        if "dfv" in locals() and isinstance(dfv, pd.DataFrame):
            dfv_local = dfv
        elif "charges_mac" in locals() and isinstance(charges_mac, pd.DataFrame):
            dfv_local = charges_mac

        if isinstance(dfv_local, pd.DataFrame) and not dfv_local.empty:
            dfv_local = dfv_local.copy()

            if "Datetime start" in dfv_local.columns:
                dfv_local["Datetime start"] = pd.to_datetime(dfv_local["Datetime start"], errors="coerce")

            vehicle_col = "Vehicle" if "Vehicle" in dfv_local.columns else None
            site_col = "Site" if "Site" in dfv_local.columns else (
                "Name Project" if "Name Project" in dfv_local.columns else None
            )

            if vehicle_col is not None:
                veh_series = dfv_local[vehicle_col].astype(str).str.strip()
                veh_series = veh_series.replace({
                    "": np.nan,
                    "nan": np.nan,
                    "none": np.nan,
                    "NULL": np.nan,
                }, regex=False)
                dfv_local[vehicle_col] = veh_series

            if vehicle_col is not None:
                mask_dfv = pd.Series(True, index=dfv_local.index)

                if site_col and st.session_state.get("site_sel"):
                    mask_dfv &= dfv_local[site_col].isin(st.session_state.site_sel)

                if (
                    "Datetime start" in dfv_local.columns
                    and st.session_state.get("d1")
                    and st.session_state.get("d2")
                ):
                    d1 = pd.Timestamp(st.session_state.get("d1"))
                    d2 = pd.Timestamp(st.session_state.get("d2")) + pd.Timedelta(days=1)
                    mask_dfv &= dfv_local["Datetime start"].ge(d1)
                    mask_dfv &= dfv_local["Datetime start"].lt(d2)

                dfv_filtered = dfv_local.loc[mask_dfv & dfv_local[vehicle_col].notna()].copy()
                if not dfv_filtered.empty:
                    dfv_filtered[vehicle_col] = dfv_filtered[vehicle_col].astype(str).str.strip()
                    dfv_filtered = dfv_filtered[dfv_filtered[vehicle_col].str.len().gt(0)]
                    dfv_filtered = dfv_filtered[dfv_filtered[vehicle_col].str.lower() != "unknown"]

                if not dfv_filtered.empty:
                    total_charges = (
                        dfv_filtered.groupby(vehicle_col)
                        .size()
                        .reset_index(name="Total Charges")
                    )

                    occ_vehicle = occ_vehicle.merge(
                        total_charges,
                        on="Vehicle",
                        how="left"
                    )

        if "Total Charges" in occ_vehicle.columns:
            occ_vehicle["Total Charges"] = occ_vehicle["Total Charges"].fillna(0).astype(int)
            occ_vehicle["Vehicle Label"] = occ_vehicle.apply(
                lambda row: f"{row['Vehicle']} ({row['Total Charges']})",
                axis=1
            )
        else:
            occ_vehicle["Vehicle Label"] = occ_vehicle["Vehicle"].astype(str)

        occ_vehicle = occ_vehicle.sort_values("Occurrences", ascending=True)

        bar_kwargs = dict(
            data_frame=occ_vehicle,
            x="Occurrences",
            y="Vehicle Label",
            orientation="h",
            title="Occurrences par véhicule (avec total charges)",
            labels={"Occurrences": "Nb d'occurrences", "Vehicle Label": "Véhicule"},
        )
        if "Total Charges" in occ_vehicle.columns:
            bar_kwargs["hover_data"] = {"Occurrences": True, "Total Charges": True}

        fig_vehicle = px.bar(**bar_kwargs)
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
        st.info("Colonnes 'Datetime start' ou 'Site' manquantes pour tracer l'histogramme.")
# Nombre d'occurrences par site et PDC
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
        st.markdown("### Nombre d'occurrences par site et PDC")
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

"""

def render():
    ctx = get_context()
    globals_dict = {"np": np, "pd": pd, "px": px, "go": go, "st": st}
    local_vars = dict(ctx.__dict__)
    local_vars.setdefault('plot', getattr(ctx, 'plot', None))
    local_vars.setdefault('hide_zero_labels', getattr(ctx, 'hide_zero_labels', None))
    local_vars.setdefault('with_charge_link', getattr(ctx, 'with_charge_link', None))
    local_vars.setdefault('evi_counts_pivot', getattr(ctx, 'evi_counts_pivot', None))
    tables_ref = local_vars.get('tables')
    if isinstance(tables_ref, dict):
        local_vars.setdefault('charges_mac', tables_ref.get('charges_mac', pd.DataFrame()))
    # remove None entries
    local_vars = {k: v for k, v in local_vars.items() if v is not None}
    exec(TAB_CODE, globals_dict, local_vars)
