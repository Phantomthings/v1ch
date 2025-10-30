import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabs.context import get_context

TAB_CODE = """
st.subheader("Projection pivot — Moments (ligne 1) × Codes (ligne 2)")

if "sess" not in locals() or "mask_nok_keep" not in locals():
    st.info("Contexte incomplet pour calculer la projection. Merci de relancer l'application.")
else:
    err = sess.loc[mask_nok_keep].copy()
    if err.empty:
        st.info("Aucune erreur après application des filtres.")
    else:
        site_col = "Site" if "Site" in err.columns else "Name Project"

        def _to_int(series, default=0):
            if series is None:
                return pd.Series(default, index=err.index)
            return pd.to_numeric(series, errors="coerce").fillna(default).astype(int)

        def _map_moment(step_val: int) -> str:
            if pd.isna(step_val):
                return "Unknown"
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

        err["step_num"] = _to_int(err.get("EVI Status during error"), default=-1)
        err["evi_code_num"] = _to_int(err.get("EVI Error Code"), default=0)
        err["ds_code_num"] = _to_int(err.get("Downstream Code PC"), default=0)

        if "moment" in err.columns:
            err["moment_label"] = err["moment"].fillna("Unknown")
        else:
            err["moment_label"] = err["step_num"].map(_map_moment)

        ds_pc = err["ds_code_num"]
        evi_code = err["evi_code_num"]

        mask_evi = ds_pc.eq(8192) | (ds_pc.eq(0) & evi_code.ne(0))
        mask_ds = ds_pc.ne(0) & ds_pc.ne(8192)

        moment_priority = {
            "Init": 0,
            "Lock Connector": 1,
            "CableCheck": 2,
            "Charge": 3,
            "Fin de charge": 4,
            "Unknown": 5,
        }

        combo_frames = []

        base_cols = [site_col, "moment_label", "step_num"]
        if "PDC" in err.columns:
            base_cols.append("PDC")

        if mask_evi.any():
            evi_tmp = err.loc[mask_evi, base_cols + ["evi_code_num"]].copy()
            evi_tmp["type_label"] = "EVI"
            evi_tmp["code_label"] = evi_tmp["evi_code_num"].apply(lambda x: f"EVI {int(x)}" if int(x) != 0 else "EVI 0")
            combo_frames.append(evi_tmp)

        if mask_ds.any():
            ds_tmp = err.loc[mask_ds, base_cols + ["ds_code_num"]].copy()
            ds_tmp["type_label"] = "Downstream"
            ds_tmp["code_label"] = ds_tmp["ds_code_num"].apply(lambda x: f"DS {int(x)}")
            combo_frames.append(ds_tmp)

        if not combo_frames:
            st.info("Aucune combinaison d'erreurs EVI/Downstream pour les filtres sélectionnés.")
        else:
            combo = pd.concat(combo_frames, ignore_index=True)
            combo = combo.rename(columns={site_col: "Site"})
            site_col = "Site"
            combo["moment_label"] = combo["moment_label"].fillna("Unknown")

            def _code_sort(label) -> float:
                try:
                    return float(str(label).split(" ")[-1])
                except Exception:
                    return float("inf")

            combo["moment_sort"] = combo["moment_label"].map(moment_priority).fillna(999)
            combo["code_sort"] = combo["code_label"].map(_code_sort)

            available_types = sorted(combo["type_label"].dropna().unique().tolist())
            type_options = ["Toutes"] + available_types
            default_type = st.session_state.get("tab5_projection_type", "Toutes")
            if default_type not in type_options:
                default_type = "Toutes"
            selected_type = st.radio(
                "Type d'erreur",
                options=type_options,
                horizontal=True,
                index=type_options.index(default_type),
                key="tab5_projection_type",
                help="Filtrez les erreurs utilisées dans le tableau croisé.",
            )

            if selected_type != "Toutes":
                combo_filtered = combo[combo["type_label"] == selected_type].copy()
            else:
                combo_filtered = combo.copy()

            if combo_filtered.empty:
                st.info("Aucune donnée pour ce type d'erreur.")
            else:
                site_options = sorted(combo_filtered[site_col].dropna().unique().tolist())
                if not site_options:
                    st.info("Aucun site disponible après filtrage.")
                else:
                    default_sites = st.session_state.get("tab5_projection_sites", [])
                    if not isinstance(default_sites, list):
                        default_sites = [default_sites] if default_sites else []
                    default_sites = [s for s in default_sites if s in site_options]
                    if not default_sites:
                        default_sites = site_options[:1]

                    selected_sites = st.multiselect(
                        "Sites (projection) - Maximum 2 sites",
                        options=site_options,
                        default=default_sites,
                        key="tab5_projection_sites",
                        help="Sélectionnez jusqu'à 2 sites pour l'analyse de projection.",
                    )

                    if len(selected_sites) > 2:
                        st.error("⚠️ Vous ne pouvez sélectionner que 2 sites maximum. Veuillez désélectionner un site.")
                        selected_sites = selected_sites[:2]

                    if not selected_sites:
                        st.info("Veuillez sélectionner au moins un site.")
                    else:
                        for site in selected_sites:
                            st.markdown(f"### 📍 {site}")

                            hide_zero = st.checkbox(
                                "Masquer colonnes vides (0)",
                                key=f"hide_zeros_{site}_{selected_type}",
                            )

                            site_combo = combo_filtered[combo_filtered[site_col] == site].copy()
                            if site_combo.empty:
                                st.info("Aucune erreur pour ce site.")
                                continue

                            site_sessions = sess[sess[SITE_COL] == site].copy() if SITE_COL in sess.columns else pd.DataFrame()
                            total_sessions = len(site_sessions)
                            ok_sessions = (
                                site_sessions["is_ok"].astype(int).sum()
                                if total_sessions and "is_ok" in site_sessions.columns
                                else 0
                            )
                            taux_reussite_site = (
                                round(ok_sessions / total_sessions * 100, 2)
                                if total_sessions
                                else 0.0
                            )
                            st.markdown(
                                f"**Taux de réussite du site : {taux_reussite_site:.2f}%**"
                                + (
                                    f" — {ok_sessions}/{total_sessions} sessions réussies"
                                    if total_sessions
                                    else ""
                                )
                            )

                            sort_cols = ["moment_sort", "code_sort", "moment_label", "code_label"]
                            site_combo = site_combo.sort_values(sort_cols)

                            has_pdc = "PDC" in site_combo.columns and site_combo["PDC"].notna().any()

                            if has_pdc:
                                g_tot = (
                                    site_combo.groupby(["moment_label", "code_label"], as_index=False)
                                    .size()
                                    .rename(columns={"size": "Nb"})
                                )
                                g_tot["PDC"] = "__TOTAL__"

                                g_pdc = (
                                    site_combo.groupby(["PDC", "moment_label", "code_label"], as_index=False)
                                    .size()
                                    .rename(columns={"size": "Nb"})
                                )

                                full = pd.concat([g_tot, g_pdc], ignore_index=True)

                                pv = full.pivot_table(
                                    index="PDC",
                                    columns=["moment_label", "code_label"],
                                    values="Nb",
                                    fill_value=0,
                                    aggfunc="sum",
                                )

                                sorted_cols = sorted(
                                    pv.columns,
                                    key=lambda c: (moment_priority.get(c[0], 999), c[1]),
                                )
                                pv = pv.reindex(columns=sorted_cols, fill_value=0)

                                pdcs = sorted(pv.index.tolist(), key=lambda x: (x != "__TOTAL__", str(x)))
                                pv = pv.reindex(pdcs)

                                df_disp = pv.reset_index()
                                df_disp["Site / PDC"] = np.where(
                                    df_disp["PDC"].eq("__TOTAL__"),
                                    f"{site} (TOTAL)",
                                    "   " + df_disp["PDC"].astype(str),
                                )
                                df_disp = df_disp.drop(columns=["PDC"])
                            else:
                                g_site = (
                                    site_combo.groupby(["moment_label", "code_label"], as_index=False)
                                    .size()
                                    .rename(columns={"size": "Nb"})
                                )
                                pv = g_site.pivot_table(
                                    index=pd.Index([site], name="Site"),
                                    columns=["moment_label", "code_label"],
                                    values="Nb",
                                    fill_value=0,
                                    aggfunc="sum",
                                )
                                sorted_cols = sorted(
                                    pv.columns,
                                    key=lambda c: (moment_priority.get(c[0], 999), c[1]),
                                )
                                pv = pv.reindex(columns=sorted_cols, fill_value=0)

                                df_disp = pv.reset_index(drop=True)
                                df_disp.insert(0, "Site / PDC", f"{site} (TOTAL)")

                            cols = df_disp.columns
                            disp_col = (
                                ("Site / PDC", "")
                                if isinstance(cols, pd.MultiIndex) and ("Site / PDC", "") in cols
                                else "Site / PDC"
                            )

                            if isinstance(cols, pd.MultiIndex):
                                value_cols = [c for c in cols if c != disp_col]
                                df_disp = df_disp.loc[:, [disp_col] + value_cols]
                            else:
                                value_cols = [c for c in cols if c != "Site / PDC"]
                                df_disp = df_disp[["Site / PDC"] + value_cols]

                            if not value_cols:
                                st.info("Aucune donnée à projeter pour ce site.")
                                continue

                            numeric_part = df_disp[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

                            if hide_zero and not numeric_part.empty:
                                col_sums = numeric_part.sum(axis=0)
                                keep_cols = [c for c in value_cols if col_sums[c] > 0]
                                if not keep_cols:
                                    st.info("Toutes les colonnes sont nulles pour ce site.")
                                    continue
                                value_cols = keep_cols
                                df_disp = pd.concat(
                                    [df_disp[[disp_col]], numeric_part[keep_cols]],
                                    axis=1,
                                )
                                numeric_part = df_disp[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

                            _total_col = (
                                ("∑", "Total")
                                if value_cols and isinstance(value_cols[0], tuple)
                                else "∑ Total"
                            )
                            _total_pct_col = (
                                ("∑", "%")
                                if isinstance(_total_col, tuple)
                                else "∑ %"
                            )

                            df_disp[_total_col] = numeric_part.sum(axis=1)

                            base_rows = df_disp[df_disp[disp_col].astype(str).str.startswith("   ")].copy()
                            if base_rows.empty:
                                base_rows = df_disp.copy()

                            col_totals = (
                                base_rows[value_cols]
                                .apply(pd.to_numeric, errors="coerce")
                                .fillna(0)
                                .sum()
                            )

                            total_general_value = int(col_totals.sum())

                            sum_dict = {disp_col: "TOTAL GÉNÉRAL"}
                            sum_dict.update({col: int(col_totals[col]) for col in value_cols})
                            sum_dict[_total_col] = total_general_value
                            sum_dict[_total_pct_col] = 100.0 if total_general_value else 0.0

                            sum_row = pd.DataFrame(
                                [sum_dict],
                                columns=[disp_col] + value_cols + [_total_col, _total_pct_col],
                            )

                            df_disp = pd.concat([df_disp, sum_row], ignore_index=True)

                            if total_general_value:
                                df_disp[_total_pct_col] = (
                                    df_disp[_total_col] / total_general_value * 100
                                ).round(1)
                            else:
                                df_disp[_total_pct_col] = 0.0

                            df_disp.loc[
                                df_disp[disp_col] == "TOTAL GÉNÉRAL",
                                _total_pct_col,
                            ] = 100.0 if total_general_value else 0.0

                            final_cols = [disp_col] + value_cols + [_total_col, _total_pct_col]

                            def _cell_color(v):
                                try:
                                    x = float(v)
                                except Exception:
                                    return ""
                                if x == 0:
                                    return "background-color: #ffffff;"
                                if x <= 2:
                                    return "background-color: #E8F1FB;"
                                if x <= 6:
                                    return "background-color: #CFE3F7;"
                                if x <= 15:
                                    return "background-color: #A9CFF2;"
                                if x <= 25:
                                    return "background-color: #7DB5EA;"
                                if x <= 50:
                                    return "background-color: #4F97D9; color: white;"
                                if x <= 100:
                                    return "background-color: #2F6FB7; color: white;"
                                return "background-color: #1F4F8F; color: white;"

                            styled = (
                                df_disp[final_cols]
                                .style
                                .applymap(_cell_color, subset=value_cols)
                                .format(precision=0, na_rep="")
                                .format({_total_pct_col: "{:.1f}%"})
                                .set_table_styles([
                                    {
                                        "selector": "th.col_heading.level0",
                                        "props": [("text-align", "center")],
                                    },
                                    {
                                        "selector": "th.col_heading.level1",
                                        "props": [("text-align", "center")],
                                    },
                                ])
                            )
                            st.dataframe(styled, use_container_width=True)

    st.markdown(\"\"\"
    **Légende (occurrences)**
    <span style=\"display:inline-block;width:14px;height:14px;background:#ffffff;border:1px solid #ddd;\"></span> 0
    <span style=\"display:inline-block;width:14px;height:14px;background:#E8F1FB;\"></span> 0–2
    <span style=\"display:inline-block;width:14px;height:14px;background:#CFE3F7;\"></span> 2–6
    <span style=\"display:inline-block;width:14px;height:14px;background:#A9CFF2;\"></span> 6–15
    <span style=\"display:inline-block;width:14px;height:14px;background:#7DB5EA;\"></span> 15–25
    <span style=\"display:inline-block;width:14px;height:14px;background:#4F97D9;\"></span> 25–50
    <span style=\"display:inline-block;width:14px;height:14px;background:#2F6FB7;\"></span> 50–100
    <span style=\"display:inline-block;width:14px;height:14px;background:#1F4F8F;\"></span> >100
    <style>
    .stDataFrame table thead th:first-child {min-width: 220px !important;}
    </style>
    \"\"\", unsafe_allow_html=True)


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
