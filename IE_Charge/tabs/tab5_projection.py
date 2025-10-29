import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabs.context import get_context

TAB_CODE = """
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
    site_options = [] if evi_f.empty else sorted(evi_f["Site"].dropna().unique())

    if not site_options:
        st.info("Aucune combinaison sur ce périmètre (après filtres).")

    else:
        default_sites = st.session_state.get("tab5_projection_sites")
        if not isinstance(default_sites, list):
            default_sites = []
        default_sites = [s for s in default_sites if s in site_options][:2]
        if not default_sites:
            default_sites = [site_options[0]]

        selected_sites = st.multiselect(
            "Sites (projection)",
            options=site_options,
            default=default_sites,
            key="tab5_projection_sites",
            help="Sélectionnez jusqu'à 2 sites pour l'analyse de projection.",
        )

        if len(selected_sites) > 2:
            st.warning("Vous pouvez sélectionner au maximum 2 sites pour la projection.")
            selected_sites = selected_sites[:2]
            st.session_state["tab5_projection_sites"] = selected_sites

        if not selected_sites:
            st.info("Sélectionnez au moins un site pour afficher la projection.")

        for site in selected_sites:
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
            _total_pct_col = ("∑", "%") if isinstance(_total_col, tuple) else "∑ %"
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
            _sum_dict.update({col: int(_col_totals[col]) for col in value_cols})
            total_general_value = int(_col_totals.sum())
            _sum_dict[_total_col] = total_general_value
            _sum_dict[_total_pct_col] = 100.0 if total_general_value else 0.0
            _sum_row = pd.DataFrame([_sum_dict], columns=[disp_col] + value_cols + [_total_col, _total_pct_col])

            df_disp = pd.concat([df_disp, _sum_row], ignore_index=True)

            if total_general_value:
                df_disp[_total_pct_col] = (df_disp[_total_col] / total_general_value * 100).round(1)
            else:
                df_disp[_total_pct_col] = 0.0
            df_disp.loc[df_disp[disp_col] == "TOTAL GÉNÉRAL", _total_pct_col] = 100.0 if total_general_value else 0.0

            final_cols = [disp_col] + value_cols + [_total_col, _total_pct_col]

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
                .format({
                    _total_pct_col: "{:.1f}%"
                })
                .set_table_styles([
                    {"selector": "th.col_heading.level0", "props": [("text-align", "center")]},
                    {"selector": "th.col_heading.level1", "props": [("text-align", "center")]},
                ])
            )
            st.dataframe(styled, use_container_width=True)
    st.markdown(\"\"\"
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
