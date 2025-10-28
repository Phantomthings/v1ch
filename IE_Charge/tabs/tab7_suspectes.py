import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabs.context import get_context

TAB_CODE = """
st.subheader("Transactions suspectes (<1 kWh)")
suspicious = tables.get("suspicious_under_1kwh", pd.DataFrame())
if suspicious.empty:
    st.success("Aucune transaction suspecte détectée (<1 kWh).")
else:
    df_s = suspicious.copy()
    if "Site" in df_s.columns:
        site_options = st.session_state.get("site_sel", [])
        if not site_options:
            site_options = sorted(df_s["Site"].dropna().unique().tolist())
        if site_options:
            default_site = st.session_state.get("tab7_site_single")
            if default_site not in site_options:
                default_site = site_options[0]

            if len(site_options) == 1:
                selected_site = site_options[0]
                st.session_state["tab7_site_single"] = selected_site
                st.caption(f"Site sélectionné : {selected_site}")
            else:
                selected_site = st.selectbox(
                    "Sélection du site",
                    options=site_options,
                    index=site_options.index(default_site) if default_site in site_options else 0,
                    key="tab7_site_single",
                    help="Choisissez un site. Le tableau affiche les transactions suspectes pour un seul site.",
                )

            selected_site = st.session_state.get("tab7_site_single", default_site)
            df_s = df_s[df_s["Site"] == selected_site]
        else:
            st.info("Aucun site disponible pour cette vue.")
            df_s = df_s.iloc[0:0]

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

