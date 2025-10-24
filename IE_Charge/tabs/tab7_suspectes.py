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

