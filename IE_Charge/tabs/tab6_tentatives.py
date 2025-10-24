import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabs.context import get_context

BASE_CHARGE_URL = "https://elto.nidec-asi-online.com/Charge/detail?id=""

TAB_CODE = """
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

"""

def render():
    ctx = get_context()
    globals_dict = {"np": np, "pd": pd, "px": px, "go": go, "st": st, "BASE_CHARGE_URL": BASE_CHARGE_URL}
    local_vars = dict(ctx.__dict__)
    local_vars.setdefault('plot', getattr(ctx, 'plot', None))
    local_vars.setdefault('hide_zero_labels', getattr(ctx, 'hide_zero_labels', None))
    local_vars.setdefault('with_charge_link', getattr(ctx, 'with_charge_link', None))
    local_vars.setdefault('evi_counts_pivot', getattr(ctx, 'evi_counts_pivot', None))
    # remove None entries
    local_vars = {k: v for k, v in local_vars.items() if v is not None}
    exec(TAB_CODE, globals_dict, local_vars)

