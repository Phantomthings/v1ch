import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabs.context import get_context

TAB_CODE = """
st.markdown("### ⚠️ Alertes : erreurs récurrentes par PDC")
errors_only = sess_kpi[~sess_kpi["is_ok_filt"]].copy()
if errors_only.empty:
    st.info("Aucune erreur dans le périmètre.")
else:
    errors_only["Datetime start"] = pd.to_datetime(errors_only["Datetime start"], errors="coerce")
    errors_only = errors_only.dropna(subset=["Datetime start", "PDC", "type_erreur"])
    errors_only = errors_only.sort_values(["PDC", "type_erreur", "Datetime start"]).reset_index()

    alert_rows = []

    for (pdc, err_type), group in errors_only.groupby(["PDC", "type_erreur"]):
        times = group["Datetime start"]
        idxs = group["index"]

        for i in range(len(times)):
            t0 = times.iloc[i]
            t1 = t0 + pd.Timedelta(hours=12)
            window = times[(times >= t0) & (times <= t1)]
            if len(window) >= 3:
                idx3 = idxs.iloc[i] 
                row = sess_kpi.loc[idx3]

                alert_rows.append({
                    "Site": row.get(SITE_COL, "—"),
                    "PDC": pdc,
                    "Type d'erreur": err_type,
                    "Détection": t0,
                    "Occurrences sur 12h": len(window),
                    "Moment": row.get("moment", "—"),
                    "EVI Code": row.get("EVI Error Code", "—"),
                    "Downstream Code PC": row.get("Downstream Code PC", "—")
                })
                break  

    if not alert_rows:
        st.success("✅ Aucune alerte détectée.")
    else:
        df_alertes = pd.DataFrame(alert_rows).sort_values("Détection", ascending=False)
        st.dataframe(df_alertes, use_container_width=True)
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

