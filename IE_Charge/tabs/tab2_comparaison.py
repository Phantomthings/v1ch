import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabs.context import get_context

TAB_CODE = """
st.subheader("Statistiques par site")
by_site_f = (
    sess_kpi.groupby(SITE_COL, as_index=False)
            .agg(
                Total_Charges=("is_ok_filt", "count"),
                Charges_OK=("is_ok_filt", "sum")
            )
)
by_site_f["Charges_NOK"] = by_site_f["Total_Charges"] - by_site_f["Charges_OK"]
by_site_f["% Réussite"] = np.where(
    by_site_f["Total_Charges"].gt(0),
    (by_site_f["Charges_OK"] / by_site_f["Total_Charges"] * 100).round(2),
    0.0
)
by_site_f["% Échec"] = np.where(
    by_site_f["Total_Charges"].gt(0),
    (by_site_f["Charges_NOK"] / by_site_f["Total_Charges"] * 100).round(2),
    0.0
)
by_site_f = by_site_f.reset_index(drop=True)
st.dataframe(by_site_f, use_container_width=True, hide_index=True)
# Barres 
by_site_sorted = by_site_f.sort_values("Total_Charges", ascending=True)
sites = by_site_sorted[SITE_COL].tolist()
ok = by_site_sorted["Charges_OK"]
nok = by_site_sorted["Charges_NOK"]
ok_pct = by_site_sorted["% Réussite"]
nok_pct = by_site_sorted["% Échec"]

fig = go.Figure()

# Traces en Nombre
fig.add_bar(name="Charges OK", x=sites, y=ok, text=ok, marker_color="royalblue", visible=True)
fig.add_bar(name="Charges NOK", x=sites, y=nok, text=nok, marker_color="orangered", visible=True)

# Traces en %
fig.add_bar(name="Charges OK (%)", x=sites, y=ok_pct, 
            text=[f"{v:.1f}%" for v in ok_pct], marker_color="royalblue", visible=False)
fig.add_bar(name="Charges NOK (%)", x=sites, y=nok_pct,
            text=[f"{v:.1f}%" for v in nok_pct], marker_color="orangered", visible=False)

# Boutons toggle
fig.update_layout(
    barmode="group",
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=1.1, y=1.15,
            buttons=[
                dict(label="Nombre", method="update",
                    args=[{"visible": [True, True, False, False]},
                        {"yaxis": {"title": "Nombre de charges"}}]),
                dict(label="%", method="update",
                    args=[{"visible": [False, False, True, True]},
                        {"yaxis": {"title": "% de charges"}}])
            ]
        )
    ]
)

fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True, key="chart_ok_nok_toggle")
st.divider()
st.subheader("Analyse temporelle")
if "Datetime start" not in sess_kpi.columns:
    st.info("Colonne 'Datetime start' absente.")
else:
    base = sess_kpi.copy()
    if base.empty:
        st.warning("Aucune charge sur ce périmètre")
    else:
        base["hour"] = pd.to_datetime(base["Datetime start"], errors="coerce").dt.hour

        g = (
            base.dropna(subset=["hour"])
                .groupby([SITE_COL, "hour"])
                .size()
                .reset_index(name="Nb")
        )

        if g.empty:
            st.info("Pas d'heures valides pour les charges.")
        else:
            peak = g.loc[g.groupby(SITE_COL)["Nb"].idxmax()][[SITE_COL, "hour", "Nb"]] \
                    .rename(columns={"hour": "Heure de pic", "Nb": "Nb au pic"})

            def _w_median_hours(dfh: pd.DataFrame) -> int:
                s = dfh.sort_values("hour")
                c = s["Nb"].cumsum()
                half = s["Nb"].sum() / 2.0
                return int(s.loc[c >= half, "hour"].iloc[0])

            med = g.groupby(SITE_COL).apply(_w_median_hours).reset_index(name="Heure médiane")

            summ = peak.merge(med, on=SITE_COL, how="left")
            for col in ["Heure de pic", "Heure médiane"]:
                summ[col] = summ[col].astype(int).apply(lambda x: f"{x:02d}:00")
            st.dataframe(
                summ[[SITE_COL, "Heure de pic", "Nb au pic", "Heure médiane"]].sort_values(SITE_COL),
                use_container_width=True,
                hide_index=True
            )

            # HEATMAP 
            pivot = g.pivot(index=SITE_COL, columns="hour", values="Nb").fillna(0)

            fig = px.imshow(
                pivot,
                labels=dict(x="Heure", y="Site", color="Nb charges"),
                x=[f"{h:02d}:00" for h in pivot.columns],
                y=pivot.index,
                color_continuous_scale="Blues",
                aspect="auto"
            )

            fig.update_layout(
                xaxis=dict(side="top")
            )

            st.plotly_chart(fig, use_container_width=True, key="heatmap_all")
            st.divider()
            site_options = summ[SITE_COL].tolist() 
            if len(site_options) == 0:
                st.info("Aucun site disponible après filtres pour le zoom horaire.")
            else:
                site_focus_both = st.selectbox(
                    "📊 Zoom sur un site",
                    options=site_options
                )

                base_site = base[base[SITE_COL] == site_focus_both].copy()
                ok_focus_all  = base_site[base_site["is_ok_filt"]].copy()
                nok_focus_all = base_site[~base_site["is_ok_filt"]].copy()

                ok_focus_all["month"]  = pd.to_datetime(ok_focus_all["Datetime start"], errors="coerce").dt.to_period("M").astype(str)
                nok_focus_all["month"] = pd.to_datetime(nok_focus_all["Datetime start"], errors="coerce").dt.to_period("M").astype(str)

                g_ok_m  = ok_focus_all.groupby("month").size().reset_index(name="Nb").assign(Status="OK")
                g_nok_m = nok_focus_all.groupby("month").size().reset_index(name="Nb").assign(Status="NOK")

                g_both_m = pd.concat([g_ok_m, g_nok_m], ignore_index=True)
                g_both_m["month"] = pd.to_datetime(g_both_m["month"], errors="coerce")
                g_both_m = g_both_m.dropna(subset=["month"]).sort_values("month")
                g_both_m["month"] = g_both_m["month"].dt.strftime("%Y-%m")

                piv_m = (
                    g_both_m.pivot(index="month", columns="Status", values="Nb")
                            .fillna(0)
                            .sort_index()
                )
                months = piv_m.index.tolist()

                ok_num = piv_m["OK"].tolist() if "OK" in piv_m else []
                nok_num = piv_m["NOK"].tolist() if "NOK" in piv_m else []

                tot_m = piv_m.sum(axis=1).replace(0, np.nan)
                ok_pct = (piv_m["OK"] / tot_m * 100).fillna(0).round(1).tolist() if "OK" in piv_m else [0] * len(piv_m)
                nok_pct = (piv_m["NOK"] / tot_m * 100).fillna(0).round(1).tolist() if "NOK" in piv_m else [0] * len(piv_m)

                fig_both_m = go.Figure()
                # Nombre
                fig_both_m.add_bar(name="OK (Nb)",  x=months, y=ok_num,  text=ok_num,  marker_color="#38AC21", visible=True)
                fig_both_m.add_bar(name="NOK (Nb)", x=months, y=nok_num, text=nok_num, marker_color="#EF553B", visible=True)
                # %
                fig_both_m.add_bar(name="OK (%)",  x=months, y=ok_pct,  text=[f"{v:.1f}%" for v in ok_pct],  marker_color="#38AC21", visible=False)
                fig_both_m.add_bar(name="NOK (%)", x=months, y=nok_pct, text=[f"{v:.1f}%" for v in nok_pct], marker_color="#EF553B", visible=False)

                fig_both_m.update_layout(
                    title=f"Distribution mensuelle OK vs NOK — {site_focus_both}",
                    barmode="group",
                    xaxis=dict(type="category"),
                    updatemenus=[
                        dict(
                            type="buttons", direction="right", x=1.1, y=1.15,
                            buttons=[
                                dict(label="Nombre", method="update",
                                    args=[{"visible": [True, True, False, False]},
                                        {"yaxis": {"title": "Nombre"}}]),
                                dict(label="%", method="update",
                                    args=[{"visible": [False, False, True, True]},
                                        {"yaxis": {"title": "%"}}]),
                            ]
                        )
                    ]
                )
                fig_both_m.update_traces(textposition="outside")
                plot(fig_both_m, f"tab2_ok_nok_month_distribution_{site_focus_both}")
                months_all = months 
                if months_all:
                    month_focus = st.selectbox(
                        "📅 Focus: afficher le détail par jour pour un mois",
                        options=months_all,
                        index=len(months_all)-1,
                        key=f"month_focus_days_{site_focus_both}"
                    )

                    ok_mo  = ok_focus_all[ok_focus_all["month"]  == month_focus].copy()
                    nok_mo = nok_focus_all[nok_focus_all["month"] == month_focus].copy()

                    ok_mo["day"]  = pd.to_datetime(ok_mo["Datetime start"],  errors="coerce").dt.strftime("%Y-%m-%d")
                    nok_mo["day"] = pd.to_datetime(nok_mo["Datetime start"], errors="coerce").dt.strftime("%Y-%m-%d")

                    per = pd.Period(month_focus, freq="M")
                    m_start = per.to_timestamp(how="start")
                    m_end   = per.to_timestamp(how="end")
                    days = pd.date_range(m_start, m_end, freq="D").strftime("%Y-%m-%d")

                    g_ok_d  = ok_mo.groupby("day").size().reindex(days, fill_value=0).reset_index()
                    g_ok_d.columns = ["day", "Nb"];  g_ok_d["Status"] = "OK"
                    g_nok_d = nok_mo.groupby("day").size().reindex(days, fill_value=0).reset_index()
                    g_nok_d.columns = ["day", "Nb"]; g_nok_d["Status"] = "NOK"

                    g_both_d = pd.concat([g_ok_d, g_nok_d], ignore_index=True)

                    piv_d = g_both_d.pivot(index="day", columns="Status", values="Nb").fillna(0)
                    days_ord = piv_d.index.tolist()
                    ok_num_d  = (piv_d["OK"]  if "OK"  in piv_d else 0).tolist()
                    nok_num_d = (piv_d["NOK"] if "NOK" in piv_d else 0).tolist()

                    tot_d = (piv_d.sum(axis=1)).replace(0, np.nan)
                    ok_pct_d  = (piv_d["OK"]  / tot_d * 100).fillna(0).round(1).tolist() if "OK"  in piv_d else [0]*len(days_ord)
                    nok_pct_d = (piv_d["NOK"] / tot_d * 100).fillna(0).round(1).tolist() if "NOK" in piv_d else [0]*len(days_ord)

                    fig_both_d = go.Figure()
                    # Nombre
                    fig_both_d.add_bar(name="OK (Nb)",  x=days_ord, y=ok_num_d,  text=ok_num_d,  marker_color="#38AC21", visible=True)
                    fig_both_d.add_bar(name="NOK (Nb)", x=days_ord, y=nok_num_d, text=nok_num_d, marker_color="#EF553B", visible=True)
                    # %
                    fig_both_d.add_bar(name="OK (%)",  x=days_ord, y=ok_pct_d,  text=[f"{v:.1f}%" for v in ok_pct_d],  marker_color="#38AC21", visible=False)
                    fig_both_d.add_bar(name="NOK (%)", x=days_ord, y=nok_pct_d, text=[f"{v:.1f}%" for v in nok_pct_d], marker_color="#EF553B", visible=False)

                    fig_both_d.update_layout(
                        title=f"OK vs NOK par jour — {site_focus_both} — {month_focus}",
                        barmode="group",
                        xaxis=dict(type="category", tickangle=-45),
                        updatemenus=[
                            dict(
                                type="buttons", direction="right", x=1.1, y=1.15,
                                buttons=[
                                    dict(label="Nombre", method="update",
                                        args=[{"visible": [True, True, False, False]},
                                            {"yaxis": {"title": "Nombre"}}]),
                                    dict(label="%", method="update",
                                        args=[{"visible": [False, False, True, True]},
                                            {"yaxis": {"title": "%"}}]),
                                ]
                            )
                        ]
                    )
                    fig_both_d.update_traces(textposition="outside")
                    plot(fig_both_d, f"tab2_ok_nok_day_distribution_{site_focus_both}_{month_focus}")
                else:
                    st.info("Aucun mois disponible pour le focus journalier.")
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

