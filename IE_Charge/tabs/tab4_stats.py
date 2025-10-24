import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabs.context import get_context

TAB_CODE = """
st.subheader("Statistiques générales")
# CSS
st.markdown(\"\"\"
    <style>
    .kpi-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .kpi-card {
      background: #0f172a; border: 1px solid #1f2937; border-radius: 14px;
      padding: 14px 16px; box-shadow: 0 2px 10px rgba(0,0,0,.15);
    }
    .kpi-title{ font-size:13px; color:#cbd5e1; margin:0 0 6px 0; }
    .kpi-value{ font-size:24px; font-weight:700; color:#f8fafc; margin:0; }
    .kpi-sub{ font-size:12px; color:#94a3b8; margin-top:4px; }
    .kpi-tag{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; margin-left:6px; background:#111827; border:1px solid #334155; color:#93c5fd;}
    .sec { font-weight:600; margin:18px 0 8px 0; }
    </style>
    \"\"\", unsafe_allow_html=True)

def card(title, value, sub=""):
    st.markdown(
        f\"\"\"<div class="kpi-card">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>\"\"\",
        unsafe_allow_html=True
    )

if "is_ok" not in sess.columns:
    st.info("Colonne is_ok absente.")
else:
    ok = sess[sess["is_ok"]].copy()
    if ok.empty:
        st.warning("Aucune charge OK dans ce périmètre.")
    else:
        dt_s   = pd.to_datetime(ok.get("Datetime start"), errors="coerce")
        dt_e   = pd.to_datetime(ok.get("Datetime end"), errors="coerce")
        energy = pd.to_numeric(ok.get("Energy (Kwh)"), errors="coerce")
        pmean  = pd.to_numeric(ok.get("Mean Power (Kw)"), errors="coerce")
        pmax   = pd.to_numeric(ok.get("Max Power (Kw)"), errors="coerce")
        soc_s  = pd.to_numeric(ok.get("SOC Start"), errors="coerce")
        soc_e  = pd.to_numeric(ok.get("SOC End"), errors="coerce")
        dur_min = (dt_e - dt_s).dt.total_seconds() / 60

        def date_of(idx):
            if pd.isna(idx) or idx not in ok.index: return "—"
            d = dt_e.loc[idx]
            if pd.isna(d): d = dt_s.loc[idx]
            return d.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(d) else "—"

        def lieu_of(idx):
            if pd.isna(idx) or idx not in ok.index: return "—"
            site = str(ok.loc[idx].get("Site", ok.loc[idx].get("Name Project", ""))) or "—"
            pdc  = str(ok.loc[idx].get("PDC", "—"))
            return f"{site} — PDC {pdc}"

        def idxmin_thresh(series: pd.Series, thr: float):
            s = series.where(series >= thr)
            return s.idxmin() if s.notna().any() else np.nan

        # seuils minima à ignorer
        THR_ENERGY = 4.0
        THR_PMEAN  = 4.0
        THR_PMAX   = 4.0

        # ÉNERGIE
        e_total = round(float(energy.sum(skipna=True)), 3) if energy.notna().any() else 0
        e_mean  = round(float(energy.mean(skipna=True)), 3) if energy.notna().any() else 0
        e_min_i = idxmin_thresh(energy, THR_ENERGY)
        e_max_i = energy.idxmax() if energy.notna().any() else np.nan
        e_min_v = (round(float(energy.loc[e_min_i]), 3) if e_min_i==e_min_i else "—")
        e_max_v = (round(float(energy.loc[e_max_i]), 3) if e_max_i==e_max_i else "—")

        # Pmean
        pm_mean = round(float(pmean.mean(skipna=True)), 3) if pmean.notna().any() else 0
        pm_min_i = idxmin_thresh(pmean, THR_PMEAN)
        pm_max_i = pmean.idxmax() if pmean.notna().any() else np.nan
        pm_min_v = (round(float(pmean.loc[pm_min_i]), 3) if pm_min_i==pm_min_i else "—")
        pm_max_v = (round(float(pmean.loc[pm_max_i]), 3) if pm_max_i==pm_max_i else "—")

        # Pmax
        px_mean = round(float(pmax.mean(skipna=True)), 3) if pmax.notna().any() else 0
        px_min_i = idxmin_thresh(pmax, THR_PMAX)
        px_max_i = pmax.idxmax() if pmax.notna().any() else np.nan
        px_min_v = (round(float(pmax.loc[px_min_i]), 3) if px_min_i==px_min_i else "—")
        px_max_v = (round(float(pmax.loc[px_max_i]), 3) if px_max_i==px_max_i else "—")

        # SOC
        soc_start_mean = round(float(soc_s.mean(skipna=True)), 2) if soc_s.notna().any() else 0
        soc_end_mean   = round(float(soc_e.mean(skipna=True)), 2) if soc_e.notna().any() else 0

        # Durées
        d_mean = round(float(dur_min.mean(skipna=True)), 1) if dur_min.notna().any() else 0
        d_min_i = idxmin_thresh(dur_min, 1.0) 
        d_max_i = dur_min.idxmax() if dur_min.notna().any() else np.nan
        d_min_v = (round(float(dur_min.loc[d_min_i]), 1) if d_min_i==d_min_i else "—")
        d_max_v = (round(float(dur_min.loc[d_max_i]), 1) if d_max_i==d_max_i else "—")
        st.divider()
        # ÉNERGIE
        st.markdown('#### ⚡ Énergie <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: card("Total (kWh)", f"{e_total}")
        with c2: card("Moyenne (kWh)", f"{e_mean}")
        with c3: card("Min (kWh) (≥4)", f"{e_min_v}", f"{date_of(e_min_i)} — {lieu_of(e_min_i)}")
        c4, c5, _ = st.columns(3)
        with c4: card("Max (kWh)", f"{e_max_v}", f"{date_of(e_max_i)} — {lieu_of(e_max_i)}")
        st.divider()    
        # Pmean
        st.markdown('#### 🔌 Puissance moyenne (kW) <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: card("Moyenne (kW)", f"{pm_mean}")
        with c2: card("Min (kW) (≥4)", f"{pm_min_v}", f"{date_of(pm_min_i)} — {lieu_of(pm_min_i)}")
        with c3: card("Max (kW)", f"{pm_max_v}", f"{date_of(pm_max_i)} — {lieu_of(pm_max_i)}")
        st.divider()
        # Pmax
        st.markdown('#### 🚀 Puissance maximale (kW) <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: card("Moyenne (kW)", f"{px_mean}")
        with c2: card("Min (kW) (≥4)", f"{px_min_v}", f"{date_of(px_min_i)} — {lieu_of(px_min_i)}")
        with c3: card("Max (kW)", f"{px_max_v}", f"{date_of(px_max_i)} — {lieu_of(px_max_i)}")
        st.divider()
        # SOC
        st.markdown('#### 🔋 SOC <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)
        if soc_s.notna().any() and soc_e.notna().any():
            soc_gain = soc_e - soc_s
            soc_gain_mean = round(float(soc_gain.mean(skipna=True)), 2)
        else:
            soc_gain_mean = "—"
        c1, c2, c3 = st.columns(3)
        with c1: card("SOC début moyen (%)", f"{soc_start_mean}")
        with c2: card("SOC fin moyen (%)", f"{soc_end_mean}")
        with c3: card("SOC moyen de recharge (%)", f"{soc_gain_mean}")
        st.divider()

        st.markdown('#### 🔋 Charges 900V <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)
        c900 = pd.to_numeric(sess["charge_900V"], errors="coerce").fillna(0).astype(int)

        total_900 = int(c900.sum())
        total_all = len(sess)
        pct_900   = round(total_900 / total_all * 100, 2) if total_all > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        with c2: card("Total charges 900V", f"{total_900}")
        with c1: card("Total charges", f"{total_all}")
        with c3: card("% en 900V", f"{pct_900}%")
        st.divider()
        # Durées
        st.markdown('#### ⏱️ Durées de charge (min) <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: card("Moyenne (min)", f"{d_mean}")
        with c2: card("Min (min)", f"{d_min_v}", f"{date_of(d_min_i)} — {lieu_of(d_min_i)}")
        with c3: card("Max (min)", f"{d_max_v}", f"{date_of(d_max_i)} — {lieu_of(d_max_i)}")

        # Charge par jour
        st.divider()
        st.markdown('#### 📅 Charges par jour <span class="kpi-tag">OK only</span>', unsafe_allow_html=True)

        d_site = tables.get("charges_daily_by_site", pd.DataFrame()).copy()

        if d_site.empty:
            st.info("Feuille 'charges_daily_by_site' absente (relance kpi_cal).")
        else:
            # garder seulement le périmètre de sites déjà filtré dans l'app (si présent)
            if "Site" in d_site.columns and site_sel:
                d_site = d_site[d_site["Site"].isin(site_sel)]

            # OK only + filtre dates
            d_site = d_site[d_site["Status"] == "OK"].copy()
            d_site["day_dt"] = pd.to_datetime(d_site["day"], errors="coerce")
            d_site = d_site.dropna(subset=["day_dt"])

            d1_day = pd.to_datetime(d1_ts).floor("D")
            d2_day = (pd.to_datetime(d2_ts) - pd.Timedelta(seconds=1)).floor("D")
            d_site = d_site[(d_site["day_dt"] >= d1_day) & (d_site["day_dt"] <= d2_day)]

            if d_site.empty:
                st.info("Aucune charge OK sur la période (après filtres).")
            else:
                # Total journalier global (tous sites confondus)
                daily_tot = (
                    d_site.groupby("day_dt", as_index=False)["Nb"].sum()
                        .sort_values("day_dt")
                )

                nb_days  = int(daily_tot["day_dt"].nunique())
                mean_day = round(float(daily_tot["Nb"].mean()), 2) if nb_days else 0.0
                med_day  = round(float(daily_tot["Nb"].median()), 2) if nb_days else 0.0

                c1, c2, c3 = st.columns(3)
                with c1: card("Jours couverts", f"{nb_days}")
                with c2: card("Moyenne / jour (OK)", f"{mean_day}")
                with c3: card("Médiane / jour (OK)", f"{med_day}")

                # Min / Max global 
                max_row = d_site.loc[d_site["Nb"].idxmax()]
                max_date = max_row["day_dt"]
                max_site = str(max_row["Site"])
                max_v = int(max_row["Nb"])
                min_row = d_site.loc[d_site["Nb"].idxmin()]
                min_date = min_row["day_dt"]
                min_site = str(min_row["Site"])
                min_v = int(min_row["Nb"])

                c4, c5, _ = st.columns(3)
                with c4:
                    card(
                        "Min / jour (OK)",
                        f"{int(min_v)}",  # ❌ min_v n'existe pas encore ici
                        f"{min_date.strftime('%Y-%m-%d')} — site: {min_site} ({max_site})"
                    )
                with c5:
                    card(
                        "Max / jour (OK)",
                        f"{max_v}",
                        f"{max_date.strftime('%Y-%m-%d')} — site: {max_site}"
                    )

        st.divider()
        st.subheader("Taux de réussite/échec par type de véhicule")
        charges_mac = tables.get("charges_mac", pd.DataFrame())
        if charges_mac.empty:
            st.info("Feuille 'charges_mac' absente.")
        else:
            dfv = charges_mac.copy()

            if "Datetime start" in dfv.columns:
                dfv["Datetime start"] = pd.to_datetime(dfv["Datetime start"], errors="coerce")

            site_col_v = "Site" if "Site" in dfv.columns else ("Name Project" if "Name Project" in dfv.columns else None)

            if site_col_v is None or "Datetime start" not in dfv.columns or "is_ok" not in dfv.columns:
                st.info("Colonnes requises manquantes dans 'charges_mac'.")
            else:
                dfv["is_ok"] = dfv["is_ok"].map(
                    lambda x: True if str(x).strip().lower() in {"1", "true", "vrai", "yes", "y"} else False
                )
                veh = dfv["Vehicle"].astype(str) if "Vehicle" in dfv.columns else pd.Series("", index=dfv.index, dtype="object")
                veh = veh.str.strip()
                veh = veh.replace({"": np.nan, "nan": np.nan, "none": np.nan, "NULL": np.nan}, regex=False)
                dfv["Vehicle"] = veh.fillna("Unknown")
                mask_v = (
                    dfv[site_col_v].isin(st.session_state.site_sel)
                    & dfv["Datetime start"].ge(pd.Timestamp(st.session_state.d1))
                    & dfv["Datetime start"].lt(pd.Timestamp(st.session_state.d2) + pd.Timedelta(days=1))
                )
                dfv = dfv.loc[mask_v].copy()
                dfv = dfv[dfv["Vehicle"] != "Unknown"]

                if dfv.empty:
                    st.info("Aucune donnée Vehicle (hors Unknown) sur ce périmètre.")
                else:
                    g = (
                        dfv.groupby("Vehicle", dropna=False)["is_ok"]
                        .agg(total="size", ok="sum")
                        .reset_index()
                    )
                    g["nok"] = g["total"] - g["ok"]
                    g["% Réussite"] = np.where(g["total"].gt(0), (g["ok"] / g["total"] * 100).round(2), 0.0)
                    g["% Échec"] = 100 - g["% Réussite"]
                    g = g.sort_values(["total", "% Réussite"], ascending=[False, False]).reset_index(drop=True)
                    st.dataframe(g, use_container_width=True)
                    fig2 = px.bar(
                        g, x="Vehicle", y="% Réussite",
                        labels={"% Réussite": "% Réussite", "Vehicle": "Vehicle"},
                        title="Taux de réussite par type de Vehicle (%)",
                        color="% Réussite", color_continuous_scale="GnBu"
                    )
                    fig2.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
                    fig2.update_layout(coloraxis_showscale=False, xaxis=dict(type="category"), yaxis=dict(range=[0, 100]))
                    plot(fig2, "tab4_Vehicle_success_no_unknown")
        st.divider()
        st.subheader("⏱️ Durée de fonctionnement totale (heures)")

        dsd = tables.get("durations_site_daily", pd.DataFrame()).copy()
        dpd = tables.get("durations_pdc_daily",  pd.DataFrame()).copy()

        if dsd.empty and dpd.empty:
            st.info("Tables pré-calculées absentes : 'durations_site_daily' / 'durations_pdc_daily'.")
        else:
            d1_day = pd.to_datetime(d1_ts).floor("D")
            d2_day = (pd.to_datetime(d2_ts) - pd.Timedelta(seconds=1)).floor("D")
            if "day" in dsd.columns: dsd["day"] = pd.to_datetime(dsd["day"], errors="coerce")
            if "day" in dpd.columns: dpd["day"] = pd.to_datetime(dpd["day"], errors="coerce")

            # PAR SITE
            if not dsd.empty:
                m_site = dsd["Site"].isin(site_sel) & dsd["day"].ge(d1_day) & dsd["day"].le(d2_day)
                by_site = (
                    dsd.loc[m_site]
                    .groupby("Site", as_index=False)["dur_min"].sum()
                    .assign(Heures=lambda d: (d["dur_min"] / 60).round(1))
                    [["Site", "Heures"]]
                    .sort_values("Heures", ascending=False)  # plus élevé → plus bas
                    .reset_index(drop=True)
                )
                st.dataframe(by_site, use_container_width=True)

                # couleurs distinctes
                import plotly.express as px
                palette = px.colors.qualitative.D3 + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
                cats_sites = by_site["Site"].tolist()
                color_map_sites = {s: palette[i % len(palette)] for i, s in enumerate(cats_sites)}

                h_min, h_max = float(by_site["Heures"].min()), float(by_site["Heures"].max())
                site_text = [f"{h} {'🔺' if h==h_max else ('🔻' if h==h_min else '')}" for h in by_site["Heures"]]

                fig_site = px.bar(
                    by_site, x="Heures", y="Site",
                    orientation="h", text=site_text,
                    color="Site", color_discrete_map=color_map_sites,
                    title="Durée totale par site (heures)",
                )
                fig_site.update_traces(textposition="outside")
                fig_site.update_layout(
                    showlegend=False,
                    yaxis=dict(categoryorder="array", categoryarray=cats_sites), 
                )
                plot(fig_site, "dur_site_precalc")

            if not dpd.empty:
                m_pdc_all = dpd["Site"].isin(site_sel) & dpd["day"].ge(d1_day) & dpd["day"].le(d2_day)
                by_pdc_all = (
                    dpd.loc[m_pdc_all]
                    .groupby(["Site", "PDC"], as_index=False)["dur_min"].sum()
                    .assign(Heures=lambda d: (d["dur_min"] / 60).round(1))
                    [["Site", "PDC", "Heures"]]
                )
        st.divider()
        if not dpd.empty:
            m_pdc_all = dpd["Site"].isin(site_sel) & dpd["day"].ge(d1_day) & dpd["day"].le(d2_day)
            by_pdc_all = (
                dpd.loc[m_pdc_all]
                .groupby(["Site", "PDC"], as_index=False)["dur_min"].sum()
                .assign(Heures=lambda d: (d["dur_min"] / 60).round(1))
                [["Site", "PDC", "Heures"]]
            )
            sites_opts = sorted(by_pdc_all["Site"].dropna().unique().tolist())
            if sites_opts:
                site_focus = st.selectbox("🏢 Site", options=sites_opts, index=0, key="dur_site_sel_tab4_precalc")

                bp = (
                    by_pdc_all[by_pdc_all["Site"] == site_focus]
                    .drop(columns=["Site"])
                    .sort_values("Heures", ascending=True) 
                    .reset_index(drop=True)
                )
                st.dataframe(bp, use_container_width=True)

                if not bp.empty:
                    import plotly.graph_objects as go
                    import plotly.express as px

                    # palette franche (pas de fade)
                    palette = px.colors.qualitative.D3 + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly

                    h_min, h_max = float(bp["Heures"].min()), float(bp["Heures"].max())

                    fig_pdc = go.Figure()
                    for i, row in bp.iterrows():
                        pdc = str(row["PDC"])
                        h   = float(row["Heures"])
                        txt = f"{h} {'🔺' if h==h_max else ('🔻' if h==h_min else '')}"
                        fig_pdc.add_bar(
                            x=[h], y=[pdc], orientation="h",
                            marker=dict(color=palette[i % len(palette)]),  # pas de line => pas de contour
                            text=[txt], textposition="outside",
                            name=pdc, showlegend=False
                        )

                    fig_pdc.update_traces(marker_line_width=0)

                    cats_pdc = bp["PDC"].astype(str).tolist()
                    fig_pdc.update_layout(
                        title=f"Durée totale par PDC — {site_focus} (heures)",
                        yaxis=dict(categoryorder="array", categoryarray=cats_pdc),
                        plot_bgcolor="#ffffff",
                        paper_bgcolor="#ffffff",
                        bargap=0.15
                    )
                    if hasattr(fig_pdc.layout, "coloraxis"):
                        fig_pdc.layout.coloraxis = None
                    plot(fig_pdc, f"dur_pdc_precalc_{site_focus}") 



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

