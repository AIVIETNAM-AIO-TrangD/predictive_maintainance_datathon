"""PCP well failure-risk dashboard (NLinear v1 model).

Run from the workspace root:
    streamlit run dashboard/app.py

Reads (all produced by the pipeline in src/):
    outputs/forecast_fail_next_30d.parquet        risk per well for the next 30 days  (src/predict.py --forecast)
    outputs/predictions_fail_next_30d_v1.parquet  daily risk history, Apr-Oct 2026    (src/predict.py --start 2026-04-01)
    outputs/drivers_channels_v1.parquet           per-well feature contributions       (src/explain.py)
    outputs/drivers_groups_v1.parquet             per-day factor-group contributions   (src/explain.py)
    outputs/evaluation/*.csv                      test-period model evaluation         (src/evaluate.py)
    data/processed/{wells,dashboard,failure_events,jobs_clean}.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
OUT, DATA, FIG = ROOT / "outputs", ROOT / "data" / "processed", ROOT / "reports" / "figures"
BUNDLE = Path(__file__).resolve().parent / "bundle"      # slim copy for deployment (src/make_dashboard_bundle.py)
if not (OUT / "forecast_fail_next_30d.parquet").exists() and (BUNDLE / "forecast_fail_next_30d.parquet").exists():
    OUT = DATA = BUNDLE
MODEL = "v1"

BLUE, RED, GRAY, INK, MUTED = "#2a78d6", "#e34948", "#9a9893", "#0b0b0b", "#52514e"
TIER_COLORS = {"high": "#e34948", "medium": "#eda100", "low": "#9a9893"}
BASE_RATE = 0.131   # share of producing well-days followed by a failure within 30 days (test period)

LABELS = {
    "hours_on_prod": "Hours on production", "hours_down": "Hours down", "oil_produced_m3": "Oil rate (m3/d)",
    "water_produced_m3": "Water rate (m3/d)", "gas_produced_e3m3": "Gas rate", "tubing_pressure_kpa": "Tubing pressure",
    "casing_pressure_kpa": "Casing pressure", "pump_efficiency_pct": "Pump efficiency", "pcp_pump_rpm": "Pump RPM",
    "pcp_pump_rod_torque_ftlb": "Rod torque", "pcp_pump_electric_current_a": "Motor current (A)",
    "pcp_pump_volume_throughput_m3d": "Pump throughput", "pcp_pump_displacement_m3d100rpm": "Pump displacement",
    "water_cut_7d": "Water cut (7d)", "downtime_days_30d": "Downtime days (30d)",
    "fail_count_prior": "Failures since mid-2025", "fail_count_90d": "Failures (90d)", "fail_count_365d": "Failures (365d)",
    "fail_count_prior_rods": "Rod failures", "fail_count_prior_pump": "Pump failures",
    "fail_count_prior_tubing": "Tubing failures", "days_since_last_failure": "Days since last failure",
    "job_count_prior": "Jobs since 2021", "job_count_90d": "Jobs (90d)", "job_count_365d": "Jobs (365d)",
    "job_count_prior_rod_truck": "Rod-truck jobs", "job_count_prior_workover": "Workovers",
    "job_count_prior_pump_change": "Pump changes", "days_since_last_job": "Days since last job",
    "days_since_last_rod_truck": "Days since last rod-truck job", "days_since_last_workover": "Days since last workover",
    "days_since_last_pump_change": "Days since last pump change", "well_age_years": "Well age (years)",
    "surface_latitude": "Latitude", "surface_longitude": "Longitude",
    "field_operating_area": "Field", "pcp_pump_make": "Pump make",
}


def label(c: str) -> str:
    if c in LABELS:
        return LABELS[c]
    for k, v in {"_std_7d": " variability (7d)", "_3v30": " change (3d vs 30d)", "_7v30": " change (7d vs 30d)",
                 "_30d": " (30d)"}.items():
        if c.endswith(k):
            base = c[: -len(k)]
            base = {"has_amps": "Amps readings", "has_torque": "Torque readings", "has_eff": "Efficiency readings"}.get(base, base)
            return base.replace("_", " ").capitalize() + v
    return c.replace("_", " ").capitalize()


# ------------------------------------------------------------------------------------------ data
@st.cache_data
def load_forecast() -> pd.DataFrame:
    f = pd.read_parquet(OUT / "forecast_fail_next_30d.parquet")
    w = pd.read_parquet(DATA / "wells.parquet", columns=["uwi", "surface_latitude", "surface_longitude",
                                                          "pcp_pump_make", "pcp_pump_model"])
    f = f.merge(w, on="uwi", how="left")
    ch = pd.read_parquet(OUT / f"drivers_channels_{MODEL}.parquet")
    top = (ch[ch.contribution > 0].sort_values("contribution", ascending=False).groupby("uwi").head(1)
           .set_index("uwi"))
    f["top_driver"] = f.uwi.map(top.channel.map(label))
    hist = ch[ch.channel.isin(["fail_count_365d", "days_since_last_failure"])].pivot(index="uwi", columns="channel",
                                                                                     values="value")
    f = f.join(hist, on="uwi")
    return f


@st.cache_data
def load_drivers() -> pd.DataFrame:
    return pd.read_parquet(OUT / f"drivers_channels_{MODEL}.parquet")


@st.cache_data
def load_history() -> pd.DataFrame:
    return pd.read_parquet(OUT / f"predictions_fail_next_30d_{MODEL}.parquet",
                           columns=["uwi", "production_date", "fail_next_30d_prob_cal", "fail_next_30d",
                                    "incomplete_reporting_day"])


@st.cache_data
def load_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = pd.read_parquet(DATA / "failure_events.parquet", columns=["uwi", "event_start", "event_end", "failure_group",
                                                                    "cause_group", "repair_job_category"])
    jobs = pd.read_parquet(DATA / "jobs_clean.parquet", columns=["uwi", "start_date", "end_date", "job_category",
                                                                  "job_type", "job_subtype", "job_duration_days"])
    return ev, jobs


@st.cache_data
def load_ops(uwi: str) -> pd.DataFrame:
    cols = ["production_date", "oil_produced_m3", "water_produced_m3", "hours_on_prod", "hours_down",
            "pcp_pump_rpm", "pcp_pump_rod_torque_ftlb", "pump_efficiency_pct", "downtime_code_1"]
    return pd.read_parquet(DATA / "dashboard.parquet", columns=cols, filters=[("uwi", "==", uwi)]).sort_values("production_date")


@st.cache_data
def load_eval() -> dict[str, pd.DataFrame]:
    d = OUT / "evaluation"
    return {k: pd.read_csv(d / f"{k}.csv").query("model == @MODEL") for k in ("thresholds", "tiers", "topk", "policies")}


@st.cache_data
def load_survival() -> dict[str, pd.DataFrame] | None:
    d = OUT / "survival"
    if not (d / "summary.csv").exists():
        return None
    return {"summary": pd.read_csv(d / "summary.csv"), "km": pd.read_parquet(d / "km_curves.parquet"),
            "wells": pd.read_parquet(d / "current_wells.parquet")}


SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]   # categorical order, never cycled


def style(fig, height=None):
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=60, b=10), font=dict(color=INK),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                      **({"height": height} if height else {}))
    fig.update_xaxes(gridcolor="#eceae6"); fig.update_yaxes(gridcolor="#eceae6")
    return fig


# ------------------------------------------------------------------------------------------ app
st.set_page_config(page_title="PCP Failure Risk", layout="wide")

REQUIRED = [OUT / "forecast_fail_next_30d.parquet", OUT / f"predictions_fail_next_30d_{MODEL}.parquet",
            OUT / f"drivers_channels_{MODEL}.parquet",
            OUT / "evaluation" / "thresholds.csv", DATA / "wells.parquet", DATA / "dashboard.parquet",
            DATA / "failure_events.parquet", DATA / "jobs_clean.parquet"]
missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
if missing:
    st.title("PCP failure risk")
    st.error("The dashboard's data files are not available here, so there is nothing to show yet.")
    st.markdown("Missing files:\n" + "\n".join(f"- `{m}`" for m in missing))
    st.markdown("These files are produced by the pipeline in `src/`. For a deployment, build the slim copy with "
                "`python src/make_dashboard_bundle.py` (writes `dashboard/bundle/`).")
    st.stop()

fc = load_forecast()
anchor = fc.production_date.max()
win_start, win_end = fc.forecast_window_start.max(), fc.forecast_window_end.max()

with st.sidebar:
    st.title("PCP failure risk")
    st.caption(f"Model: NLinear ({MODEL}). Scored on **{anchor:%Y-%m-%d}** (last fully reported day) for failures "
               f"starting **{win_start:%b %d} – {win_end:%b %d, %Y}**.")
    fields = sorted(fc.field_operating_area.dropna().unique())
    sel_fields = st.multiselect("Field", fields, placeholder="All fields")
    sel_tiers = st.multiselect("Risk tier", ["high", "medium", "low"], default=["high", "medium", "low"])
    only_act = st.checkbox("Only wells worth acting on (expected value > 0)")
    with st.expander("Cost assumptions"):
        st.caption("Expected value of acting = P(failure) × failure cost − intervention cost.")
        cost_fn = st.number_input("Cost of a failure", value=5.0, min_value=0.0, step=1.0)
        cost_fp = st.number_input("Cost of an intervention", value=1.0, min_value=0.0, step=0.5)
        oil_value = st.number_input("Value of lost oil per m3", value=0.0, min_value=0.0, step=10.0)
        downtime_days = st.number_input("Days lost per failure", value=3.6, min_value=0.0, step=0.5)
        st.caption("Defaults are relative units (a failure costs 5× an intervention). Enter currency to see $.")

fc = fc.copy()
fc["failure_cost"] = cost_fn + oil_value * downtime_days * fc.oil_rate_30d.fillna(0)
fc["expected_value"] = fc.fail_next_30d_prob_cal * fc.failure_cost - cost_fp
fc["priority_rank"] = fc.expected_value.rank(ascending=False, method="first").astype(int)
view = fc[fc.risk_tier.isin(sel_tiers)]
if sel_fields:
    view = view[view.field_operating_area.isin(sel_fields)]
if only_act:
    view = view[view.expected_value > 0]
view = view.sort_values("priority_rank")

st.title(f"Failure risk forecast: {win_start:%b %d} – {win_end:%b %d, %Y}")
st.caption(f"Probability that each producing PCP well has a failure starting between {win_start:%Y-%m-%d} and "
           f"{win_end:%Y-%m-%d}, using data up to {anchor:%Y-%m-%d}. The data runs to Oct 27, but Oct 26–27 are only "
           f"partly reported (most wells show zero volumes), so they are not used as model input.")

tab_fleet, tab_well, tab_factors, tab_life, tab_model = st.tabs(
    ["Fleet risk", "Well detail", "Risk factors", "Run life", "Model performance"])
surv = load_survival()

# ------------------------------------------------------------------------------------------ fleet
with tab_fleet:
    c = st.columns(5)
    c[0].metric("Wells shown", f"{len(view):,}", help="Producing wells with a clean 30-day history on the scoring day")
    c[1].metric("High-risk tier", f"{(view.risk_tier == 'high').sum():,}",
                help="Top 5% of validation risk. On the test period 47% of high-tier days were followed by a failure.")
    c[2].metric("Worth acting on", f"{(view.expected_value > 0).sum():,}", help="Expected value of an intervention > 0")
    c[3].metric("Expected failures (30 days)", f"{view.fail_next_30d_prob_cal.sum():,.0f}",
                help="Sum of calibrated failure probabilities")
    c[4].metric("Average risk", f"{view.fail_next_30d_prob_cal.mean():.0%}" if len(view) else "–",
                help=f"Fleet base rate on the test period: {BASE_RATE:.0%}")

    left, right = st.columns([3, 2])
    with left:
        m = view.dropna(subset=["surface_latitude", "surface_longitude"]).sort_values("fail_next_30d_prob_cal")
        fig = px.scatter_map(m, lat="surface_latitude", lon="surface_longitude", color="fail_next_30d_prob_cal",
                             color_continuous_scale="Reds", range_color=(0, max(0.5, m.fail_next_30d_prob_cal.max())),
                             hover_name="formatted_uwi", zoom=6, map_style="carto-positron",
                             hover_data={"field_operating_area": True, "risk_tier": True, "top_driver": True,
                                         "fail_next_30d_prob_cal": ":.0%", "surface_latitude": False,
                                         "surface_longitude": False},
                             labels={"fail_next_30d_prob_cal": "Failure risk", "field_operating_area": "Field",
                                     "risk_tier": "Tier", "top_driver": "Top driver"})
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(title="Where the risk is (each dot is a well)", coloraxis_colorbar=dict(tickformat=".0%"))
        st.plotly_chart(style(fig, 460), width="stretch")
    with right:
        by_field = (view.groupby("field_operating_area")
                    .agg(wells=("uwi", "size"), act=("expected_value", lambda s: (s > 0).sum()),
                         risk=("fail_next_30d_prob_cal", "mean"))
                    .query("wells >= 5").sort_values("act").tail(12))
        fig = go.Figure(go.Bar(x=by_field.act, y=by_field.index, orientation="h", marker_color=BLUE,
                               text=[f"{a} of {w} wells · avg risk {r:.0%}" for a, w, r in
                                     zip(by_field.act, by_field.wells, by_field.risk)],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x} wells worth acting on<extra></extra>"))
        fig.update_layout(title="Wells worth acting on, by field", xaxis_title="wells",
                          xaxis_range=[0, max(1, by_field.act.max()) * 2.4])
        st.plotly_chart(style(fig, 460), width="stretch")

    st.subheader("Priority list")
    st.caption("Ranked by expected value of acting (calibrated risk × failure cost − intervention cost). "
               "Top driver = the input that raises this well's risk the most.")
    table = view[["priority_rank", "formatted_uwi", "field_operating_area", "facility_name", "fail_next_30d_prob_cal",
                  "risk_tier", "expected_value", "top_driver", "fail_count_365d", "days_since_last_failure",
                  "oil_rate_30d", "water_produced_m3", "hours_down"]]
    st.dataframe(table, hide_index=True, width="stretch", height=420, column_config={
        "priority_rank": st.column_config.NumberColumn("Priority", format="%d"),
        "formatted_uwi": "Well", "field_operating_area": "Field", "facility_name": "Facility",
        "fail_next_30d_prob_cal": st.column_config.ProgressColumn("Failure risk (30d)", format="percent", min_value=0,
                                                                   max_value=1),
        "risk_tier": "Tier", "expected_value": st.column_config.NumberColumn("Expected value", format="%.2f"),
        "top_driver": "Top driver", "fail_count_365d": st.column_config.NumberColumn("Failures (365d)", format="%d"),
        "days_since_last_failure": st.column_config.NumberColumn("Days since failure", format="%d",
                                                                 help="730 = none recorded / 2+ years"),
        "oil_rate_30d": st.column_config.NumberColumn("Oil m3/day", format="%.1f"),
        "water_produced_m3": st.column_config.NumberColumn("Water m3", format="%.0f"),
        "hours_down": st.column_config.NumberColumn("Hours down", format="%.0f")})
    st.download_button("Download list (CSV)", table.to_csv(index=False).encode(), "priority_list.csv", "text/csv")

# ------------------------------------------------------------------------------------------ well
with tab_well:
    options = view.uwi.tolist() or fc.sort_values("priority_rank").uwi.tolist()
    names = fc.set_index("uwi")
    choices = {f"#{names.priority_rank[u]} · {names.formatted_uwi[u]} · {names.field_operating_area[u]} · "
               f"{names.fail_next_30d_prob_cal[u]:.0%} risk": u for u in options}
    uwi = choices[st.selectbox("Well", list(choices))]
    w = names.loc[uwi]
    ev, jobs = load_events()
    wev, wjobs = ev[ev.uwi == uwi].sort_values("event_start"), jobs[jobs.uwi == uwi].sort_values("start_date")

    c = st.columns(6)
    c[0].metric("Failure risk (30d)", f"{w.fail_next_30d_prob_cal:.0%}", help=f"Fleet base rate {BASE_RATE:.0%}")
    c[1].metric("Tier", w.risk_tier.capitalize())
    c[2].metric("Priority", f"#{w.priority_rank}")
    c[3].metric("Failures since mid-2025", f"{len(wev)}")
    c[4].metric("Last failure", f"{wev.event_start.max():%Y-%m-%d}" if len(wev) else "none recorded")
    c[5].metric("Pump", f"{w.pcp_pump_make or '–'}", help=f"Model: {w.pcp_pump_model}")
    life = ""
    if surv is not None and uwi in set(surv["wells"].uwi):
        sw = surv["wells"].set_index("uwi").loc[uwi]
        if pd.notna(sw.run_days_since_repair):
            life = (f" · Running **{sw.run_days_since_repair:.0f} producing days** since the last repair"
                    + (f"; typical chance of failing in the next 30 producing days for a run this old in "
                       f"{sw.km_curve_used}: **{sw.km_p_fail_30d:.0%}**" if pd.notna(sw.km_p_fail_30d) else ""))
    st.caption(f"Field: {w.field_operating_area} · Facility: {w.facility_name} · Status: {w.well_daily_status}{life}")

    ch = load_drivers()
    left, right = st.columns([2, 3])
    with left:
        d = ch[ch.uwi == uwi].copy()
        d["name"] = np.where(d.category.notna(), d.channel.map(label) + ": " + d.category.astype(str), d.channel.map(label))
        d = d.reindex(d.contribution.abs().sort_values(ascending=False).index).head(12)[::-1]
        hover = [f"value {v:,.2f} (typical {t:,.2f})" if pd.notna(v) else "" for v, t in zip(d.value, d.training_median)]
        fig = go.Figure(go.Bar(x=d.contribution, y=d.name, orientation="h",
                               marker_color=np.where(d.contribution > 0, RED, BLUE), customdata=hover,
                               hovertemplate="%{y}<br>%{customdata}<br>effect %{x:+.2f} log-odds<extra></extra>"))
        fig.update_layout(title="Why this well: top drivers (red raises risk, blue lowers it)",
                          xaxis_title="effect on risk vs a typical well (log-odds)")
        st.plotly_chart(style(fig, 440), width="stretch")
    with right:
        h = load_history()
        h = h[(h.uwi == uwi) & ~h.incomplete_reporting_day]
        fig = go.Figure(go.Scatter(x=h.production_date, y=h.fail_next_30d_prob_cal, mode="lines", line=dict(color=BLUE),
                                   name="risk", hovertemplate="%{x|%Y-%m-%d}: %{y:.0%}<extra></extra>"))
        fig.add_hline(y=BASE_RATE, line=dict(color=GRAY, dash="dash", width=1),
                      annotation_text="fleet base rate", annotation_position="top left")
        for _, e in wev[wev.event_end >= h.production_date.min()].iterrows() if len(h) else []:
            fig.add_vrect(x0=e.event_start, x1=e.event_end + pd.Timedelta(days=1), fillcolor=RED, opacity=0.25,
                          line_width=0)
        fig.update_layout(title="Predicted 30-day failure risk over time (red bands = actual failures)",
                          yaxis_tickformat=".0%", yaxis_range=[0, max(0.6, h.fail_next_30d_prob_cal.max() * 1.1)],
                          showlegend=False)
        st.plotly_chart(style(fig, 440), width="stretch")

    st.subheader("Operations")
    ops = load_ops(uwi)
    days = st.slider("Days of history", 60, 720, 240, step=30)
    ops = ops[ops.production_date >= ops.production_date.max() - pd.Timedelta(days=days)]
    st.caption("Red bands = failures · orange triangles = jobs (hover for details). Hours bars show the downtime code on hover.")
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        subplot_titles=["Oil (m3/day)", "Water (m3/day)", "Hours on production", "Pump RPM",
                                        "Rod torque (ft-lb)"])
    line = dict(color=BLUE, width=1.5)
    fig.add_trace(go.Scatter(x=ops.production_date, y=ops.oil_produced_m3, name="oil", line=line), 1, 1)
    fig.add_trace(go.Scatter(x=ops.production_date, y=ops.water_produced_m3, name="water", line=line), 2, 1)
    fig.add_trace(go.Bar(x=ops.production_date, y=ops.hours_on_prod, name="hours on", marker_color=GRAY,
                         customdata=ops.downtime_code_1.fillna(""),
                         hovertemplate="%{x|%Y-%m-%d}: %{y} h<br>%{customdata}<extra></extra>"), 3, 1)
    fig.add_trace(go.Scatter(x=ops.production_date, y=ops.pcp_pump_rpm, name="rpm", line=line), 4, 1)
    fig.add_trace(go.Scatter(x=ops.production_date, y=ops.pcp_pump_rod_torque_ftlb, name="torque", mode="lines+markers",
                             marker=dict(size=4), line=line), 5, 1)
    lo = ops.production_date.min() if len(ops) else pd.Timestamp("2026-01-01")
    for _, e in wev[wev.event_end >= lo].iterrows():
        fig.add_vrect(x0=e.event_start, x1=e.event_end + pd.Timedelta(days=1), fillcolor=RED, opacity=0.2, line_width=0)
    jj = wjobs[wjobs.start_date >= lo]
    if len(jj):
        fig.add_trace(go.Scatter(x=jj.start_date, y=[0] * len(jj), mode="markers", name="job",
                                 marker=dict(symbol="triangle-up", size=11, color="#eda100", line=dict(color="white", width=1)),
                                 customdata=jj.job_category + " · " + jj.job_type.fillna(""),
                                 hovertemplate="%{x|%Y-%m-%d}<br>%{customdata}<extra></extra>"), 3, 1)
    fig.update_layout(height=860, showlegend=False)
    st.plotly_chart(style(fig), width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("**Failure history**")
        st.dataframe(wev[["event_start", "event_end", "failure_group", "cause_group", "repair_job_category"]]
                     .sort_values("event_start", ascending=False)
                     .assign(event_start=lambda x: x.event_start.dt.date, event_end=lambda x: x.event_end.dt.date)
                     .rename(columns={"event_start": "Start", "event_end": "End", "failure_group": "Component",
                                      "cause_group": "Cause", "repair_job_category": "Repair"}),
                     hide_index=True, width="stretch")
    with b:
        st.markdown("**Jobs (most recent first)**")
        st.dataframe(wjobs[["start_date", "job_category", "job_type", "job_subtype", "job_duration_days"]]
                     .sort_values("start_date", ascending=False).head(30)
                     .assign(start_date=lambda x: x.start_date.dt.date, job_duration_days=lambda x: x.job_duration_days.round(1))
                     .rename(columns={"start_date": "Start", "job_category": "Category", "job_type": "Type",
                                      "job_subtype": "Subtype", "job_duration_days": "Days"}),
                     hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------------ factors
with tab_factors:
    ch = load_drivers()
    st.markdown("What pushes risk up across the fleet, from the model's own contributions on the scoring day. "
                "Each contribution compares a well with a typical well (training median); positive = higher risk.")
    tiers = fc.set_index("uwi").risk_tier
    d = ch.assign(tier=ch.uwi.map(tiers))
    g = d.groupby(["group", "uwi", "tier"], as_index=False).contribution.sum()
    gm = g.assign(segment=np.where(g.tier == "high", "high-risk wells", "other wells")).groupby(
        ["group", "segment"]).contribution.mean().unstack()
    gm = gm.sort_values("high-risk wells")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=gm.index, x=gm["high-risk wells"], orientation="h", name="high-risk wells", marker_color=RED))
        fig.add_trace(go.Bar(y=gm.index, x=gm["other wells"], orientation="h", name="other wells", marker_color=GRAY))
        fig.update_layout(title="Average effect by factor group", xaxis_title="effect on risk (log-odds)", barmode="group")
        st.plotly_chart(style(fig, 460), width="stretch")
    with right:
        hi = d[d.tier == "high"]
        top = hi.groupby("channel").contribution.mean().sort_values().tail(12)
        fig = go.Figure(go.Bar(x=top.values, y=[label(c) for c in top.index], orientation="h", marker_color=RED,
                               hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        fig.update_layout(title="Strongest risk drivers among high-risk wells", xaxis_title="average effect (log-odds)")
        st.plotly_chart(style(fig, 460), width="stretch")

    st.markdown("**How often each factor group is a well's biggest risk driver** (wells worth acting on)")
    act = fc[fc.expected_value > 0].uwi
    tg = g[g.uwi.isin(act)].sort_values("contribution", ascending=False).groupby("uwi").head(1).group.value_counts()
    fig = go.Figure(go.Bar(x=tg.values, y=tg.index, orientation="h", marker_color=BLUE, text=tg.values,
                           textposition="outside", cliponaxis=False))
    fig.update_layout(xaxis_title="wells", yaxis=dict(autorange="reversed"), xaxis_range=[0, tg.max() * 1.15])
    st.plotly_chart(style(fig, 320), width="stretch")

    st.subheader("Exploratory analysis (training history)")
    st.markdown(
        "- **Past failures and repairs are the strongest signal.** Wells with 4+ failures in the past year fail within "
        "30 days 31% of the time vs 7% with none; a rod-truck job in the last 30 days: 30% vs 5% never.\n"
        "- **Higher RPM and water cut raise risk** (RPM: 5% in the lowest decile to 23% in the highest; >99% water cut: 21%).\n"
        "- **Falling pump efficiency is only a weak warning** (about 2% below normal in the month before a failure).\n"
        "- **Winter field** has 4.3 failures per well-year vs 1.9 overall.")
    for f, cap in [("04_history.png", "Failure rate vs failure and repair history"),
                   ("02_operating_hypotheses.png", "Failure rate across operating conditions"),
                   ("06_field_and_make.png", "Failures per well-year by field and pump make")]:
        if (FIG / f).exists():
            st.image(str(FIG / f), caption=cap)

# ------------------------------------------------------------------------------------------ run life
with tab_life:
    if surv is None:
        st.info("Run-life results are not available. Build them with `python src/survival.py`.")
    else:
        sm, km = surv["summary"], surv["km"]
        allr = sm[sm.dimension == "all"].iloc[0]
        st.markdown(
            "How long wells keep producing after a repair before the next failure, 2021–2026 (Kaplan–Meier). "
            "Run length counts **producing days** only. Failures before June 2025 are reconstructed from repair jobs "
            "(83% of repair jobs match a recorded failure), so treat these as approximate.")
        rep = sm[sm.dimension == "repaired"].set_index("group")
        rpm = sm[sm.dimension == "rpm_band"].set_index("group")
        c = st.columns(4)
        c[0].metric("Median run life", f"{allr.median_run_days:.0f} days",
                    help=f"{int(allr.runs):,} runs, {int(allr.wells)} wells")
        c[1].metric("Still running after 180 days", f"{allr.survive_180d:.0%}")
        if {"Rods", "Pump"} <= set(rep.index):
            c[2].metric("After a rod vs a pump repair",
                        f"{rep.median_run_days['Rods']:.0f} vs {rep.median_run_days['Pump']:.0f} days",
                        help="Median run life depending on what was repaired at the start of the run")
        if {"<150", "350+"} <= set(rpm.index):
            c[3].metric("RPM < 150 vs 350+",
                        f"{rpm.median_run_days['<150']:.0f} vs {rpm.median_run_days['350+']:.0f} days",
                        help="Median run life by pump speed in the first 30 days of the run")

        dims = {"Field": "field", "Pump RPM (first 30 days)": "rpm_band", "Water cut (first 30 days)": "water_cut_band",
                "Pump make": "pump_make", "Component repaired": "repaired"}
        dim = dims[st.radio("Compare by", list(dims), horizontal=True)]
        order = {"rpm_band": ["<150", "150-250", "250-350", "350+"],
                 "water_cut_band": ["<80%", "80-95%", "95-99%", "99%+"]}.get(dim)
        g = sm[sm.dimension == dim]
        names = [x for x in order if x in set(g.group)] if order else g.nlargest(6, "runs").group.tolist()
        fig = go.Figure()
        ref = km[(km.dimension == "all") & (km.t <= 365)]
        fig.add_trace(go.Scatter(x=ref.t, y=ref.survival, name="all runs",
                                 line=dict(color=GRAY, dash="dash", width=1.5, shape="hv")))
        for k, name in enumerate(names):
            cv = km[(km.dimension == dim) & (km.group == name) & (km.t <= 365)]
            med = g.set_index("group").median_run_days[name]
            fig.add_trace(go.Scatter(x=cv.t, y=cv.survival, name=f"{name} (median {med:.0f} d)",
                                     line=dict(color=SERIES[k], width=2, shape="hv"),
                                     hovertemplate=f"{name}<br>day %{{x}}: %{{y:.0%}} still running<extra></extra>"))
        fig.add_hline(y=0.5, line=dict(color=GRAY, width=1, dash="dot"))
        fig.update_layout(title="Share of runs still going, by producing days since the last repair",
                          xaxis_title="producing days since last repair", yaxis_tickformat=".0%", yaxis_range=[0, 1.02])
        style(fig, 460)
        fig.update_layout(legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.99))
        st.plotly_chart(fig, width="stretch")
        tbl = g.set_index("group").loc[names].reset_index()
        st.dataframe(tbl[["group", "runs", "failures", "wells", "median_run_days", "survive_30d", "survive_90d",
                          "survive_180d", "survive_365d"]], hide_index=True, width="stretch", column_config={
            "group": "Group", "median_run_days": st.column_config.NumberColumn("Median run (days)", format="%d"),
            **{f"survive_{x}d": st.column_config.NumberColumn(f"Running after {x} d", format="percent")
               for x in (30, 90, 180, 365)}})

        st.subheader("Wells most due for a failure by run life")
        st.caption("Run-life view of risk: chance of a failure in the next 30 producing days for a run this old in the "
                   "well's field, next to the model's 30-day risk. Wells running longer than almost any observed run "
                   "have no run-life estimate. Filtered by the sidebar.")
        cw = surv["wells"].merge(view[["uwi", "formatted_uwi", "fail_next_30d_prob_cal", "risk_tier"]], on="uwi")
        cw = cw.dropna(subset=["km_p_fail_30d"]).sort_values("km_p_fail_30d", ascending=False)
        st.dataframe(cw[["formatted_uwi", "field_operating_area", "pump_make", "run_days_since_repair", "km_p_fail_30d",
                         "fail_next_30d_prob_cal", "risk_tier"]].head(200), hide_index=True, width="stretch", height=380,
                     column_config={
                         "formatted_uwi": "Well", "field_operating_area": "Field", "pump_make": "Pump make",
                         "run_days_since_repair": st.column_config.NumberColumn("Producing days since repair", format="%d"),
                         "km_p_fail_30d": st.column_config.NumberColumn("Run-life chance (30d)", format="percent"),
                         "fail_next_30d_prob_cal": st.column_config.NumberColumn("Model risk (30d)", format="percent"),
                         "risk_tier": "Tier"})

# ------------------------------------------------------------------------------------------ model
with tab_model:
    ev_ = load_eval()
    st.markdown("Evaluated on **2026-06-27 onward**, a period never used for training, threshold choice or tier "
                "cut-offs. *Held-out wells* are 237 wells the model never saw.")
    th = ev_["thresholds"]
    th = th[th.threshold == "val cost-optimal"][["group", "value", "precision", "recall", "f1", "pr_auc", "roc_auc",
                                                 "tn", "fp", "fn", "tp"]]
    st.markdown(f"**At the cost-optimal threshold ({th.value.iloc[0]:.3f}, chosen on validation)**")
    st.dataframe(th.drop(columns="value").rename(columns={"group": "Wells", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}),
                 hide_index=True, width="stretch", column_config={
                     c: st.column_config.NumberColumn(format="%.3f") for c in ("precision", "recall", "f1", "PR-AUC", "ROC-AUC")})
    a, b = st.columns(2)
    with a:
        t = ev_["tiers"].query("group == 'all wells'")
        fig = go.Figure(go.Bar(x=t.tier, y=t.failure_rate, marker_color=[TIER_COLORS[x] for x in t.tier],
                               text=[f"{r:.0%} fail · {s:.0%} of all failures" for r, s in zip(t.failure_rate, t.share_of_failures)],
                               textposition="outside", cliponaxis=False))
        fig.add_hline(y=BASE_RATE, line=dict(color=GRAY, dash="dash", width=1), annotation_text="base rate", annotation_position="top left")
        fig.update_layout(title="Actual failure rate by risk tier (test)", yaxis_tickformat=".0%",
                          yaxis_range=[0, t.failure_rate.max() * 1.3])
        st.plotly_chart(style(fig, 380), width="stretch")
    with b:
        k = ev_["topk"].query("group == 'all wells'")
        fig = go.Figure(go.Bar(x=k.top, y=k.precision, marker_color=BLUE,
                               text=[f"{p:.0%} · {l:.1f}× base" for p, l in zip(k.precision, k.lift)],
                               textposition="outside", cliponaxis=False))
        fig.add_hline(y=BASE_RATE, line=dict(color=GRAY, dash="dash", width=1), annotation_text="base rate", annotation_position="top left")
        fig.update_layout(title="Share of flagged well-days that failed, top-K riskiest (test)",
                          xaxis_title="top share of well-days by risk", yaxis_tickformat=".0%",
                          yaxis_range=[0, k.precision.max() * 1.3])
        st.plotly_chart(style(fig, 380), width="stretch")
    st.markdown("**Cost simulation** (one decision per well every 30 days; failure = 5 units, intervention = 1 unit)")
    p = ev_["policies"].query("group == 'all wells'")
    p = p[~p.policy.str.contains("field cap|0.647")][["policy", "actions", "tp", "fp", "fn", "precision", "recall",
                                                       "saving_pct"]]
    st.dataframe(p.sort_values("saving_pct", ascending=False), hide_index=True, width="stretch", column_config={
        "saving_pct": st.column_config.NumberColumn("Saving vs doing nothing", format="percent"),
        "precision": st.column_config.NumberColumn(format="%.3f"), "recall": st.column_config.NumberColumn(format="%.3f")})
