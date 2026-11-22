"""Run-life (survival) analysis of PCP wells, 2021-2026.

A "run" starts the day after a repair ends and lasts until the next failure starts. Run length is
counted in PRODUCING days (hours_on_prod > 0), so shut-in periods do not count as pump wear.
Failures come from data/processed/failure_events_v3.parquet: recorded failures from 2025-06 and
repair-job proxies before that (src/merge.py --proxy-before 2025-06-01; 83% precision, 99% recall
against recorded failures). The first run of every well is dropped (its start, i.e. when the pump was
installed, is before 2021 and unknown); the last run is right-censored at the last complete data day.

Kaplan-Meier curves are estimated overall and by field, pump make, RPM band, water-cut band and the
component repaired at the start of the run. For every well currently running, the KM curve of its field
gives the chance of a failure within the next 30 producing days given how long it has already run.

Usage (workspace root, any env with pandas/pyarrow/matplotlib):  python src/survival.py
Writes outputs/survival/{intervals,km_curves,current_wells}.parquet, summary.csv and
reports/figures/09_run_life.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, FIG = ROOT / "data" / "processed", ROOT / "outputs" / "survival", ROOT / "reports" / "figures"
HORIZON = 30
MIN_GROUP = 100        # runs needed for a group's own curve
MAKE_FIX = {"EURO PUMP": "EUROPUMP", "WEEATHERFORD": "WEATHERFORD", "BMW PUMP": "BMW", "PCP PUMP": "PCP",
            "LIFTING SOLUTIONS": "LSI", "MANTL": "MANTL"}


def norm_make(s: pd.Series) -> pd.Series:
    s = s.str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    return s.replace(MAKE_FIX).fillna("UNKNOWN")


def kaplan_meier(t: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Survival S(t) at each distinct event time (t in producing days)."""
    t, event = np.asarray(t), np.asarray(event).astype(bool)
    times = np.sort(np.unique(t[event]))
    at_risk = np.array([(t >= x).sum() for x in times])
    deaths = np.array([((t == x) & event).sum() for x in times])
    surv = np.cumprod(1 - deaths / at_risk)
    return pd.DataFrame({"t": np.r_[0, times], "survival": np.r_[1.0, surv], "at_risk": np.r_[len(t), at_risk]})


def s_at(curve: pd.DataFrame, x: float) -> float:
    i = np.searchsorted(curve.t.to_numpy(), x, side="right") - 1
    return float(curve.survival.iloc[max(i, 0)])


def median_of(curve: pd.DataFrame) -> float:
    below = curve[curve.survival <= 0.5]
    return float(below.t.iloc[0]) if len(below) else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ops = pd.read_parquet(DATA / "dashboard.parquet", columns=["uwi", "production_date", "hours_on_prod",
                                                               "oil_produced_m3", "water_produced_m3", "pcp_pump_rpm"])
    ops = ops.sort_values(["uwi", "production_date"]).reset_index(drop=True)
    # last complete day: drop trailing partly-reported days (most wells reporting 0 oil and 0 water)
    zero = (ops.oil_produced_m3.eq(0) & ops.water_produced_m3.eq(0)).groupby(ops.production_date).mean()
    last_day = zero[zero <= 2 * zero.median()].index.max()
    ops = ops[ops.production_date <= last_day]
    ops["running"] = (ops.hours_on_prod > 0).astype(int)
    ops["cum_run"] = ops.groupby("uwi").running.cumsum()

    ev = pd.read_parquet(DATA / "failure_events_v3.parquet",
                         columns=["uwi", "event_start", "event_end", "failure_group", "source"])
    ev = ev[ev.uwi.isin(set(ops.uwi)) & (ev.event_start <= last_day)].sort_values(["uwi", "event_start"])

    # runs: from the end of one failure to the start of the next (or to last_day, censored)
    runs = ev.assign(start=ev.event_end + pd.Timedelta(days=1), repaired=ev.failure_group, start_source=ev.source)
    runs["end"] = runs.groupby("uwi").event_start.shift(-1)
    runs["failed"] = runs.end.notna()
    runs["end"] = runs.end.fillna(last_day + pd.Timedelta(days=1))
    runs = runs[runs.end > runs.start][["uwi", "start", "end", "failed", "repaired", "start_source"]].reset_index(drop=True)

    cum = ops.set_index(["uwi", "production_date"]).cum_run
    def cum_before(dates):                    # producing days strictly before each date
        idx = pd.MultiIndex.from_arrays([runs.uwi, dates - pd.Timedelta(days=1)])
        return cum.reindex(idx).to_numpy()
    c0, c1 = cum_before(runs.start), cum_before(runs.end)
    runs["run_days"] = c1 - c0
    runs["calendar_days"] = (runs.end - runs.start).dt.days
    runs = runs.dropna(subset=["run_days"])
    runs = runs[runs.run_days >= 0]

    # covariates from the first 30 calendar days of the run (known when the run starts)
    first = ops.merge(runs[["uwi", "start"]].reset_index(), on="uwi")
    first = first[(first.production_date >= first.start) & (first.production_date < first.start + pd.Timedelta(days=30))]
    cov = first.groupby("index").agg(rpm=("pcp_pump_rpm", "median"), oil=("oil_produced_m3", "sum"),
                                     water=("water_produced_m3", "sum"))
    runs = runs.join(cov)
    runs["water_cut"] = runs.water / (runs.water + runs.oil).where(lambda v: v > 0)
    wells = pd.read_parquet(DATA / "wells.parquet", columns=["uwi", "field_operating_area", "pcp_pump_make"])
    wells["pump_make"] = norm_make(wells.pcp_pump_make)
    runs = runs.merge(wells[["uwi", "field_operating_area", "pump_make"]], on="uwi", how="left")
    runs["rpm_band"] = pd.cut(runs.rpm, [-1, 150, 250, 350, 1e4], labels=["<150", "150-250", "250-350", "350+"]).astype(str)
    runs["water_cut_band"] = pd.cut(runs.water_cut, [-0.01, 0.8, 0.95, 0.99, 1.0],
                                    labels=["<80%", "80-95%", "95-99%", "99%+"]).astype(str)
    runs.loc[runs.rpm.isna(), "rpm_band"] = "unknown"
    runs.loc[runs.water_cut.isna(), "water_cut_band"] = "unknown"
    # drop each well's first run: the pump's install date (before 2021) is unknown
    first_run = runs.groupby("uwi").start.transform("min").eq(runs.start)
    runs = runs[~first_run | (runs.start_source == "recorded")].reset_index(drop=True)
    runs.to_parquet(OUT / "intervals.parquet", index=False)

    # KM curves and summary
    dims = {"all": None, "field": "field_operating_area", "pump_make": "pump_make", "rpm_band": "rpm_band",
            "water_cut_band": "water_cut_band", "repaired": "repaired"}
    curves, rows = [], []
    for dim, col in dims.items():
        groups = [("all runs", runs)] if col is None else [(g, d) for g, d in runs.groupby(col) if len(d) >= MIN_GROUP
                                                          and g not in ("unknown", "UNKNOWN", "nan")]
        for g, d in groups:
            km = kaplan_meier(d.run_days.to_numpy(), d.failed.to_numpy())
            curves.append(km.assign(dimension=dim, group=g))
            rows.append({"dimension": dim, "group": g, "runs": len(d), "failures": int(d.failed.sum()),
                         "wells": d.uwi.nunique(), "median_run_days": median_of(km),
                         **{f"survive_{x}d": s_at(km, x) for x in (30, 90, 180, 365)}})
    km_all = pd.concat(curves, ignore_index=True)
    km_all.to_parquet(OUT / "km_curves.parquet", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    # current wells: running days since the last repair, and the KM chance of failing in the next 30 days
    last_end = ev.groupby("uwi").event_end.max()
    cur = ops.groupby("uwi").agg(cum_now=("cum_run", "last"), last_status_day=("production_date", "max"))
    cur["last_repair_end"] = last_end
    cum_at = cum.reindex(pd.MultiIndex.from_arrays([cur.index, cur.last_repair_end])).to_numpy()
    cur["run_days_since_repair"] = np.where(cur.last_repair_end.notna(), cur.cum_now - cum_at, np.nan)
    cur = cur.join(wells.set_index("uwi")[["field_operating_area", "pump_make"]])
    fields = set(summary.query("dimension == 'field'").group)

    def p_fail(row):
        if np.isnan(row.run_days_since_repair):
            return np.nan, "no repair since 2021"
        dim, grp = ("field", row.field_operating_area) if row.field_operating_area in fields else ("all", "all runs")
        km = km_all[(km_all.dimension == dim) & (km_all.group == grp)]
        i = np.searchsorted(km.t.to_numpy(), row.run_days_since_repair + HORIZON, side="right") - 1
        if km.at_risk.iloc[max(i, 0)] < 30:          # too few runs that long: no reliable estimate
            return np.nan, f"{grp} (beyond observed run lengths)"
        s0 = s_at(km, row.run_days_since_repair)
        return (1 - s_at(km, row.run_days_since_repair + HORIZON) / s0 if s0 > 0 else np.nan), grp
    cur[["km_p_fail_30d", "km_curve_used"]] = cur.apply(lambda r: pd.Series(p_fail(r)), axis=1)
    cur.reset_index().to_parquet(OUT / "current_wells.parquet", index=False)

    print(f"runs {len(runs):,} ({runs.failed.sum():,} ended in failure, rest censored at {last_day.date()}), "
          f"{runs.uwi.nunique()} wells")
    show = summary.copy()
    for c in [c for c in show if c.startswith("survive")]:
        show[c] = (show[c] * 100).round(0).astype(int).astype(str) + "%"
    print(show.to_string(index=False))
    plot(km_all, summary)


def plot(km_all: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.titlesize": 10, "axes.titlelocation": "left", "axes.titlecolor": INK})
    panels = [("all", "All runs"), ("rpm_band", "By pump RPM (first 30 days)"),
              ("water_cut_band", "By water cut (first 30 days)"), ("field", "By field (largest 4)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, (dim, title) in zip(axes, panels):
        groups = summary[summary.dimension == dim]
        if dim == "field":
            groups = groups.nlargest(4, "runs")
        order = {"rpm_band": ["<150", "150-250", "250-350", "350+"],
                 "water_cut_band": ["<80%", "80-95%", "95-99%", "99%+"]}.get(dim)
        names = [g for g in (order or groups.group.tolist()) if g in set(groups.group)]
        for k, g in enumerate(names):
            c = km_all[(km_all.dimension == dim) & (km_all.group == g)]
            c = c[c.t <= 365]
            med = groups.set_index("group").median_run_days[g]
            ax.step(c.t, c.survival, where="post", lw=2, color=SERIES[k],
                    label=f"{g} (median {med:.0f} d)" if not np.isnan(med) else g)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_title(title); ax.set_xlabel("producing days since last repair"); ax.set_xlim(0, 365)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("share of runs still going")
    fig.suptitle("PCP run life 2021-2026: how long wells keep producing before the next failure (Kaplan-Meier)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "09_run_life.png", dpi=150)
    print(f"wrote {FIG / '09_run_life.png'}")


if __name__ == "__main__":
    main()
