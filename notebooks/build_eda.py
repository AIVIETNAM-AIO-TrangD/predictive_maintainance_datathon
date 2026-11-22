"""Generates notebooks/eda_failure_drivers.ipynb (run once, then execute the notebook)."""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# What drives PCP well failures? EDA on the merged dataset

**Data:** `data/processed/merged.parquet` (built by `src/clean.py` then `src/merge.py`).

**Target:** `fail_next_30d`, which is 1 if a failure event starts in the next 30 days.

**Rows used:** a day is used only when all of these hold:
- the well is producing and not in a failure
- the label is observed, meaning the date is on or after 2025-06-01 (failures were not recorded before then)
- the 30-day horizon is still inside the data

**Feature timing:** features use only information available up to and including day *t*.
""")

md(Path(__file__).with_name("_findings.md").read_text(encoding="utf-8"))

code("""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
P, FIG = ROOT / "data/processed", ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42

BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titlelocation": "left", "axes.titlecolor": INK, "lines.linewidth": 2,
})
def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=150); plt.show()
""")

md("## 1. Load and build rolling operating features (past-only)")
code("""
cols = ["uwi", "production_date", "split", "well_daily_status", "hours_on_prod", "hours_down",
        "oil_produced_m3", "water_produced_m3", "pump_efficiency_pct", "pcp_pump_rpm",
        "pcp_pump_electric_current_a", "pcp_pump_rod_torque_ftlb", "pcp_pump_volume_throughput_m3d",
        "pcp_pump_displacement_m3d100rpm", "tubing_pressure_kpa", "casing_pressure_kpa",
        "failure_active", "failure_onset", "fail_next_30d", "label_observed",
        "fail_count_prior", "fail_count_365d", "fail_count_90d", "days_since_last_failure", "last_failure_type",
        "job_count_prior", "job_count_365d", "job_count_90d", "job_count_prior_rod_truck",
        "job_count_prior_workover", "job_count_prior_pump_change", "days_since_last_job",
        "days_since_last_rod_truck", "days_since_last_workover", "days_since_last_pump_change",
        "last_job_category", "job_cost_prior", "field_operating_area", "province",
        "surface_latitude", "surface_longitude", "pcp_pump_make", "pcp_pump_model", "well_age_years"]
m = pd.read_parquet(P / "merged.parquet", columns=cols)
m = m[m.production_date >= "2025-03-01"].sort_values(["uwi", "production_date"]).reset_index(drop=True)

g = m.groupby("uwi", sort=False)
def roll(c, w, minp, how="mean", shift=0):
    return g[c].transform(lambda s: getattr(s.shift(shift).rolling(w, min_periods=minp), how)())

ops = {"pcp_pump_rod_torque_ftlb": "torque", "pcp_pump_electric_current_a": "amps", "pcp_pump_rpm": "rpm",
       "pump_efficiency_pct": "eff", "pcp_pump_volume_throughput_m3d": "throughput", "oil_produced_m3": "oil"}
for c, n in ops.items():
    m[f"{n}_30d"] = roll(c, 30, 10)
    recent, prior = roll(c, 7, 3), roll(c, 30, 10, shift=7)
    m[f"{n}_trend"] = recent - prior if n == "eff" else recent / prior.where(prior > 0) - 1
m["hours_down_30d"] = roll("hours_down", 30, 20, "sum")
m["downtime_days_30d"] = g["hours_on_prod"].transform(lambda s: (s == 0).astype(float).rolling(30, min_periods=20).sum())
w, o = roll("water_produced_m3", 30, 10), roll("oil_produced_m3", 30, 10)
m["water_cut_30d"] = w / (w + o)
m["torque_per_rpm"] = m["torque_30d"] / m["rpm_30d"].where(m["rpm_30d"] > 0)

base = m[m.label_observed & (m.failure_active == 0) & m.fail_next_30d.notna()
         & m.well_daily_status.eq("Producing")].copy()
RATE = base.fail_next_30d.mean()
print(f"analysis rows: {len(base):,}  wells: {base.uwi.nunique()}  base rate fail_next_30d: {RATE:.3f}")
""")

md("## 2. What fails, and why")
code("""
ev = pd.read_parquet(P / "failure_events.parquet")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, col, title in [(axes[0], "failure_group", "Failure events by component"),
                       (axes[1], "cause_group", "Failure events by recorded cause (top 8)")]:
    vc = ev[col].value_counts().head(8)[::-1]
    ax.barh(vc.index, vc.values, color=BLUE, height=0.6)
    for y, v in enumerate(vc.values):
        ax.text(v, y, f" {v:,} ({v / len(ev):.0%})", va="center", color=MUTED, fontsize=9)
    ax.set_title(title); ax.grid(axis="y", visible=False); ax.set_xlim(0, vc.max() * 1.3)
save(fig, "01_failure_mix")
print(pd.crosstab(ev.failure_group, ev.repair_job_category.fillna("no job matched"), margins=True))
""")

md("""
## 3. Single-feature screen

For each candidate feature, the table shows:
- **AUC**: how well the feature alone ranks days that are followed by a failure within 30 days. 0.5 means no signal. The direction column says whether higher or lower values carry the risk.
- **top/bottom decile rate**: the failure rate in the highest-risk 10% of the feature's values compared with the lowest-risk 10%.

Nearby days from the same well are highly correlated, so read these as a ranking of candidates, not as significance tests.
""")
code("""
num_feats = ["torque_30d", "amps_30d", "rpm_30d", "eff_30d", "throughput_30d", "oil_30d", "torque_per_rpm",
             "torque_trend", "amps_trend", "rpm_trend", "eff_trend", "throughput_trend", "oil_trend",
             "hours_down_30d", "downtime_days_30d", "water_cut_30d", "pcp_pump_displacement_m3d100rpm",
             "tubing_pressure_kpa", "casing_pressure_kpa",
             "fail_count_prior", "fail_count_365d", "fail_count_90d", "days_since_last_failure",
             "job_count_prior", "job_count_365d", "job_count_90d", "job_count_prior_rod_truck",
             "job_count_prior_workover", "job_count_prior_pump_change", "days_since_last_job",
             "days_since_last_rod_truck", "days_since_last_workover", "days_since_last_pump_change",
             "job_cost_prior", "well_age_years", "surface_latitude", "surface_longitude"]
rows = []
for f in num_feats:
    d = base[[f, "fail_next_30d"]].dropna()
    if d[f].nunique() < 2: continue
    auc = roc_auc_score(d.fail_next_30d, d[f])
    dec = pd.qcut(d[f].rank(method="first"), 10, labels=False)
    r = d.groupby(dec).fail_next_30d.mean()
    hi, lo = (r.iloc[-1], r.iloc[0]) if auc >= 0.5 else (r.iloc[0], r.iloc[-1])
    rows.append({"feature": f, "AUC": max(auc, 1 - auc), "risk when": "higher" if auc >= 0.5 else "lower",
                 "risky decile rate": hi, "safe decile rate": lo, "lift": hi / max(lo, 1e-9),
                 "coverage": len(d) / len(base)})
screen = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
screen.style.format({"AUC": "{:.3f}", "risky decile rate": "{:.1%}", "safe decile rate": "{:.1%}",
                     "lift": "{:.1f}x", "coverage": "{:.0%}"})
""")

md("## 4. Hypotheses, one panel each: failure rate across the feature's range")
code("""
def rate_curve(ax, f, label, bins=10, logx=False):
    d = base[[f, "fail_next_30d"]].dropna()
    q = pd.qcut(d[f].rank(method="first"), bins, labels=False)
    s = d.groupby(q).agg(x=(f, "median"), rate=("fail_next_30d", "mean"))
    ax.axhline(RATE, color=MUTED, lw=1, ls="--")
    ax.plot(s.x, s.rate, color=BLUE, marker="o", ms=5)
    ax.set_title(label); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    if logx: ax.set_xscale("symlog")

panels = [("torque_30d", "Rod torque, 30-day mean (ft-lb)"), ("rpm_30d", "Pump RPM, 30-day mean"),
          ("amps_30d", "Motor current, 30-day mean (A)"), ("torque_per_rpm", "Torque per RPM (load proxy)"),
          ("eff_30d", "Pump efficiency, 30-day mean"), ("eff_trend", "Efficiency change: last 7d vs prior 30d"),
          ("throughput_trend", "Throughput change (relative)"), ("hours_down_30d", "Hours down, last 30 days"),
          ("water_cut_30d", "Water cut, 30-day")]
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=True)
for ax, (f, lab) in zip(axes.flat, panels):
    rate_curve(ax, f, lab, logx=f == "hours_down_30d")
for ax in axes[:, 0]: ax.set_ylabel("fail in next 30 days")
fig.suptitle(f"Failure rate by decile of each feature (dashed = overall {RATE:.0%})", x=0.01, ha="left", color=INK)
save(fig, "02_operating_hypotheses")
""")

md("## 5. What happens to the pump in the 60 days before a failure")
code("""
# each metric divided by that event's own baseline (median of days -90..-31 before onset)
# RPM, torque, amps and throughput change on only 2-9% of days (setpoints / occasional field
# entries), so their pre-failure median is flat by construction. Only daily-measured signals here.
for c in ["pcp_pump_rpm", "pcp_pump_rod_torque_ftlb", "pcp_pump_electric_current_a", "pcp_pump_volume_throughput_m3d"]:
    d = m.groupby("uwi")[c].diff()
    print(f"{c:32s} value changes day-to-day on {(d.fillna(0) != 0)[m[c].notna()].mean():.1%} of days")
metrics = {"pump_efficiency_pct": "Pump efficiency", "oil_produced_m3": "Oil rate",
           "water_produced_m3": "Water rate", "hours_on_prod": "Hours on production"}
full = pd.read_parquet(P / "merged.parquet", columns=["uwi", "production_date"] + list(metrics))
idx = full.set_index(["uwi", "production_date"])
evo = ev[ev.event_start >= "2025-06-01"]
offs = np.arange(-90, 8)
grid = pd.MultiIndex.from_arrays([np.repeat(evo.uwi.values, len(offs)),
                                  np.repeat(evo.event_start.values, len(offs)) + np.tile(offs, len(evo)).astype("timedelta64[D]")])
win = idx.reindex(grid).reset_index(drop=True)
win["eid"], win["off"] = np.repeat(np.arange(len(evo)), len(offs)), np.tile(offs, len(evo))
fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharex=True)
summary = {}
for ax, (c, lab) in zip(axes.flat, metrics.items()):
    bl = win[win.off.between(-90, -31)].groupby("eid")[c].median()
    rel = win[c] / win.eid.map(bl.where(bl > 0))
    s = rel.groupby(win.off).median().loc[-60:7]
    q1, q3 = rel.groupby(win.off).quantile(0.25).loc[-60:7], rel.groupby(win.off).quantile(0.75).loc[-60:7]
    ax.fill_between(s.index, q1, q3, color=BLUE, alpha=0.15, lw=0)
    ax.plot(s.index, s.values, color=BLUE)
    ax.axhline(1, color=MUTED, lw=1, ls="--"); ax.axvline(0, color=ORANGE, lw=1.5)
    ax.set_title(lab)
    summary[lab] = {"day -30": s.loc[-30], "day -14": s.loc[-14], "day -7": s.loc[-7], "day -1": s.loc[-1]}
axes.flat[-1].axis("off")
axes.flat[-1].text(0, 0.5, "Median across events (line)\\nand middle 50% (band),\\nrelative to each event's own\\nday -90..-31 baseline.\\nOrange line = failure onset.",
                   color=MUTED, va="center")
fig.suptitle(f"Pump signals before {len(evo):,} failure onsets (x = days relative to onset)", x=0.01, ha="left", color=INK)
save(fig, "03_event_study")
pd.DataFrame(summary).T.round(3)
""")

md("## 6. History: past failures and maintenance")
code("""
def rate_by(ax, s, title):
    r = base.groupby(s, observed=True).fail_next_30d.agg(["mean", "size"])
    ax.bar(r.index.astype(str), r["mean"], color=BLUE, width=0.6)
    ax.axhline(RATE, color=MUTED, lw=1, ls="--")
    for i, (v, n) in enumerate(zip(r["mean"], r["size"])):
        ax.text(i, v, f"{v:.0%}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_title(title); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", visible=False)
    return r

day_bins, day_lab = [-1, 30, 90, 180, 365, 10_000], ["≤30d", "31-90d", "91-180d", "181-365d", ">1y"]
def by_days(c):
    return pd.cut(base[c], day_bins, labels=day_lab).cat.add_categories("never").fillna("never")
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
r1 = rate_by(axes[0, 0], base.fail_count_365d.clip(upper=4).astype(int).astype(str).replace("4", "4+"), "Failures in the previous 365 days")
r2 = rate_by(axes[0, 1], by_days("days_since_last_failure"), "Time since last failure (records from mid-2025)")
r3 = rate_by(axes[1, 0], by_days("days_since_last_rod_truck"), "Time since last rod-truck job (minor repair)")
r4 = rate_by(axes[1, 1], by_days("days_since_last_pump_change"), "Pump age: time since last pump change")
save(fig, "04_history")
fig, ax = plt.subplots(figsize=(7, 3.5))
r5 = rate_by(ax, base.job_count_prior_rod_truck.clip(upper=10).astype(int).map(lambda v: "10+" if v == 10 else str(v)),
             "Rod-truck jobs since 2021 (repair history before failure records began)")
save(fig, "05_rod_job_history")
""")

md("## 7. Well and location")
code("""
wl = base.groupby("uwi").agg(days=("fail_next_30d", "size"), field=("field_operating_area", "first"),
                             make=("pcp_pump_make", "first"), lat=("surface_latitude", "first"),
                             lon=("surface_longitude", "first"), province=("province", "first"))
wl["events"] = base.groupby("uwi").failure_onset.sum()  # onsets on producing days
on = m[m.label_observed].groupby("uwi").failure_onset.sum()
wl["events"] = on.reindex(wl.index).fillna(0)
wl = wl[wl.days >= 180]
wl["per_year"] = wl.events / (wl.days / 365.25)
overall = wl.events.sum() / (wl.days.sum() / 365.25)

def per_group(col, min_wells=15):
    t = wl.groupby(col).agg(wells=("events", "size"), events=("events", "sum"), years=("days", lambda d: d.sum() / 365.25))
    t["failures_per_well_year"] = t.events / t.years
    return t[t.wells >= min_wells].sort_values("failures_per_well_year")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, title in [(axes[0], "field", "Failures per well-year by field (≥15 wells)"),
                       (axes[1], "make", "Failures per well-year by pump make (≥15 wells)")]:
    t = per_group(col)
    ax.barh(t.index, t.failures_per_well_year, color=BLUE, height=0.6)
    ax.axvline(overall, color=MUTED, lw=1, ls="--")
    for y, (v, n) in enumerate(zip(t.failures_per_well_year, t.wells)):
        ax.text(v, y, f" {v:.2f}  ({n} wells)", va="center", color=MUTED, fontsize=8)
    ax.set_title(title); ax.grid(axis="y", visible=False); ax.set_xlim(0, t.failures_per_well_year.max() * 1.35)
save(fig, "06_field_and_make")
field_tbl, make_tbl = per_group("field"), per_group("make")
print(f"overall: {overall:.2f} failures per well-year over {len(wl)} wells")
print("share of wells with zero failures:", f"{(wl.events == 0).mean():.0%}",
      "| top 10% of wells account for", f"{wl.events.nlargest(int(len(wl) * .1)).sum() / wl.events.sum():.0%} of failures")

fig, ax = plt.subplots(figsize=(8, 6))
d = wl.dropna(subset=["lat", "lon"]).sort_values("per_year")
sc = ax.scatter(d.lon, d.lat, c=d.per_year.clip(upper=d.per_year.quantile(.95)), cmap="Blues", s=14,
                edgecolors="#9a9893", linewidths=0.3)
cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label("failures per well-year (capped at p95)", color=MUTED)
ax.set_title("Where failures happen (each dot is a well)"); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
save(fig, "07_map")
""")

md("""
## 8. All features together: gradient boosting + permutation importance

- **Model:** gradient boosting, trained on the training wells and scored on the 237 held-out wells (`split == 'test_well'`), so it is never scored on a well it has seen.
- **Importance:** permutation importance on the held-out wells, measured as the drop in AUC when a feature is shuffled.

This isn't the final model. It only tests which features still matter once the others are taken into account.
""")
code("""
cat_feats = ["field_operating_area", "pcp_pump_make", "last_job_category", "last_failure_type", "province"]
X_cols = num_feats + cat_feats
X = base[X_cols].copy()
for c in cat_feats: X[c] = X[c].astype("category")
y = base.fail_next_30d.astype(int)
tr, te = base.split.ne("test_well").values, base.split.eq("test_well").values
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                     categorical_features="from_dtype", random_state=SEED)
clf.fit(X[tr], y[tr])
auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
print(f"held-out-well AUC: {auc:.3f}   (train rows {tr.sum():,}, test rows {te.sum():,})")

rng = np.random.default_rng(SEED)
sub = rng.choice(np.flatnonzero(te), size=min(60_000, te.sum()), replace=False)
pi = permutation_importance(clf, X.iloc[sub], y.iloc[sub], scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X_cols).sort_values(ascending=False)

group = lambda f: ("history: failures" if f.startswith(("fail_", "days_since_last_failure", "last_failure")) else
                   "history: jobs" if ("job" in f or f.startswith("days_since_last_")) else
                   "well / location" if f in ("field_operating_area", "pcp_pump_make", "province", "surface_latitude",
                                               "surface_longitude", "well_age_years", "pcp_pump_displacement_m3d100rpm") else
                   "operations")
top = imp.head(20)[::-1]
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.barh(top.index, top.values, color=BLUE, height=0.6)
for y_, (f, v) in enumerate(top.items()):
    ax.text(v, y_, f"  {group(f)}", va="center", color=MUTED, fontsize=8)
ax.set_title(f"Top 20 features by permutation importance\\n(AUC drop on held-out wells; model AUC {auc:.2f})")
ax.grid(axis="y", visible=False); ax.set_xlim(0, top.max() * 1.45)
save(fig, "08_importance")
print(imp.groupby(imp.index.map(group)).sum().sort_values(ascending=False).round(4))
""")

md("""
## 9. Ablation: how much does each feature family add?
Each model below drops one feature family. The AUC loss on held-out wells shows how much that family contributes beyond the others.
""")
code("""
fams = {f: group(f) for f in X_cols}
res = {"all features": auc}
for fam in sorted(set(fams.values())):
    keep = [f for f in X_cols if fams[f] != fam]
    c2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                        categorical_features="from_dtype", random_state=SEED).fit(X.loc[tr, keep], y[tr])
    res[f"without {fam}"] = roc_auc_score(y[te], c2.predict_proba(X.loc[te, keep])[:, 1])
pd.Series(res).round(3)
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path(__file__).with_name("eda_failure_drivers.ipynb")
nbf.write(nb, out)
print("wrote", out)
