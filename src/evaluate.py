"""Cost-aware evaluation of fail_next_30d predictions.

Usage (workspace root):
    python src/evaluate.py                                    # compares v1 and v2 prediction files
    python src/evaluate.py --cost-fn 10 --cost-fp 1           # different cost ratio
    python src/evaluate.py --oil-value 600 --cost-fn 40000 --cost-fp 5000   # currency units, incl. lost oil

Inputs: prediction files written by src/predict.py with --start 2026-04-01, so they contain
validation-period rows (period == "val_period") and test rows (period == "test_period").

Leakage rules: every threshold, the probability calibration, the field cap of the budget policy and the
risk-tier cut-offs are fitted on validation rows; test rows (date >= 2026-06-27) are only scored.

Reports (also written to outputs/evaluation/):
  thresholds.csv   precision/recall/f1/pr_auc/roc_auc + confusion matrix per threshold and well group
  topk.csv         precision/recall/lift of the top 1/5/10/20% riskiest rows
  tiers.csv        failure rate and share of failures captured by risk tier
  policies.csv     cost simulation at monthly decision snapshots (one decision per well per 30 days);
                   top-K budgets are wells per snapshot for the whole fleet, scaled to each group's size
  sensitivity.csv  how the cost-optimal threshold and saving move with the FN:FP cost ratio
  summary.md       the main tables in one file
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from costs import (CostParams, apply_calibration, calibrate, cost_optimal_threshold,  # noqa: E402
                   expected_value, failure_cost, outcome_costs)

TARGET, PROB = "fail_next_30d", "fail_next_30d_prob"
TEST_START, VAL_START = pd.Timestamp("2026-06-27"), pd.Timestamp("2026-04-01")
GROUPS = {"all wells": lambda d: d,
          "wells seen in training": lambda d: d[d.well_group != "held_out_well"],
          "237 held-out wells": lambda d: d[d.well_group == "held_out_well"]}


def rank_metrics(y, p):
    o = np.argsort(-p, kind="stable"); ys = y[o]
    ap = (np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / max(ys.sum(), 1)
    r = pd.Series(p).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return float(ap), float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def at_threshold(y, p, thr):
    act = p >= thr
    tp, fp = int((act & (y == 1)).sum()), int((act & (y == 0)).sum())
    fn, tn = int((~act & (y == 1)).sum()), int((~act & (y == 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"flagged_share": act.mean(), "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "accuracy": (tp + tn) / len(y), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def best_f1_threshold(y, p):
    o = np.argsort(-p, kind="stable"); ys, ps = y[o], p[o]
    tp = np.cumsum(ys); prec = tp / np.arange(1, len(ys) + 1); rec = tp / ys.sum()
    return float(ps[(2 * prec * rec / np.maximum(prec + rec, 1e-9)).argmax()])


def capped_topk(snap: pd.DataFrame, score: str, k: int, cap: float | None) -> np.ndarray:
    """Pick k wells by score; if cap is set, no field may take more than cap*k of the picks."""
    order = snap.sort_values(score, ascending=False)
    if cap is None:
        chosen = order.index[:k]
    else:
        limit, taken, chosen = max(1, int(np.ceil(cap * k))), {}, []
        for i, f in zip(order.index, order.field_operating_area.fillna("?")):
            if taken.get(f, 0) < limit:
                chosen.append(i); taken[f] = taken.get(f, 0) + 1
                if len(chosen) == k:
                    break
    act = np.zeros(len(snap), bool)
    act[snap.index.get_indexer(chosen)] = True
    return act


def snapshots(d: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One decision per well every 30 days: non-overlapping 30-day outcome windows."""
    days = pd.date_range(start, end, freq="30D")
    return d[d.production_date.isin(days)].reset_index(drop=True)


def evaluate(name: str, path: Path, cp: CostParams, budget: int, out_dir: Path, keys: pd.DataFrame | None) -> dict:
    d = pd.read_parquet(path)
    d = d[d.eval_row].copy()
    if keys is not None:     # score every model on exactly the same well-days
        d = d.merge(keys, on=["uwi", "production_date"])
    val = d[d.period.eq("val_period") & d.well_group.ne("held_out_well")].reset_index(drop=True)
    test = d[d.period.eq("test_period")].reset_index(drop=True)
    assert len(val) and len(test), f"{path} needs validation and test rows (run predict.py with --start 2026-04-01)"
    assert val.production_date.max() < TEST_START
    yv, pv = val[TARGET].to_numpy(int), val[PROB].to_numpy()

    # ---- everything below is fitted on validation rows only
    p = np.clip(pv, 1e-7, 1 - 1e-7)
    ab = calibrate(np.log(p / (1 - p)), yv)
    fc_val = failure_cost(val.oil_rate_30d.to_numpy(), cp)
    thresholds = {"0.5": 0.5, "0.647 (v1 val-F1)": 0.647, "val F1-optimal": best_f1_threshold(yv, pv),
                  "val cost-optimal": cost_optimal_threshold(yv, pv, cp, fc_val)}
    hi, med = np.quantile(pv, 0.95), np.quantile(pv, 0.80)
    for df in (val, test):
        df["p_cal"] = apply_calibration(df[PROB].to_numpy(), ab)
        df["failure_cost"] = failure_cost(df.oil_rate_30d.to_numpy(), cp)
        df["ev"] = expected_value(df.p_cal.to_numpy(), df.failure_cost.to_numpy(), cp)
        df["tier"] = np.select([df[PROB] >= hi, df[PROB] >= med], ["high", "medium"], default="low")

    # field cap for the budget policy: chosen on validation snapshots
    vs = snapshots(val, VAL_START, val.production_date.max())
    cap_cost = {}
    for cap in (None, 0.5, 0.35, 0.25):
        cost = sum(outcome_costs(g[TARGET].to_numpy(int), capped_topk(g.reset_index(drop=True), "ev", budget, cap),
                                 g.failure_cost.to_numpy(), cp)["total_cost"] for _, g in vs.groupby("production_date"))
        cap_cost[cap] = cost
    best_cap = min(cap_cost, key=cap_cost.get)

    # ---- test: scoring only
    thr_rows, topk_rows, tier_rows, pol_rows = [], [], [], []
    # budget is "wells per snapshot" for the whole fleet; a sub-group gets the same share of its own wells
    fleet = snapshots(test, TEST_START, test.production_date.max()).groupby("production_date").size().mean()
    for gname, sel in GROUPS.items():
        t = sel(test)
        y, s = t[TARGET].to_numpy(int), t[PROB].to_numpy()
        ap, auc = rank_metrics(y, s)
        for tname, thr in thresholds.items():
            thr_rows.append({"model": name, "group": gname, "threshold": tname, "value": thr, "rows": len(t),
                             "positive_rate": y.mean(), "pr_auc": ap, "roc_auc": auc, **at_threshold(y, s, thr)})
        o = np.argsort(-s, kind="stable")
        for k in (0.01, 0.05, 0.10, 0.20):
            top = o[: max(1, int(len(s) * k))]
            topk_rows.append({"model": name, "group": gname, "top": f"{k:.0%}", "rows": len(top),
                              "precision": y[top].mean(), "recall": y[top].sum() / y.sum(), "lift": y[top].mean() / y.mean(),
                              "winter_share": t.field_operating_area.iloc[top].eq("Winter").mean()})
        for tier in ("high", "medium", "low"):
            m = t.tier.eq(tier).to_numpy()
            tier_rows.append({"model": name, "group": gname, "tier": tier, "rows": int(m.sum()), "failure_rate": y[m].mean(),
                              "share_of_failures": y[m].sum() / y.sum(),
                              "winter_share": t.field_operating_area[m].eq("Winter").mean()})

        ts = snapshots(t, TEST_START, t.production_date.max())
        policies = {f"threshold {k}": ("thr", v) for k, v in thresholds.items()}
        policies.update({"EV > 0 (calibrated, oil-weighted)": ("ev", None),
                         f"top-{budget} by risk": ("topk", (PROB, None)),
                         f"top-{budget} by EV": ("topk", ("ev", None)),
                         f"top-{budget} by EV, field cap {best_cap}": ("topk", ("ev", best_cap))})
        wint_fail = ts[TARGET].eq(1) & ts.field_operating_area.eq("Winter")
        for pname, (kind, arg) in policies.items():
            tot, acts_winter = None, 0
            for _, g in ts.groupby("production_date"):
                g = g.reset_index(drop=True)
                if kind == "thr":
                    act = g[PROB].to_numpy() >= arg
                elif kind == "ev":
                    act = g.ev.to_numpy() > 0
                else:
                    k = max(1, int(round(budget * len(g) / fleet)))
                    act = capped_topk(g, arg[0], min(k, len(g)), arg[1])
                r = outcome_costs(g[TARGET].to_numpy(int), act, g.failure_cost.to_numpy(), cp)
                acts_winter += int((act & g.field_operating_area.eq("Winter").to_numpy()).sum())
                tot = r if tot is None else {k: tot[k] + r[k] for k in ("tp", "fp", "fn", "tn", "actions", "total_cost",
                                                                         "do_nothing_cost", "saving")}
            tot["saving_pct"] = tot["saving"] / tot["do_nothing_cost"] if tot["do_nothing_cost"] else np.nan
            pol_rows.append({"model": name, "group": gname, "policy": pname, "snapshots": ts.production_date.nunique(),
                             "decisions": len(ts), **tot,
                             "precision": tot["tp"] / max(tot["actions"], 1),
                             "recall": tot["tp"] / max(tot["tp"] + tot["fn"], 1),
                             "winter_share_of_actions": acts_winter / max(tot["actions"], 1),
                             "winter_share_of_failures": wint_fail.sum() / max(ts[TARGET].sum(), 1)})

    sens = []
    for ratio in (2, 5, 10, 20):
        c2 = CostParams(cost_fn=ratio * cp.cost_fp, cost_fp=cp.cost_fp, oil_value_per_m3=cp.oil_value_per_m3,
                        downtime_days=cp.downtime_days, prevent_rate=cp.prevent_rate)
        thr = cost_optimal_threshold(yv, pv, c2, failure_cost(val.oil_rate_30d.to_numpy(), c2))
        ts = snapshots(test, TEST_START, test.production_date.max())
        r = outcome_costs(ts[TARGET].to_numpy(int), ts[PROB].to_numpy() >= thr, failure_cost(ts.oil_rate_30d.to_numpy(), c2), c2)
        sens.append({"model": name, "fn_to_fp_cost": ratio, "val_cost_threshold": thr,
                     "test_actions": r["actions"], "test_recall": r["tp"] / max(r["tp"] + r["fn"], 1),
                     "test_saving_pct": r["saving_pct"]})
    return {"thresholds": pd.DataFrame(thr_rows), "topk": pd.DataFrame(topk_rows), "tiers": pd.DataFrame(tier_rows),
            "policies": pd.DataFrame(pol_rows), "sensitivity": pd.DataFrame(sens),
            "fitted": {"model": name, "calibration_ab": ab, "thresholds": thresholds, "field_cap": best_cap,
                       "val_cap_costs": cap_cost}}


def md(df: pd.DataFrame, cols: list[str]) -> str:
    f = df[cols].copy()
    for c in f.columns:
        if f[c].dtype.kind == "f":
            f[c] = f[c].map(lambda v: "" if pd.isna(v) else (f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
        "| " + " | ".join(map(str, r)) + " |" for r in f.itertuples(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", default=["v1=outputs/predictions_fail_next_30d_v1.parquet",
                                                  "v2=outputs/predictions_fail_next_30d_v2.parquet"],
                    help="name=path pairs of prediction files containing val + test rows")
    ap.add_argument("--cost-fn", type=float, default=CostParams.cost_fn)
    ap.add_argument("--cost-fp", type=float, default=CostParams.cost_fp)
    ap.add_argument("--oil-value", type=float, default=CostParams.oil_value_per_m3)
    ap.add_argument("--downtime-days", type=float, default=None,
                    help="days lost per failure (default: mean event length of failures before the test period)")
    ap.add_argument("--prevent-rate", type=float, default=CostParams.prevent_rate)
    ap.add_argument("--budget", type=int, default=50, help="wells that can be acted on per 30-day snapshot")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/evaluation"))
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.downtime_days is None:
        ev = pd.read_parquet("data/processed/failure_events.parquet", columns=["event_start", "event_days"])
        args.downtime_days = float(ev.loc[ev.event_start < TEST_START, "event_days"].mean())
    cp = CostParams(args.cost_fn, args.cost_fp, args.oil_value, args.downtime_days, args.prevent_rate)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("cost parameters:", cp.dict(), f"| budget {args.budget} wells per 30-day snapshot")

    parts = {k: [] for k in ("thresholds", "topk", "tiers", "policies", "sensitivity")}
    fitted = []
    specs = [s.split("=", 1) for s in args.pred if Path(s.split("=", 1)[1]).exists()]
    keys = None
    for _, path in specs:
        k = pd.read_parquet(path, columns=["uwi", "production_date", "eval_row"]).query("eval_row")[["uwi", "production_date"]]
        keys = k if keys is None else keys.merge(k, on=["uwi", "production_date"])
    if len(specs) > 1:
        print(f"comparing {len(specs)} models on the {len(keys):,} labelled well-days they all score")
    for name, path in specs:
        r = evaluate(name, Path(path), cp, args.budget, args.out_dir, keys if len(specs) > 1 else None)
        for k in parts:
            parts[k].append(r[k])
        fitted.append(r["fitted"])
    res = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    for k, v in res.items():
        v.to_csv(args.out_dir / f"{k}.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    show = lambda df, cols: print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))  # noqa: E731
    print("\n=== fitted on validation (never on test) ===")
    for f in fitted:
        print(f"{f['model']}: thresholds {', '.join(f'{k}={v:.3f}' for k, v in f['thresholds'].items())}; "
              f"field cap for budget policy: {f['field_cap']}")
    th_cols = ["model", "group", "threshold", "value", "precision", "recall", "f1", "pr_auc", "roc_auc", "tn", "fp", "fn", "tp"]
    print("\n=== test metrics by threshold ==="); show(res["thresholds"], th_cols)
    tk_cols = ["model", "group", "top", "rows", "precision", "recall", "lift", "winter_share"]
    print("\n=== test top-K slices ==="); show(res["topk"], tk_cols)
    ti_cols = ["model", "group", "tier", "rows", "failure_rate", "share_of_failures", "winter_share"]
    print("\n=== test risk tiers ==="); show(res["tiers"], ti_cols)
    po_cols = ["model", "group", "policy", "actions", "tp", "fp", "fn", "precision", "recall", "total_cost",
               "saving_pct", "winter_share_of_actions", "winter_share_of_failures"]
    print("\n=== test cost simulation (monthly snapshots; cost units as configured) ===")
    show(res["policies"], po_cols)
    print("\n=== sensitivity to FN:FP cost ratio (all wells, threshold policy) ===")
    show(res["sensitivity"], list(res["sensitivity"].columns))

    summary = ["# fail_next_30d: cost-aware evaluation (test from 2026-06-27; all choices fitted on validation)",
               f"\nCost parameters: {cp.dict()}; budget {args.budget} wells per 30-day snapshot.\n",
               "## Thresholds\n", md(res["thresholds"], th_cols),
               "\n\n## Top-K slices\n", md(res["topk"], tk_cols),
               "\n\n## Risk tiers\n", md(res["tiers"], ti_cols),
               "\n\n## Cost simulation\n", md(res["policies"], po_cols),
               "\n\n## Cost-ratio sensitivity\n", md(res["sensitivity"], list(res["sensitivity"].columns))]
    (args.out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out_dir}/ (thresholds, topk, tiers, policies, sensitivity .csv + summary.md)")


if __name__ == "__main__":
    main()
