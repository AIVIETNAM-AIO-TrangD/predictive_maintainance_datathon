"""Score fail_next_30d with a trained NLinear checkpoint.

Feature building, scaling and windowing come from src/features.py, the same code src/train.py
trains with (v1 = the src/model.ipynb pipeline, v2 = the improved pipeline). Scores every
producing well-day in [--start, --end] that has a clean lookback window.

Usage (from the workspace root, datathon-2026 env):
    python src/predict.py                                   # v1 model, test period, threshold 0.5
    python src/predict.py --model models/nlinear_fail30_v2.pt --out outputs/predictions_fail_next_30d_v2.parquet
    python src/predict.py --threshold cost                  # cost-optimal cut-off picked on validation
    python src/predict.py --start 2026-04-01                # also score the validation period (for evaluate.py)
    python src/predict.py --forecast                        # one row per well on the last complete day
    python src/predict.py --verify                          # v1 only: reproduce the notebook's test predictions

Columns added for triage:
    fail_next_30d_prob_cal  probability after Platt calibration fitted on validation rows
    oil_rate_30d            recent oil per producing day (<= t), used for the failure-cost estimate
    expected_value          p_cal * prevent_rate * failure_cost - cost_fp   (see src/costs.py)
    priority_rank           1 = act first, ranked by expected_value within each day

Nothing from the test period (date >= the checkpoint's test start) is used to fit anything:
risk-tier cut-offs, calibration and the f1/cost thresholds all come from validation rows.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from costs import (CostParams, apply_calibration, calibrate, cost_optimal_threshold,  # noqa: E402
                   expected_value, failure_cost, oil_rate_30d)
from features import (P, TARGET, build_frame, load_model, score, to_matrix, window_ok)  # noqa: E402

CONTEXT = ["oil_produced_m3", "water_produced_m3", "hours_down", "hours_on_prod", "pump_efficiency_pct", "lost_oil_m3"]


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    o = np.argsort(-p, kind="stable"); ys = y[o]
    ap = float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / max(ys.sum(), 1))
    r = pd.Series(p).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    auc = float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
    return {"rows": len(y), "positive_rate": y.mean(), "accuracy": (tp + tn) / len(y), "precision": prec,
            "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9), "pr_auc": ap, "roc_auc": auc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def best_f1_threshold(y, p):
    o = np.argsort(-p, kind="stable"); ys, ps = y[o], p[o]
    tp = np.cumsum(ys); prec = tp / np.arange(1, len(ys) + 1); rec = tp / ys.sum()
    return float(ps[(2 * prec * rec / np.maximum(prec + rec, 1e-9)).argmax()])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("models/nlinear_fail30.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs/predictions_fail_next_30d.parquet"))
    ap.add_argument("--start", help="first day to score (default: checkpoint test start)")
    ap.add_argument("--end", help="last day to score (default: end of data)")
    ap.add_argument("--threshold", default="0.5",
                    help="probability cut for fail_next_30d_pred: a number, 'f1' or 'cost' (both picked on validation)")
    ap.add_argument("--high-quantile", type=float, default=0.95,
                    help="risk_tier=high above this quantile of validation-period probabilities (default top 5%%)")
    ap.add_argument("--medium-quantile", type=float, default=0.80, help="risk_tier=medium above this quantile")
    ap.add_argument("--cost-fn", type=float, default=CostParams.cost_fn, help="cost of a missed failure")
    ap.add_argument("--cost-fp", type=float, default=CostParams.cost_fp, help="cost of acting on a well")
    ap.add_argument("--oil-value", type=float, default=CostParams.oil_value_per_m3, help="value per lost m3 of oil")
    ap.add_argument("--downtime-days", type=float, default=CostParams.downtime_days)
    ap.add_argument("--prevent-rate", type=float, default=CostParams.prevent_rate)
    ap.add_argument("--verify", action="store_true", help="v1: compare with data/processed/predictions_test.parquet")
    ap.add_argument("--forecast", action="store_true",
                    help="score each well once, on the anchor day, for failures in the next 30 days")
    ap.add_argument("--anchor", help="forecast anchor day (default: last fully reported day of the data)")
    args = ap.parse_args()
    if args.forecast and args.out == ap.get_default("out"):
        args.out = Path("outputs/forecast_fail_next_30d.parquet")
    cp = CostParams(args.cost_fn, args.cost_fp, args.oil_value, args.downtime_days, args.prevent_rate)

    os.chdir(ROOT)                                   # every path is relative to the workspace root
    torch.set_num_threads(4)
    ck = torch.load(args.model, weights_only=False)
    fs, L = ck.get("feature_set", "v1"), ck["lookback"]
    sp = ck["splits"]
    train_end, (val_start, val_end), test_start = (pd.Timestamp(sp["train_end"]), map(pd.Timestamp, sp["val"]),
                                                   pd.Timestamp(sp["test_start"]))
    start = pd.Timestamp(args.start) if args.start else test_start
    print(f"model {args.model}: feature set {fs}, lookback {L}, {len(ck['channels'])} channels")

    m, pos, built = build_frame(fs)
    channels = ck["channels"]
    assert set(channels) <= set(built), "feature pipeline does not provide the checkpoint's channels"
    ok = window_ok(m, pos, L)
    X, cats = to_matrix(m, channels, ck["scaler"], ck["cats"], ck["vocab"])
    model = load_model(ck, len(channels))
    m["oil_rate_30d"] = oil_rate_30d(m).to_numpy()

    date = m.production_date
    end = pd.Timestamp(args.end) if args.end else date.max()
    producing = m.well_daily_status.eq("Producing").to_numpy(dtype=bool, na_value=False)
    held = m.split.eq("test_well").to_numpy(dtype=bool, na_value=False)
    not_failing = m.failure_active.eq(0).to_numpy(dtype=bool, na_value=False)
    labelled = m[TARGET].notna().to_numpy()
    val_wells = set(ck.get("val_wells", []))

    # everything fitted here uses validation-period rows only (never the test period)
    val_rows = np.flatnonzero(producing & not_failing & labelled & ok & ~held
                              & ((date >= val_start) & (date <= val_end)).to_numpy())
    zv = score(model, X, cats, val_rows, L, logits=True)
    pv, yv = 1 / (1 + np.exp(-zv)), m[TARGET].to_numpy()[val_rows].astype(int)
    hi_cut, med_cut = np.quantile(pv, args.high_quantile), np.quantile(pv, args.medium_quantile)
    ab = calibrate(zv, yv)
    fc_val = failure_cost(m.oil_rate_30d.to_numpy()[val_rows], cp)
    thr_opts = {"f1": best_f1_threshold(yv, pv),
                "cost": cost_optimal_threshold(yv, pv, cp, fc_val)}
    thr = thr_opts[args.threshold] if args.threshold in thr_opts else float(args.threshold)
    print(f"validation rows {len(val_rows):,}: tiers high >= {hi_cut:.3f}, medium >= {med_cut:.3f}; "
          f"f1-threshold {thr_opts['f1']:.3f}, cost-threshold {thr_opts['cost']:.3f}; using {thr:.3f}")

    if args.forecast:
        recent = producing & (date >= date.max() - pd.Timedelta(days=60)).to_numpy()
        raw = pd.read_parquet(P / "dashboard.parquet", columns=["uwi", "production_date", "oil_produced_m3", "water_produced_m3"],
                              filters=[("production_date", ">=", (date.max() - pd.Timedelta(days=60)).to_pydatetime())])
        rz = m.loc[recent, ["uwi", "production_date"]].merge(raw, on=["uwi", "production_date"], how="left")
        zero = (rz.oil_produced_m3.eq(0) & rz.water_produced_m3.eq(0)).groupby(rz.production_date).mean()
        complete = zero[zero <= 2 * zero.median()].index
        anchor = pd.Timestamp(args.anchor) if args.anchor else complete.max()
        if anchor not in complete:
            print(f"WARNING: anchor {anchor.date()} is partly reported; its forecasts are unreliable")
        start = end = anchor
        at = (date == anchor).to_numpy()
        print(f"forecast anchor {anchor.date()} -> failure window {(anchor + pd.Timedelta(days=1)).date()} .. "
              f"{(anchor + pd.Timedelta(days=30)).date()} (data ends {date.max().date()}); wells on anchor day "
              f"{at.sum():,}, producing {(at & producing).sum():,}, scored {(at & producing & ok).sum():,}")

    rows = np.flatnonzero(producing & ok & ((date >= start) & (date <= end)).to_numpy())
    prob = score(model, X, cats, rows, L)
    p_cal = apply_calibration(prob, ab)
    fc = failure_cost(m.oil_rate_30d.to_numpy()[rows], cp)

    d = date.iloc[rows]
    role = np.select([d <= train_end, (d >= val_start) & (d <= val_end), d >= test_start],
                     ["train_period", "val_period", "test_period"], default="purge_gap")
    uw = m.uwi.iloc[rows].to_numpy()
    out = pd.DataFrame({
        "uwi": uw, "production_date": d.to_numpy(),
        "well_daily_status": m.well_daily_status.iloc[rows].to_numpy(),
        "field_operating_area": m.field_operating_area.iloc[rows].to_numpy(),
        "fail_next_30d_prob": prob.astype("float32"),
        "fail_next_30d_pred": (prob >= thr).astype("int8"),
        "risk_tier": np.select([prob >= hi_cut, prob >= med_cut], ["high", "medium"], default="low"),
        "fail_next_30d_prob_cal": p_cal.astype("float32"),
        "oil_rate_30d": m.oil_rate_30d.to_numpy()[rows],
        "expected_value": expected_value(p_cal, fc, cp).astype("float32"),
        TARGET: m[TARGET].to_numpy()[rows],
        "period": role,
        "well_group": np.where(held[rows], "held_out_well",
                               np.where(pd.Series(uw).isin(val_wells), "val_well", "training_well")),
        # labelled rows from the evaluation population: label known and the well not already failing
        "eval_row": labelled[rows] & not_failing[rows],
    })
    if "incomplete_reporting_day" in m:
        out["incomplete_reporting_day"] = m.incomplete_reporting_day.to_numpy()[rows]
    del X, m

    wells = pd.read_parquet(P / "wells.parquet", columns=["uwi", "formatted_uwi", "facility_name"])
    ctx = pd.read_parquet(P / "dashboard.parquet", columns=["uwi", "production_date"] + CONTEXT,
                          filters=[("production_date", ">=", start.to_pydatetime())])
    out = out.merge(wells, on="uwi", how="left").merge(ctx, on=["uwi", "production_date"], how="left")
    out["priority_rank"] = out.groupby("production_date").expected_value.rank(ascending=False, method="first").astype("int32")

    if "incomplete_reporting_day" not in out:   # v1 pipeline does not flag them itself
        zero = (out.oil_produced_m3.eq(0) & out.water_produced_m3.eq(0)).groupby(out.production_date).mean()
        bad = zero[(zero > 2 * zero.median()) & (zero > 0.4)].index
        out["incomplete_reporting_day"] = out.production_date.isin(bad)
    bad_days = sorted(out.loc[out.incomplete_reporting_day, "production_date"].dt.date.astype(str).unique())
    if bad_days:
        print(f"partly-reported day(s) flagged: {bad_days}"
              + (" (v2: their zeros were treated as missing, not shutdowns)" if fs == "v2" else ""))

    cols = ["uwi", "formatted_uwi", "production_date", "field_operating_area", "facility_name", "well_daily_status",
            "fail_next_30d_prob", "fail_next_30d_pred", "risk_tier", "fail_next_30d_prob_cal", "expected_value",
            "priority_rank", "oil_rate_30d"] + CONTEXT + [TARGET, "eval_row", "period", "well_group",
                                                          "incomplete_reporting_day"]
    out = out[cols].sort_values(["uwi", "production_date"]).reset_index(drop=True)
    if args.forecast:
        out.insert(3, "forecast_window_start", out.production_date + pd.Timedelta(days=1))
        out.insert(4, "forecast_window_end", out.production_date + pd.Timedelta(days=30))
        out = out.sort_values("priority_rank").reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    last_complete = out.loc[~out.incomplete_reporting_day, "production_date"].max()
    preview = out[out.production_date == last_complete].sort_values("priority_rank").head(200)
    preview_path = args.out.with_name(args.out.stem + "_preview.csv")
    preview.to_csv(preview_path, index=False)
    print(f"\nwrote {args.out} ({len(out):,} rows, {out.uwi.nunique()} wells, "
          f"{out.production_date.min().date()} .. {out.production_date.max().date()})")
    print(f"wrote {preview_path} (last complete day {last_complete.date()}, top 200 by priority)")
    print("periods scored:", out.period.value_counts().to_dict())

    ev = out[out.eval_row & out.period.eq("test_period")]
    if args.forecast:
        print(f"\nforecast mode: no metrics. Tiers: {out.risk_tier.value_counts().to_dict()}; "
              f"pred=1 at threshold {thr:.3f}: {int(out.fail_next_30d_pred.sum())}; EV>0: {int((out.expected_value > 0).sum())}")
    elif ev.empty:
        print("\nno labelled test-period rows in range: metrics skipped")
    else:
        res = {}
        for name, sub in [("test, all wells", ev), ("test, seen wells", ev[ev.well_group != "held_out_well"]),
                          ("test, held-out wells", ev[ev.well_group == "held_out_well"])]:
            if len(sub):
                res[name] = metrics(sub[TARGET].to_numpy().astype(int), sub.fail_next_30d_prob.to_numpy(), thr)
        tbl = pd.DataFrame(res).T
        print(f"\nmetrics at threshold {thr:.3f} (labelled test-period rows only):")
        print(tbl[["rows", "positive_rate", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]]
              .to_string(float_format=lambda v: f"{v:.3f}"))
        a = res["test, all wells"]
        print("\nconfusion matrix (test, all wells):")
        print(pd.DataFrame([[a["tn"], a["fp"]], [a["fn"], a["tp"]]], index=["actual 0", "actual 1"],
                           columns=["pred 0", "pred 1"]).to_string())

    if args.verify:
        ref = pd.read_parquet(P / "predictions_test.parquet")
        j = ref.merge(out[["uwi", "production_date", "fail_next_30d_prob"]], on=["uwi", "production_date"])
        diff = (j.risk - j.fail_next_30d_prob).abs().max()
        print(f"\nverify: {len(j):,} of {len(ref):,} notebook test rows matched; max |prob diff| = {diff:.2e}")
        assert len(j) == len(ref) and diff < 1e-5, "inference does not reproduce the notebook's predictions"


if __name__ == "__main__":
    main()
