"""Train NLinear variants for fail_next_30d and pick one on validation only.

Usage (workspace root, datathon-2026 env):
    python src/train.py                       # staged experiments, saves the selected model
    python src/train.py --quick               # 3 epochs per run (smoke test)

Protocol (no test information is used anywhere in this script; it never scores a test-period row):
  * train  2025-06-01 .. 2026-03-01 | 30-day purge | val 2026-04-01 .. 2026-05-27 | 30-day purge | test >= 2026-06-27
  * the 237 held-out test wells are excluded from everything here
  * a further 15% of the remaining wells ("val wells", seed 42) are excluded from training and
    used only in validation, so unseen-well recall can be tuned without touching the test wells
  * scaler statistics come from training-period rows of training wells only
  * every variant is trained on the same sample set (windows valid for the longest lookback, 60 days)

Stages (greedy, each stage keeps the best validation PR-AUC so far):
  1 feature set  v1 channels vs v2 channels (src/features.py)
  2 pos_weight   1, sqrt(neg/pos), neg/pos
  3 lookback     14, 30, 60 days
  4 regularisation  weight decay 1e-4 / 1e-3 / 1e-2, dropout 0.1 / 0.3
  5 categorical inputs  field + pump make, make only, none
  6 Winter concentration  drop location inputs (lat/lon + field); field-balanced loss weights (share^-0.5, share^-1)
Final pick: highest validation PR-AUC among runs whose Winter share of the top-5% validation risk is at
most 2x Winter's share of validation failures (falls back to the best PR-AUC if none qualify).

Outputs: outputs/experiments/experiment_results.csv (+ .md), models/nlinear_fail30_v2.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from costs import CostParams, cost_optimal_threshold, failure_cost, oil_rate_30d  # noqa: E402
from features import (CATS, LABEL_START, MAX_LOOKBACK, TARGET, V2_EXTRA, NLinearClassifier,  # noqa: E402
                      build_frame, fit_scaler, to_matrix, window_ok)

SEED = 42
HORIZON = 30
TEST_START = pd.Timestamp("2026-06-27")
VAL_START = pd.Timestamp("2026-04-01")
GAP = pd.Timedelta(days=HORIZON)
TRAIN_END = VAL_START - GAP - pd.Timedelta(days=1)
VAL_END = TEST_START - GAP - pd.Timedelta(days=1)
VAL_WELL_FRAC = 0.15
CONCENTRATION_LIMIT = 2.0


def ap_score(y, p):
    o = np.argsort(-p, kind="stable"); ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / max(ys.sum(), 1))


def auc_score(y, p):
    r = pd.Series(p).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def f1_at_best(y, p):
    o = np.argsort(-p, kind="stable"); ys, ps = y[o], p[o]
    tp = np.cumsum(ys); prec = tp / np.arange(1, len(ys) + 1); rec = tp / ys.sum()
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9); k = int(f1.argmax())
    return float(ps[k]), float(f1[k])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-model", type=Path, default=Path("models/nlinear_fail30_v2.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/experiments"))
    args = ap.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    m, pos, full_channels = build_frame("v2")
    print(f"frame {m.shape} built in {time.time() - t0:.0f}s; partly-reported days masked: {m.attrs['partly_reported_days']}")
    date = m.production_date
    held = m.split.eq("test_well").to_numpy(dtype=bool, na_value=False)
    wells = np.sort(m.uwi[~held].unique())
    rng = np.random.default_rng(SEED)
    val_wells = set(rng.choice(wells, size=int(round(len(wells) * VAL_WELL_FRAC)), replace=False))
    is_vw = m.uwi.isin(val_wells).to_numpy()

    ok = window_ok(m, pos, MAX_LOOKBACK)
    usable = (m[TARGET].notna() & m.failure_active.eq(0) & m.well_daily_status.eq("Producing")).to_numpy(
        dtype=bool, na_value=False) & ok
    in_range = lambda a, b: ((date >= a) & (date <= b)).to_numpy()  # noqa: E731
    idx = {"train": np.flatnonzero(usable & ~held & ~is_vw & in_range(LABEL_START, TRAIN_END)),
           "val": np.flatnonzero(usable & ~held & in_range(VAL_START, VAL_END))}
    assert date.iloc[idx["val"]].max() < TEST_START and not held[idx["train"]].any() and not held[idx["val"]].any()
    val_vw = is_vw[idx["val"]]
    y_all = m[TARGET].fillna(0).to_numpy("float32").copy()
    yv = y_all[idx["val"]].astype(int)
    winter = m.field_operating_area.eq("Winter").to_numpy(dtype=bool, na_value=False)
    wv = winter[idx["val"]]
    winter_pos_share = float(wv[yv == 1].mean())
    oil_rate = oil_rate_30d(m).to_numpy()
    fc_val = failure_cost(oil_rate[idx["val"]], CostParams())
    print({k: len(v) for k, v in idx.items()}, f"val wells {len(val_wells)} ({val_vw.sum():,} val rows); "
          f"Winter share of val failures {winter_pos_share:.2f}")

    fit_rows = (~held) & (~is_vw) & (date <= TRAIN_END).to_numpy()
    scaler = fit_scaler(m, full_channels, fit_rows)
    vocab = {c: {v: i + 1 for i, v in enumerate(sorted(m.loc[fit_rows, c].dropna().unique()))} for c in CATS}
    X, codes = to_matrix(m, full_channels, scaler, CATS, vocab)
    X_t, C_t, Y_t = torch.from_numpy(X), torch.from_numpy(codes), torch.from_numpy(y_all)
    v1_channels = [c for c in full_channels if c not in V2_EXTRA]

    field_codes = codes[:, CATS.index("field_operating_area")]

    def run(name, channels, L=30, pw="ratio", wd=1e-4, dropout=0.1, cats=("field_operating_area", "pcp_pump_make"),
            balance=0.0):
        torch.manual_seed(SEED)
        ch = torch.as_tensor([full_channels.index(c) for c in channels])
        cat_ix = torch.as_tensor([CATS.index(c) for c in cats], dtype=torch.long)
        off = torch.arange(-L + 1, 1)

        def batch(rows):
            r = torch.as_tensor(rows)
            return X_t[r[:, None] + off][:, :, ch], C_t[r][:, cat_ix], Y_t[r]

        def predict(model, rows):
            model.eval(); out = []
            with torch.no_grad():
                for i in range(0, len(rows), 8192):
                    xb, cb, _ = batch(rows[i:i + 8192]); out.append(torch.sigmoid(model(xb, cb)).numpy())
            return np.concatenate(out)

        model = NLinearClassifier(L, len(channels), 8, n_cats=[len(vocab[c]) for c in cats], dropout=dropout)
        ytr = y_all[idx["train"]]; ratio = (1 - ytr.mean()) / ytr.mean()
        pos_w = {"1": 1.0, "sqrt": float(np.sqrt(ratio)), "ratio": float(ratio)}[pw]
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w), reduction="none")
        # field balancing: weight_f = share_f^-balance (normalised to mean 1 over training rows)
        share = pd.Series(field_codes[idx["train"]]).value_counts(normalize=True)
        w_field = torch.tensor((share ** -balance / (share ** -balance * share).sum()).reindex(
            range(field_codes.max() + 1)).fillna(1.0).to_numpy(), dtype=torch.float32)
        F_t = torch.from_numpy(field_codes)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=wd)
        r = np.random.default_rng(SEED)
        best, state, wait, ep = -1.0, None, 0, 0
        start = time.time()
        for ep in range(1, (3 if args.quick else 40) + 1):
            model.train()
            order = r.permutation(idx["train"])
            for i in range(0, len(order), 2048):
                xb, cb, yb = batch(order[i:i + 2048])
                loss = (loss_fn(model(xb, cb), yb) * w_field[F_t[torch.as_tensor(order[i:i + 2048])]]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            score = ap_score(yv, predict(model, idx["val"]))
            if score > best:
                best, wait, state = score, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 5:
                    break
        model.load_state_dict(state)
        pv = predict(model, idx["val"])
        thr, f1 = f1_at_best(yv, pv)
        top = np.argsort(-pv)[: len(pv) // 20]
        cthr = cost_optimal_threshold(yv, pv, CostParams(), fc_val)
        act = pv >= cthr
        cost = ((act & (yv == 1)) * CostParams().cost_fp + (act & (yv == 0)) * CostParams().cost_fp
                + (~act & (yv == 1)) * fc_val).sum()
        res = {"run": name, "channels": "v2" if len(channels) > len(v1_channels) else "v1", "n_channels": len(channels),
               "lookback": L, "pos_weight": pw, "weight_decay": wd, "dropout": dropout, "cats": "+".join(cats) or "none", "field_balance": balance,
               "epochs": ep, "seconds": round(time.time() - start),
               "val_pr_auc": best, "val_roc_auc": auc_score(yv, pv), "val_best_f1": f1, "val_f1_threshold": thr,
               "val_wells_pr_auc": ap_score(yv[val_vw], pv[val_vw]),
               "val_wells_recall_at_f1_thr": float((pv[val_vw & (yv == 1)] >= thr).mean()),
               "val_top5_precision": float(yv[top].mean()),
               "winter_share_top5": float(wv[top].mean()),
               "winter_concentration": float(wv[top].mean() / winter_pos_share),
               "val_cost_vs_do_nothing": float(cost / (yv * fc_val).sum())}
        print(f"  {name:28s} AP {best:.4f}  AUC {res['val_roc_auc']:.3f}  F1 {f1:.3f}  "
              f"val-wells AP {res['val_wells_pr_auc']:.3f} recall {res['val_wells_recall_at_f1_thr']:.2f}  "
              f"Winter top5 {res['winter_share_top5']:.2f} (x{res['winter_concentration']:.1f})  "
              f"cost {res['val_cost_vs_do_nothing']:.3f}  [{ep} ep, {res['seconds']}s]", flush=True)
        return res, model, dict(channels=channels, L=L, pw=pw, pos_w=pos_w, wd=wd, dropout=dropout, cats=list(cats),
                                thr=thr, cost_thr=cthr, balance=balance)

    results, models = [], {}

    def keep(out):
        res, model, cfg = out
        results.append(res); models[res["run"]] = (model, cfg)
        pd.DataFrame(results).to_csv(args.out_dir / "experiment_results.csv", index=False)
        return res

    best_of = lambda runs: max(runs, key=lambda r: r["val_pr_auc"])  # noqa: E731
    print("stage 1: feature set")
    s1 = [keep(run("v1 channels", v1_channels)), keep(run("v2 channels", full_channels))]
    b = best_of(s1); ch = models[b["run"]][1]["channels"]
    print("stage 2: pos_weight")
    s2 = [b] + [keep(run(f"{b['channels']} pw={pw}", ch, pw=pw)) for pw in ("1", "sqrt")]
    b = best_of(s2); pw = b["pos_weight"]
    print("stage 3: lookback")
    s3 = [b] + [keep(run(f"{b['channels']} pw={pw} L={L}", ch, L=L, pw=pw)) for L in (14, 60)]
    b = best_of(s3); L = b["lookback"]
    print("stage 4: regularisation")
    s4 = [b] + [keep(run(f"{b['channels']} pw={pw} L={L} wd={wd} do={do}", ch, L=L, pw=pw, wd=wd, dropout=do))
                for wd, do in ((1e-3, 0.1), (1e-2, 0.1), (1e-3, 0.3))]
    b = best_of(s4); wd, do = b["weight_decay"], b["dropout"]
    print("stage 5: categorical inputs")
    for cats in (("pcp_pump_make",), ()):
        keep(run(f"{b['channels']} pw={pw} L={L} wd={wd} do={do} cats={'+'.join(cats) or 'none'}", ch,
                 L=L, pw=pw, wd=wd, dropout=do, cats=cats))

    b = best_of([r for r in results if r["lookback"] == L and r["weight_decay"] == wd and r["dropout"] == do])
    cats_b = () if b["cats"] == "none" else tuple(b["cats"].split("+"))
    print("stage 6: Winter concentration")
    no_loc = [c for c in ch if c not in ("surface_latitude", "surface_longitude")]
    keep(run(f"{b['channels']} best + no location", no_loc, L=L, pw=pw, wd=wd, dropout=do, cats=()))
    for bal in (0.5, 1.0):
        keep(run(f"{b['channels']} best + field balance {bal}", ch, L=L, pw=pw, wd=wd, dropout=do, cats=cats_b,
                 balance=bal))

    res = pd.DataFrame(results)
    ok_conc = res[res.winter_concentration <= CONCENTRATION_LIMIT]
    pick = (ok_conc if len(ok_conc) else res).sort_values("val_pr_auc", ascending=False).iloc[0]
    res["selected"] = res.run.eq(pick.run)
    res.to_csv(args.out_dir / "experiment_results.csv", index=False)
    fmt = res.drop(columns=["val_f1_threshold"]).copy()
    for c in fmt.select_dtypes("float").columns:
        fmt[c] = fmt[c].map(lambda v: f"{v:.3g}")
    try:
        table = fmt.to_markdown(index=False)
    except ImportError:                       # tabulate not installed: plain pipe table
        table = "| " + " | ".join(fmt.columns) + " |\n|" + "---|" * len(fmt.columns) + "\n" + "\n".join(
            "| " + " | ".join(map(str, r)) + " |" for r in fmt.itertuples(index=False))
    (args.out_dir / "experiment_results.md").write_text(
        "# NLinear experiments (validation only)\n\n" + table + "\n", encoding="utf-8")
    print(f"\nselected: {pick.run} (val PR-AUC {pick.val_pr_auc:.4f}, Winter concentration x{pick.winter_concentration:.1f}"
          f"{'' if len(ok_conc) else ' -- no run met the concentration limit'})")

    model, cfg = models[pick.run]
    torch.save({"state_dict": model.state_dict(), "feature_set": "v2", "channels": cfg["channels"],
                "scaler": {c: scaler[c] for c in cfg["channels"]}, "cats": cfg["cats"],
                "vocab": {c: vocab[c] for c in cfg["cats"]}, "lookback": cfg["L"], "d": 8, "individual": False,
                "dropout": cfg["dropout"], "pos_weight": cfg["pos_w"], "weight_decay": cfg["wd"], "field_balance": cfg["balance"],
                "threshold": cfg["thr"], "cost_threshold_default_costs": cfg["cost_thr"],
                "val_wells": sorted(val_wells), "experiment": pick.run,
                "splits": {"train_end": str(TRAIN_END.date()), "val": [str(VAL_START.date()), str(VAL_END.date())],
                           "test_start": str(TEST_START.date())}},
               args.out_model)
    (args.out_dir / "selected.json").write_text(json.dumps(pick.to_dict(), indent=2, default=str))
    print(f"saved {args.out_model}; results in {args.out_dir}/ ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
