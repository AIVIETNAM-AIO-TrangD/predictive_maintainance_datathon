"""Does training on 2021-2026 (repair-job proxy labels before 2025-06) beat training on 9 months?

Usage (workspace root, datathon-2026 env; needs data/processed/merged_v3.parquet from
`python src/merge.py --raw <raw dir> --proxy-before 2025-06-01`):
    python src/train_proxy.py

Same model, hyper-parameters and split as the original v1 notebook model (lookback 30, D 8,
pos_weight neg/pos, weight decay 1e-4, dropout 0.1, field + pump-make embeddings, the 237 test wells
excluded). Only the training period / labels change:
    B  2025-06-01 .. 2026-03-01  recorded failures only (control: same period as v1)
    C  2021-04-01 .. 2026-03-01  repair-job proxies before 2025-06, recorded failures after
    D  as C, proxy-era rows weighted 0.5 (they are noisier)
Selection is on validation PR-AUC with recorded-failure labels (2026-04-01 .. 2026-05-27). Test rows are
never scored here. The selected model is saved as models/nlinear_fail30_v3.pt.
"""
from __future__ import annotations

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
from features import CATS, MAX_LOOKBACK, TARGET, NLinearClassifier, build_frame, fit_scaler, to_matrix, window_ok  # noqa: E402

SEED, L, D = 42, 30, 8
PROXY_END = pd.Timestamp("2025-06-01")
TEST_START, VAL_START = pd.Timestamp("2026-06-27"), pd.Timestamp("2026-04-01")
GAP = pd.Timedelta(days=30)
TRAIN_END, VAL_END = VAL_START - GAP - pd.Timedelta(days=1), TEST_START - GAP - pd.Timedelta(days=1)
VARIANTS = {"B: 9 months, recorded labels": (pd.Timestamp("2025-06-01"), 1.0),
            "C: 2021-2026, proxy + recorded labels": (pd.Timestamp("2021-04-01"), 1.0),
            "D: as C, proxy rows weight 0.5": (pd.Timestamp("2021-04-01"), 0.5)}


def ap_score(y, p):
    o = np.argsort(-p, kind="stable"); ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / max(ys.sum(), 1))


def auc_score(y, p):
    r = pd.Series(p).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main() -> None:
    os.chdir(ROOT)
    torch.set_num_threads(6)
    out_dir = Path("outputs/experiments"); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m, pos, channels = build_frame("v3")
    date = m.production_date
    held = m.split.eq("test_well").to_numpy(dtype=bool, na_value=False)
    ok = window_ok(m, pos, MAX_LOOKBACK)
    usable = (m[TARGET].notna() & m.failure_active.eq(0) & m.well_daily_status.eq("Producing")).to_numpy(
        dtype=bool, na_value=False) & ok
    in_range = lambda a, b: ((date >= a) & (date <= b)).to_numpy()  # noqa: E731
    val = np.flatnonzero(usable & ~held & in_range(VAL_START, VAL_END))
    assert date.iloc[val].max() + GAP < TEST_START
    y_all = m[TARGET].fillna(0).to_numpy("float32").copy()
    yv = y_all[val].astype(int)
    print(f"frame {m.shape} in {time.time() - t0:.0f}s; val rows {len(val):,} (positive rate {yv.mean():.3f})", flush=True)

    results, best = [], None
    for name, (start, proxy_w) in VARIANTS.items():
        tr = np.flatnonzero(usable & ~held & in_range(start, TRAIN_END))
        fit_rows = (~held) & in_range(start - pd.Timedelta(days=MAX_LOOKBACK), TRAIN_END)
        scaler = fit_scaler(m, channels, fit_rows)
        vocab = {c: {v: i + 1 for i, v in enumerate(sorted(m.loc[fit_rows, c].dropna().unique()))} for c in CATS}
        X, codes = to_matrix(m, channels, scaler, CATS, vocab)
        X_t, C_t, Y_t = torch.from_numpy(X), torch.from_numpy(codes), torch.from_numpy(y_all)
        W_t = torch.from_numpy(np.where(date.to_numpy() < PROXY_END, proxy_w, 1.0).astype("float32"))
        off = torch.arange(-L + 1, 1)

        def predict(model, rows):
            model.eval(); out = []
            with torch.no_grad():
                for i in range(0, len(rows), 8192):
                    r = torch.as_tensor(rows[i:i + 8192])
                    out.append(torch.sigmoid(model(X_t[r[:, None] + off], C_t[r])).numpy())
            return np.concatenate(out)

        torch.manual_seed(SEED)
        model = NLinearClassifier(L, len(channels), D, n_cats=[len(vocab[c]) for c in CATS], dropout=0.1)
        ytr = y_all[tr]
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - ytr.mean()) / ytr.mean()), reduction="none")
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        rng = np.random.default_rng(SEED)
        best_ap, state, wait, ep, start_t = -1.0, None, 0, 0, time.time()
        for ep in range(1, 41):
            model.train()
            order = rng.permutation(tr)
            for i in range(0, len(order), 2048):
                r = torch.as_tensor(order[i:i + 2048])
                loss = (loss_fn(model(X_t[r[:, None] + off], C_t[r]), Y_t[r]) * W_t[r]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            ap = ap_score(yv, predict(model, val))
            print(f"  {name} epoch {ep:2d}: val PR-AUC {ap:.4f}", flush=True)
            if ap > best_ap:
                best_ap, wait, state = ap, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 5:
                    break
        model.load_state_dict(state)
        pv = predict(model, val)
        top = np.argsort(-pv)[: len(pv) // 20]
        res = {"variant": name, "train_start": str(start.date()), "train_rows": len(tr),
               "train_wells": int(m.uwi.iloc[tr].nunique()), "positive_rate": float(ytr.mean()),
               "proxy_weight": proxy_w, "epochs": ep, "minutes": round((time.time() - start_t) / 60, 1),
               "val_pr_auc": best_ap, "val_roc_auc": auc_score(yv, pv), "val_top5_precision": float(yv[top].mean())}
        results.append(res)
        print(f"{name}: {res}", flush=True)
        pd.DataFrame(results).to_csv(out_dir / "proxy_experiment.csv", index=False)
        if best is None or best_ap > best[0]:
            best = (best_ap, name, {"state_dict": model.state_dict(), "feature_set": "v3", "channels": channels,
                                    "scaler": scaler, "cats": CATS, "vocab": vocab, "lookback": L, "d": D,
                                    "individual": False, "dropout": 0.1, "experiment": name,
                                    "train_start": str(start.date()), "proxy_weight": proxy_w,
                                    "splits": {"train_end": str(TRAIN_END.date()),
                                               "val": [str(VAL_START.date()), str(VAL_END.date())],
                                               "test_start": str(TEST_START.date())}})
        del X, X_t

    torch.save(best[2], "models/nlinear_fail30_v3.pt")
    (out_dir / "proxy_selected.json").write_text(json.dumps({"selected": best[1], "val_pr_auc": best[0]}, indent=2))
    print(f"\nselected {best[1]} (val PR-AUC {best[0]:.4f}); saved models/nlinear_fail30_v3.pt "
          f"({(time.time() - t0) / 60:.0f} min total)")


if __name__ == "__main__":
    main()
