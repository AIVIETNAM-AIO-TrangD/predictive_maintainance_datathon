"""Exact per-well risk drivers for the NLinear model (for the dashboard).

NLinear is linear in its inputs, so the score splits exactly into one additive term per channel:

    logit = const + sum_c contribution_c + field term + pump-make term
    contribution_c = sum_j H[c, j] * ( (W @ (x_c - x_c[-1]))_j + x_c[-1] )

where x_c is the channel's scaled window (0 = the training median), W/b the shared temporal layer and
H the head weights for that channel. A contribution is therefore "how much this channel moves the risk
compared with a typical well whose value sits at the training median for the whole window".
Contributions are reported on the calibrated log-odds scale (Platt scaling fitted on validation rows,
the same as src/predict.py), so positive = pushes the failure probability up.

Usage (workspace root, datathon-2026 env):
    python src/explain.py                         # v1 (the dashboard model)
    python src/explain.py --model models/nlinear_fail30_v2.pt --pred outputs/predictions_fail_next_30d_v2.parquet --tag v2
Writes:
    outputs/drivers_channels_<tag>.parquet  one row per (well, channel) on the last complete day
    outputs/drivers_groups_<tag>.parquet    one row per scored well-day, one column per factor group
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
from costs import calibrate  # noqa: E402
from features import TARGET, build_frame, load_model, score, to_matrix, window_ok  # noqa: E402

GROUPS = {
    "Failure history": lambda c: c.startswith("fail_count") or c == "days_since_last_failure",
    "Repair & job history": lambda c: c.startswith("job_count") or c.startswith("days_since_last_"),
    "Water & water cut": lambda c: c.startswith("water") or c == "gas_produced_e3m3",
    "Oil production": lambda c: c.startswith("oil"),
    "Pump load (RPM, torque, amps)": lambda c: c.startswith(("pcp_pump", "rpm_", "torque_", "amps_", "throughput_",
                                                             "has_amps", "has_torque")),
    "Pump efficiency": lambda c: c.startswith(("pump_efficiency", "eff_", "has_eff")),
    "Downtime & run hours": lambda c: c.startswith(("hours", "downtime")),
    "Pressures": lambda c: c.endswith("pressure_kpa"),
    "Well age & location": lambda c: c in ("well_age_years", "surface_latitude", "surface_longitude"),
}


def group_of(c: str) -> str:
    for g, f in GROUPS.items():
        if f(c):
            return g
    return "Other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("models/nlinear_fail30.pt"))
    ap.add_argument("--pred", type=Path, default=Path("outputs/predictions_fail_next_30d_v1.parquet"))
    ap.add_argument("--tag", default="v1", help="suffix of the output files (drivers_*_<tag>.parquet)")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)

    ck = torch.load(args.model, weights_only=False)
    L, chans, cats = ck["lookback"], ck["channels"], ck["cats"]
    m, pos, built = build_frame(ck.get("feature_set", "v1"))
    assert set(chans) <= set(built)
    ok = window_ok(m, pos, L)
    X, codes = to_matrix(m, chans, ck["scaler"], cats, ck["vocab"])
    model = load_model(ck, len(chans))

    # calibration exactly as src/predict.py: Platt on validation rows of non-test wells
    sp = ck["splits"]
    vs, ve = map(pd.Timestamp, sp["val"])
    date = m.production_date
    producing = m.well_daily_status.eq("Producing").to_numpy(dtype=bool, na_value=False)
    held = m.split.eq("test_well").to_numpy(dtype=bool, na_value=False)
    val_rows = np.flatnonzero(producing & m.failure_active.eq(0).to_numpy(dtype=bool, na_value=False)
                              & m[TARGET].notna().to_numpy() & ok & ~held & ((date >= vs) & (date <= ve)).to_numpy())
    a, b = calibrate(score(model, X, codes, val_rows, L, logits=True), m[TARGET].to_numpy()[val_rows].astype(int))

    # rows to explain = rows scored in the prediction file
    pred = pd.read_parquet(args.pred, columns=["uwi", "production_date", "incomplete_reporting_day"])
    key = pd.DataFrame({"uwi": m.uwi.to_numpy(), "production_date": date.to_numpy(), "row": np.arange(len(m))})
    rows = pred.merge(key, on=["uwi", "production_date"], how="inner")
    rows = rows[ok[rows.row.to_numpy()]].reset_index(drop=True)
    r_idx = rows.row.to_numpy()

    sd = model.state_dict()
    W, bt = sd["temporal.weight"].numpy(), sd["temporal.bias"].numpy()          # (d, L), (d,)
    C, d = len(chans), W.shape[0]
    w_head, b_head = sd["head.1.weight"].numpy()[0], float(sd["head.1.bias"].numpy()[0])
    H = w_head[: C * d].reshape(C, d)
    const = b_head + float((H @ bt).sum())
    emb_terms = []
    off = C * d
    for k, c in enumerate(cats):
        E = sd[f"embs.{k}.weight"].numpy()                                        # (n+1, emb)
        hk = w_head[off: off + E.shape[1]]; off += E.shape[1]
        emb_terms.append((c, E @ hk))                                             # term per category code

    groups = [group_of(c) for c in chans]
    contrib = np.empty((len(r_idx), C), dtype="float32")
    offs = np.arange(-L + 1, 1)
    for i in range(0, len(r_idx), 8192):
        win = X[r_idx[i:i + 8192, None] + offs]                                   # (B, L, C)
        last = win[:, -1, :]
        zt = np.einsum("blc,dl->bcd", win - last[:, None, :], W) + last[:, :, None]
        contrib[i:i + 8192] = np.einsum("bcd,cd->bc", zt, H)
    cat_contrib = {c: t[codes[r_idx, k]] for k, (c, t) in enumerate(emb_terms)}
    logit = const + contrib.sum(1) + sum(cat_contrib.values())

    check = score(model, X, codes, r_idx[:5000], L, logits=True)
    err = np.abs(check - logit[:5000]).max()
    print(f"decomposition check on 5,000 rows: max |logit difference| = {err:.2e}")
    assert err < 1e-3, "contributions do not add up to the model output"

    # centre the field / pump-make terms on the fleet average so "typical" means typical
    for c in cats:
        mean = cat_contrib[c].mean(); cat_contrib[c] = cat_contrib[c] - mean; const += mean

    # calibrated log-odds scale: a * logit + b
    g = pd.DataFrame(a * contrib, columns=chans).T.groupby(groups).sum().T
    g["Field & pump make"] = a * sum(cat_contrib.values())
    g.insert(0, "baseline", a * const + b)
    g.insert(0, "production_date", rows.production_date.to_numpy())
    g.insert(0, "uwi", rows.uwi.to_numpy())
    g["risk_logit_cal"] = a * logit + b
    g.astype({c: "float32" for c in g.columns if c not in ("uwi", "production_date")}).to_parquet(
        args.out_dir / f"drivers_groups_{args.tag}.parquet", index=False)

    last_day = rows.loc[~rows.incomplete_reporting_day, "production_date"].max()
    sel = np.flatnonzero(rows.production_date.eq(last_day).to_numpy())
    raw = m.iloc[r_idx[sel]][chans].reset_index(drop=True)
    long = pd.DataFrame({
        "uwi": np.repeat(rows.uwi.to_numpy()[sel], C),
        "production_date": last_day,
        "channel": np.tile(chans, len(sel)),
        "group": np.tile(groups, len(sel)),
        "value": raw.to_numpy(dtype="float32").ravel(),
        "training_median": np.tile([ck["scaler"][c]["median"] if not ck["scaler"][c]["log1p"]
                                    else float(np.expm1(ck["scaler"][c]["median"])) for c in chans], len(sel)),
        "contribution": (a * contrib[sel]).ravel(),
    })
    for k, c in enumerate(cats):
        inv = {v: kk for kk, v in ck["vocab"][c].items()}
        long = pd.concat([long, pd.DataFrame({
            "uwi": rows.uwi.to_numpy()[sel], "production_date": last_day, "channel": c, "group": "Field & pump make",
            "value": np.nan, "category": [inv.get(v, "unknown") for v in codes[r_idx[sel], k]],
            "contribution": a * cat_contrib[c][sel]})], ignore_index=True)
    long.to_parquet(args.out_dir / f"drivers_channels_{args.tag}.parquet", index=False)
    print(f"wrote {args.out_dir / f'drivers_groups_{args.tag}.parquet'} ({len(g):,} well-days) and "
          f"{args.out_dir / f'drivers_channels_{args.tag}.parquet'} ({len(sel):,} wells on {last_day.date()}); "
          f"calibration a={a:.3f}, b={b:.3f}")


if __name__ == "__main__":
    main()
