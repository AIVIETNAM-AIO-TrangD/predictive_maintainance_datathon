"""Cost-aware decision helpers for fail_next_30d.

Cost model (all values in the same relative "cost units", or in currency if you prefer):
    failure_cost_i = cost_fn + oil_value_per_m3 * downtime_days * oil_rate_i
        cost_fn            fixed cost of an unplanned failure (repair rig, parts, call-out)
        oil_rate_i         the well's recent oil rate (m3 per producing day, last 30 days, <= t)
        downtime_days      expected days lost per failure (default: mean event length before the test period)
        oil_value_per_m3   value of a lost m3 of oil (0 = ignore production loss)
    acting on a well costs cost_fp (inspection / preventive job), whether or not it would have failed;
    acting on a well that would fail avoids `prevent_rate` of its failure cost.

    outcome costs:   TP = cost_fp + (1 - prevent_rate) * failure_cost
                     FP = cost_fp
                     FN = failure_cost
                     TN = 0

Expected value of acting:  EV_i = p_i * prevent_rate * failure_cost_i - cost_fp
(act when EV_i > 0; rank wells by EV_i). p_i must be a calibrated probability: the model is trained
with a positive-class weight, so its raw scores are inflated. calibrate() fits Platt scaling on
validation rows only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class CostParams:
    cost_fn: float = 5.0
    cost_fp: float = 1.0
    oil_value_per_m3: float = 0.0
    downtime_days: float = 3.6
    prevent_rate: float = 1.0

    def dict(self):
        return asdict(self)


def failure_cost(oil_rate: np.ndarray, cp: CostParams) -> np.ndarray:
    return cp.cost_fn + cp.oil_value_per_m3 * cp.downtime_days * np.nan_to_num(np.asarray(oil_rate, float), nan=0.0)


def outcome_costs(y: np.ndarray, act: np.ndarray, fcost: np.ndarray, cp: CostParams) -> dict:
    y, act = np.asarray(y).astype(bool), np.asarray(act).astype(bool)
    tp, fp, fn, tn = act & y, act & ~y, ~act & y, ~act & ~y
    cost = (tp * (cp.cost_fp + (1 - cp.prevent_rate) * fcost) + fp * cp.cost_fp + fn * fcost).sum()
    baseline = (y * fcost).sum()                    # do nothing: every failure costs in full
    return {"tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()), "tn": int(tn.sum()),
            "actions": int(act.sum()), "total_cost": float(cost), "do_nothing_cost": float(baseline),
            "saving": float(baseline - cost), "saving_pct": float((baseline - cost) / baseline) if baseline else np.nan}


def expected_value(p_cal: np.ndarray, fcost: np.ndarray, cp: CostParams) -> np.ndarray:
    return p_cal * cp.prevent_rate * fcost - cp.cost_fp


def cost_optimal_threshold(y: np.ndarray, p: np.ndarray, cp: CostParams, fcost: np.ndarray | None = None) -> float:
    """Single probability cut-off minimising total cost on the rows given (use validation rows only)."""
    fcost = np.full(len(y), cp.cost_fn) if fcost is None else fcost
    o = np.argsort(-p, kind="stable")
    ys, fs, ps = np.asarray(y)[o].astype(float), fcost[o], p[o]
    # acting on the top k rows: cost(k) = sum_{i<k} [cp.cost_fp + y_i*(1-prevent)*f_i] + sum_{i>=k} y_i*f_i
    act_cost = np.cumsum(cp.cost_fp + ys * (1 - cp.prevent_rate) * fs)
    rest = (ys * fs).sum() - np.cumsum(ys * fs)
    total = np.r_[(ys * fs).sum(), act_cost + rest]
    k = int(total.argmin())
    return float(np.inf) if k == 0 else float(ps[k - 1])


def calibrate(logits_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
    """Platt scaling p = sigmoid(a * logit + b), fitted on validation rows."""
    z, y = torch.tensor(logits_val, dtype=torch.float64), torch.tensor(y_val, dtype=torch.float64)
    a, b = torch.ones(1, dtype=torch.float64, requires_grad=True), torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * z + b, y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(a.item()), float(b.item())


def apply_calibration(p_raw: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    p = np.clip(np.asarray(p_raw, float), 1e-7, 1 - 1e-7)
    return 1 / (1 + np.exp(-(ab[0] * np.log(p / (1 - p)) + ab[1])))


def oil_rate_30d(df: pd.DataFrame) -> pd.Series:
    """Oil per producing day over the last 30 days (<= t), per well. df sorted by uwi/date, daily rows."""
    g = df.groupby("uwi", sort=False)
    oil = g["oil_produced_m3"].transform(lambda s: s.rolling(30, min_periods=5).sum())
    days = g["hours_on_prod"].transform(lambda s: (s > 0).astype(float).rolling(30, min_periods=5).sum())
    return (oil / days.where(days > 0)).astype("float32")
