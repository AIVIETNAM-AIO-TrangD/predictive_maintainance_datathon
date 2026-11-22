import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")   # run with the datathon-2026 env
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import costs  # noqa: E402
import features  # noqa: E402


def test_sanitize_text_repairs_mojibake_bytes_and_control_chars():
    s = pd.Series(["Rods â€“ broken", b"Pump\x00 failure", "  Tubing\t\nleak ", None, "Ã©lastomer", "\x07"])
    out = features.sanitize_text(s).tolist()
    assert out[0] == "Rods – broken"
    assert out[1] == "Pump failure"
    assert out[2] == "Tubing leak"
    assert pd.isna(out[3])
    assert out[4] == "élastomer"
    assert pd.isna(out[5])                     # only a control character -> empty -> NaN


def test_outcome_costs_and_saving():
    cp = costs.CostParams(cost_fn=5, cost_fp=1)
    y = np.array([1, 1, 0, 0])
    act = np.array([1, 0, 1, 0])
    r = costs.outcome_costs(y, act, np.full(4, 5.0), cp)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 1)
    assert r["total_cost"] == 1 + 5 + 1          # TP pays the action, FN the failure, FP the action
    assert r["do_nothing_cost"] == 10 and r["saving"] == 3


def test_cost_optimal_threshold_acts_only_when_worth_it():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    cheap_miss = costs.CostParams(cost_fn=0.5, cost_fp=1)      # missing a failure is cheaper than acting
    assert costs.cost_optimal_threshold(y, p, cheap_miss) == np.inf
    costly_miss = costs.CostParams(cost_fn=5, cost_fp=1)
    assert costs.cost_optimal_threshold(y, p, costly_miss) == pytest.approx(0.8)


def test_failure_cost_adds_lost_oil_value():
    cp = costs.CostParams(cost_fn=10, cost_fp=1, oil_value_per_m3=2, downtime_days=3)
    assert costs.failure_cost(np.array([0.0, 5.0, np.nan]), cp).tolist() == [10, 40, 10]


def test_calibration_recovers_base_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(float)
    raw_logit = np.where(y == 1, 1.0, -0.5) + rng.normal(0, 1, 5000) + 2.0      # inflated scores
    ab = costs.calibrate(raw_logit, y)
    p = costs.apply_calibration(1 / (1 + np.exp(-raw_logit)), ab)
    assert abs(p.mean() - y.mean()) < 0.01


def test_rolling_never_crosses_wells():
    pos = np.array([0, 1, 2, 0, 1, 2])
    r = features.roll(pd.Series([1, 1, 1, 100, 100, 100], dtype="float32"), 3, "sum", 3, pos)
    assert np.isnan(r[:2]).all() and r[2] == 3 and np.isnan(r[3:5]).all() and r[5] == 300
