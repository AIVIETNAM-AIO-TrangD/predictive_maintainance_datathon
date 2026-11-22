import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import merge  # noqa: E402
import survival  # noqa: E402


def test_kaplan_meier_handles_censoring():
    # 4 runs: failures at 10 and 20, censored at 15 and 30
    km = survival.kaplan_meier(np.array([10, 15, 20, 30]), np.array([1, 0, 1, 0]))
    assert km.survival.tolist() == pytest.approx([1.0, 0.75, 0.375])   # 1-1/4, then x (1-1/2): 2 at risk at t=20
    assert survival.median_of(km) == 20
    assert survival.s_at(km, 12) == pytest.approx(0.75)


def test_proxy_failures_one_event_per_repair_episode():
    d = pd.Timestamp("2023-01-01")
    jobs = pd.DataFrame({
        "uwi": ["A"] * 4 + ["B"],
        "start_date": [d, d + pd.Timedelta(days=3), d + pd.Timedelta(days=40), d + pd.Timedelta(days=900), d],
        "job_group": ["rod_truck", "rod_truck", "pump_change", "rod_truck", "other"],
        "job_type": ["Rod Repair", "Rod Repair", "PCP Replacement", "Rod Repair", "Drilling - original"],
        "job_subtype": [None] * 5,
    })
    px = merge.proxy_failures(jobs, before=pd.Timestamp("2025-06-01"))
    assert px.failure_start_date.tolist() == [d, d + pd.Timedelta(days=40)]   # day-3 job merged; 2025-06+ and drilling dropped
    assert px.failure_group.tolist() == ["Rods", "Pump"]
    assert (px.source == "proxy").all()


def test_pump_make_names_are_normalised():
    s = pd.Series(["Europump", "EURO PUMP", "WEEATHERFORD", " bmw pump ", None])
    assert survival.norm_make(s).tolist() == ["EUROPUMP", "EUROPUMP", "WEATHERFORD", "BMW", "UNKNOWN"]
