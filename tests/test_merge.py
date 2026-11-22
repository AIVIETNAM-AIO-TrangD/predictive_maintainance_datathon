import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import merge  # noqa: E402

D = pd.Timestamp("2025-07-01")


def _panel(hours, uwi="A", start=D):
    n = len(hours)
    return pd.DataFrame({"uwi": uwi, "production_date": pd.date_range(start, periods=n),
                         "hours_on_prod": np.asarray(hours, dtype=float)})


def _fail(day, uwi="A", ftype="Rods"):
    return pd.DataFrame({"uwi": [uwi], "failure_start_date": [D + pd.Timedelta(days=day)],
                         "failure_end_date": [pd.NaT], "failure_group": [ftype], "cause_group": ["Fatigue"]})


def _jobs(rows):
    return pd.DataFrame([{"uwi": "A", "start_date": D + pd.Timedelta(days=s),
                          "end_date": D + pd.Timedelta(days=e), "job_duration_days": e - s,
                          "job_category": c, "job_type": "Rod Repair", "job_subtype": None,
                          "job_group": "rod_truck", "field_est_total": 1000.0} for s, e, c in rows])


def test_event_pulls_onset_back_and_ends_at_repair_job():
    hours = [24] * 5 + [0, 0] + [0] * 3 + [24] * 10     # down from day 5, recorded day 7
    ev = merge.build_events(_fail(7), _jobs([(8, 9, "Rod Truck")]), _panel(hours))
    assert ev["event_start"].iloc[0] == D + pd.Timedelta(days=5)
    assert ev["event_end"].iloc[0] == D + pd.Timedelta(days=9)


def test_event_end_falls_back_to_recovery():
    hours = [24] * 7 + [0, 0, 0] + [24] * 10
    ev = merge.build_events(_fail(7), _jobs([(60, 61, "Rod Truck")]), _panel(hours))
    assert ev["event_end"].iloc[0] == D + pd.Timedelta(days=9)  # day before hours recover


def test_labels_and_history_use_only_the_past():
    panel = _panel([24] * 60)
    f = pd.concat([_fail(10), _fail(40, ftype="Pump")], ignore_index=True)
    ev = merge.build_events(f, _jobs([(10, 11, "Rod Truck")]), panel)
    out = merge.add_failure_features(panel.copy(), ev).set_index("production_date")
    day = lambda k: D + pd.Timedelta(days=k)  # noqa: E731
    assert out.loc[day(10), "failure_onset"] == 1 and out.loc[day(10), "failure_active"] == 1
    assert out.loc[day(10), "fail_count_prior"] == 0      # today's failure is not history
    assert out.loc[day(11), "fail_count_prior"] == 1
    assert out.loc[day(11), "days_since_last_failure"] == 1
    assert out.loc[day(9), "fail_next_7d"] == 1 and out.loc[day(2), "fail_next_7d"] == 0
    assert out.loc[day(39), "last_failure_type"] == "Rods"
    assert np.isnan(out.loc[day(55), "fail_next_30d"])    # horizon runs past the data: unknown


def test_labels_unobserved_before_coverage():
    panel = _panel([24] * 40, start=merge.LABEL_START - pd.Timedelta(days=20))
    out = merge.add_failure_features(panel.copy(), merge.build_events(
        _fail(-100).iloc[:0], _jobs([(0, 0, "Rod Truck")]), panel))
    assert out.loc[out.production_date < merge.LABEL_START, "fail_next_30d"].isna().all()


def test_job_history_counts_strictly_before():
    panel = _panel([24] * 30)
    out = merge.add_job_features(panel.copy(), merge.clean_jobs(_jobs([(5, 6, "Rod Truck"), (20, 20, "Workover")])))
    out = out.set_index("production_date")
    assert out.loc[D + pd.Timedelta(days=5), "job_count_prior"] == 0
    assert out.loc[D + pd.Timedelta(days=6), "job_count_prior"] == 1
    assert out.loc[D + pd.Timedelta(days=25), "days_since_last_job"] == 5
    assert out.loc[D + pd.Timedelta(days=25), "last_job_category"] == "Workover"
    assert out.loc[D + pd.Timedelta(days=6), "job_active"] == 1
