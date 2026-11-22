import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clean  # noqa: E402


def _panel(oil, status=None, hours=None, uwi="W1"):
    n = len(oil)
    return pd.DataFrame({
        "uwi": [uwi] * n,
        "production_date": pd.date_range("2024-01-01", periods=n),
        "well_daily_status": pd.array(status or ["Producing"] * n, dtype="string"),
        "hours_on_prod": hours if hours is not None else [24.0] * n,
        "oil_produced_m3": oil,
        "water_produced_m3": [1.0] * n,
        "gas_produced_e3m3": [1.0] * n,
        "is_imputed": [False] * n,
    })


def test_dedupe_keeps_row_with_fewest_nulls():
    df = pd.DataFrame({
        "uwi": ["A", "A"], "production_date": ["2024-01-01"] * 2,
        "oil_produced_m3": [np.nan, 5.0], "null_count": [3, 1],
    })
    out = clean.dedupe(df)
    assert len(out) == 1 and out["oil_produced_m3"].iloc[0] == 5.0


def test_efficiency_scale():
    df = pd.DataFrame({"pump_efficiency_pct": [0.7, 1.1, 1.5, 69.0, -2.0, 150.0]})
    out = clean.fix_efficiency(df)["pump_efficiency_pct"]
    assert out.tolist()[:2] == [0.7, 1.1]          # fractions (incl. slight over-read) kept
    assert np.isnan(out[2])                         # ambiguous range -> NaN
    assert out[3] == pytest.approx(0.69)            # percentage -> fraction
    assert np.isnan(out[4]) and np.isnan(out[5])    # negative / >120% -> NaN


def test_reconcile_pump_recomputes_mismatch_and_fills_missing():
    df = pd.DataFrame({
        "pcp_pump_displacement_m3d100rpm": [50.0, 50.0, 50.0],
        "pcp_pump_rpm": [200.0, 200.0, np.nan],
        "pcp_pump_volume_throughput_m3d": [100.0, 500.0, 100.0],
    })
    out = clean.reconcile_pump(df)
    assert out["pump_reading_flag"].tolist() == [False, True, False]
    assert out["pcp_pump_volume_throughput_m3d"].tolist() == [100.0, 100.0, 100.0]
    assert out["pcp_pump_rpm"].iloc[2] == 200.0


def test_short_gap_filled_long_gap_not():
    oil = [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan, 10.0]
    out = clean.impute_short_gaps(_panel(oil), ["oil_produced_m3"])
    assert out["oil_produced_m3"].iloc[1:3].tolist() == [2.0, 3.0]
    assert out["oil_produced_m3"].iloc[5:9].isna().all()
    assert out["long_gap"].iloc[5:9].all() and not out["long_gap"].iloc[:5].any()


def test_no_imputation_across_shut_in():
    oil = [1.0, np.nan, 3.0, np.nan, 5.0]
    status = ["Producing", "Producing", "Suspended", "Producing", "Producing"]
    out = clean.impute_short_gaps(_panel(oil, status=status), ["oil_produced_m3"])
    assert out["oil_produced_m3"].isna().iloc[[1, 3]].all()


def test_no_imputation_across_wells():
    df = pd.concat([_panel([1.0, np.nan], uwi="A"), _panel([np.nan, 2.0], uwi="B")], ignore_index=True)
    out = clean.impute_short_gaps(df, ["oil_produced_m3"])
    assert out["oil_produced_m3"].isna().sum() == 2


def test_panel_is_continuous_and_flags_added_rows():
    df = pd.DataFrame({
        "uwi": ["A", "A"],
        "production_date": pd.to_datetime(["2024-01-01", "2024-01-04"]),
        "on_production_date": pd.to_datetime(["2023-12-30"] * 2),
        "oil_produced_m3": [1.0, 2.0],
    })
    out = clean.build_panel(df)
    assert len(out) == 6  # 2023-12-30 .. 2024-01-04
    assert out["is_imputed"].tolist() == [True, True, False, True, True, False]


def test_split_has_no_overlap_and_scaler_uses_train_only():
    rows = []
    for w in range(20):
        rows.append(_panel([float(w)] * 200, uwi=f"W{w:02d}"))
    df, cutoff = clean.split(pd.concat(rows, ignore_index=True))
    train, held = df[df.split == "train"], df[df.split == "test_well"]
    assert set(train.uwi).isdisjoint(held.uwi)
    assert train.production_date.max() < cutoff
    params = clean.fit_scaler(df, ["oil_produced_m3"])
    assert params["oil_produced_m3"]["median"] == pytest.approx(np.log1p(train.oil_produced_m3).median())
