"""Build the slim data bundle the deployed dashboard reads (dashboard/bundle/).

Only what dashboard/app.py displays: the forecast, the daily risk history, per-well drivers,
evaluation tables, well locations/pump info, failure events, job dates/types and the last 720 days
of a few operating columns for the scored wells. No free-text fields (job summaries/objectives,
failure comments) are included.

Usage (workspace root):  python src/make_dashboard_bundle.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, DATA, BUNDLE = ROOT / "outputs", ROOT / "data" / "processed", ROOT / "dashboard" / "bundle"
MODEL = "v1"
OPS_DAYS = 720       # the dashboard's "Days of history" slider goes up to 720


def main() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "evaluation").mkdir(parents=True)

    fc = pd.read_parquet(OUT / "forecast_fail_next_30d.parquet")
    wells = set(fc.uwi)
    fc.to_parquet(BUNDLE / "forecast_fail_next_30d.parquet", index=False)

    pd.read_parquet(OUT / f"predictions_fail_next_30d_{MODEL}.parquet",
                    columns=["uwi", "production_date", "fail_next_30d_prob_cal", "fail_next_30d",
                             "incomplete_reporting_day"]
                    ).query("uwi in @wells").astype({"fail_next_30d_prob_cal": "float32"}).to_parquet(
        BUNDLE / f"predictions_fail_next_30d_{MODEL}.parquet", index=False)
    shutil.copy(OUT / f"drivers_channels_{MODEL}.parquet", BUNDLE)
    for f in (OUT / "evaluation").glob("*.csv"):
        d = pd.read_csv(f)
        (d[d.model == MODEL] if "model" in d else d).to_csv(BUNDLE / "evaluation" / f.name, index=False)

    pd.read_parquet(DATA / "wells.parquet", columns=["uwi", "surface_latitude", "surface_longitude", "pcp_pump_make",
                                                      "pcp_pump_model"]).query("uwi in @wells").to_parquet(
        BUNDLE / "wells.parquet", index=False)
    pd.read_parquet(DATA / "failure_events.parquet",
                    columns=["uwi", "event_start", "event_end", "failure_group", "cause_group", "repair_job_category"]
                    ).query("uwi in @wells").to_parquet(BUNDLE / "failure_events.parquet", index=False)
    pd.read_parquet(DATA / "jobs_clean.parquet",
                    columns=["uwi", "start_date", "end_date", "job_category", "job_type", "job_subtype",
                             "job_duration_days"]).query("uwi in @wells").to_parquet(
        BUNDLE / "jobs_clean.parquet", index=False)

    since = (fc.production_date.max() - pd.Timedelta(days=OPS_DAYS + 5)).to_pydatetime()
    cols = ["uwi", "production_date", "oil_produced_m3", "water_produced_m3", "hours_on_prod", "hours_down",
            "pcp_pump_rpm", "pcp_pump_rod_torque_ftlb", "pump_efficiency_pct", "downtime_code_1"]
    ops = pd.read_parquet(DATA / "dashboard.parquet", columns=cols, filters=[("production_date", ">=", since)])
    ops = ops[ops.uwi.isin(wells)].sort_values(["uwi", "production_date"])
    num = cols[2:-1]
    ops[num] = ops[num].astype("float32")
    ops.to_parquet(BUNDLE / "dashboard.parquet", index=False, row_group_size=50_000)

    total = 0
    for f in sorted(BUNDLE.rglob("*.*")):
        size = f.stat().st_size; total += size
        print(f"{size / 1e6:7.2f} MB  {f.relative_to(ROOT)}")
    print(f"{total / 1e6:7.2f} MB  total")


if __name__ == "__main__":
    main()
