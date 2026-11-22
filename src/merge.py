"""Join failures and well jobs onto the daily panel.

Usage:
    python src/merge.py --processed data/processed --raw raw_dataset

Reads model.parquet + wells.parquet (from src/clean.py) and the raw
production_failures / well_jobs parquet files, writes merged.parquet.

Labels (one row per well-day):
    failure_active       1 while a failure event is ongoing (event_start..event_end)
    failure_onset        1 on the first day of an event
    active_failure_type  Rods / Pump / Tubing / Other on active days
    fail_next_7d/30d     1 if an event starts in (t, t+k]; NaN where unobserved
    label_observed       False before LABEL_START (failures were not recorded then)

History features use only information strictly before day t.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("merge")

# Failures are only recorded consistently from June 2025 (≈170/month after,
# 3 records in total before May 2025, 27 in a partial May 2025).
LABEL_START = pd.Timestamp("2025-06-01")
JOB_MATCH_BEFORE, JOB_MATCH_AFTER = 3, 7      # repair job counted as the response to a failure
DOWNTIME_LOOKBACK = 7                         # extend onset back over the zero-hours run before it
RECOVERY_HOURS, RECOVERY_MAX_DAYS = 12, 14    # fallback end: first day back above 12 h on
MAX_EVENT_DAYS = 30
HORIZONS = (7, 30)

FAILURE_GROUPS = ["Rods", "Pump", "Tubing", "Other"]
JOB_GROUPS = ["rod_truck", "workover", "pump_change", "other"]
PUMP_CHANGE_TYPES = {"Pump Repair/Replacement", "PCP Replacement", "Pump Repair", "PCP Upgrade"}


# --------------------------------------------------------------------------- cleaning
def failure_group(s: pd.Series) -> pd.Series:
    t = s.fillna("").str.lower()
    out = np.select(
        [t.str.contains("rod"), t.str.contains("pump|pcp|rotor|stator|elastomer|insert"),
         t.str.contains("tub|tbg")],
        ["Rods", "Pump", "Tubing"], default="Other")
    return pd.Series(out, index=s.index)


def clean_failures(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["failure_start_date"] = pd.to_datetime(f["failure_start_date"], errors="coerce")
    f["failure_end_date"] = pd.to_datetime(f["failure_end_date"], errors="coerce")
    f["failure_group"] = failure_group(f["failure_type"])
    f["cause_group"] = (f["cause"].str.replace("-", " ").str.replace(r"\s+", " ", regex=True)
                        .str.strip().fillna("Unknown"))
    n = len(f)
    f["_nn"] = f.notna().sum(axis=1)
    f = (f.sort_values(["uwi", "failure_start_date", "_nn"], ascending=[True, True, False])
          .drop_duplicates(["uwi", "failure_start_date"]).drop(columns="_nn"))
    early = f["failure_start_date"] < LABEL_START - pd.DateOffset(months=1)
    log.info("failures: %d dup rows removed, %d pre-coverage records dropped (before %s)",
             n - len(f), early.sum(), (LABEL_START - pd.DateOffset(months=1)).date())
    return f[~early].reset_index(drop=True)


def clean_jobs(j: pd.DataFrame) -> pd.DataFrame:
    j = j.copy()
    j["start_date"] = pd.to_datetime(j["start_date"], errors="coerce")
    j["end_date"] = pd.to_datetime(j["end_date"], errors="coerce")
    bad = j["end_date"] < j["start_date"]
    j.loc[bad, "end_date"] = pd.NaT
    j["job_category"] = j["job_category"].replace({
        "Pre Set Surface Hole": "Pre-set Surface", "Completion/workover": "Workover",
        "Initial Completion": "Completion"})
    dur = (j["end_date"] - j["start_date"]).dt.total_seconds() / 86400
    j["job_duration_days"] = dur.fillna(j["job_duration_days"].where(j["job_duration_days"] >= 0))
    pump = j["job_type"].isin(PUMP_CHANGE_TYPES) | j["job_subtype"].fillna("").str.startswith("Pump")
    j["job_group"] = np.select(
        [pump, j["job_category"].eq("Rod Truck"), j["job_category"].eq("Workover")],
        ["pump_change", "rod_truck", "workover"], default="other")
    n = len(j)
    j = j.drop_duplicates(["uwi", "start_date", "job_category", "job_type", "job_subtype"])
    log.info("jobs: %d end<start set to NaT, %d duplicates removed", bad.sum(), n - len(j))
    return j.dropna(subset=["start_date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- events
def build_events(f: pd.DataFrame, j: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """One row per failure with event_start / event_end.

    failure_end_date is empty in the source, so the end is derived from the repair job
    that responds to the failure, or from the day production recovers.
    The start is pulled back to the matched job start or to the zero-hours run that
    leads into the recorded date (wells are typically already down 1-3 days before it).
    """
    ev = f[["uwi", "failure_start_date", "failure_end_date", "failure_group", "cause_group"]].copy()
    ev["fid"] = np.arange(len(ev))

    # repair job matched within [-3, +7] days; take the nearest-starting one
    m = ev.merge(j[["uwi", "start_date", "end_date", "job_duration_days", "job_category"]], on="uwi")
    lag = (m["start_date"] - m["failure_start_date"]).dt.days
    m = m[lag.between(-JOB_MATCH_BEFORE, JOB_MATCH_AFTER)].assign(alag=lag.abs())
    m = m.sort_values(["fid", "alag"]).drop_duplicates("fid").set_index("fid")
    job_end = m["end_date"].fillna(m["start_date"] + pd.to_timedelta(
        np.ceil(m["job_duration_days"].fillna(0)), unit="D"))
    # reindex (not .map): pandas 3 cannot .map an empty datetime Series
    ev["repair_job_category"] = m["job_category"].reindex(ev["fid"]).to_numpy()
    ev["job_start"] = m["start_date"].reindex(ev["fid"]).to_numpy()
    ev["job_end"] = job_end.reindex(ev["fid"]).to_numpy()

    hours = panel.set_index(["uwi", "production_date"])["hours_on_prod"]

    def hours_at(offset_days: int) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([ev["uwi"], ev["failure_start_date"] + pd.Timedelta(days=offset_days)])
        return hours.reindex(idx).to_numpy()

    # onset: walk back over contiguous zero-hours days before the recorded start
    back = np.zeros(len(ev), dtype=int)
    still = np.ones(len(ev), dtype=bool)
    for k in range(1, DOWNTIME_LOOKBACK + 1):
        still &= hours_at(-k) == 0
        back[still] = k
    down_start = ev["failure_start_date"] - pd.to_timedelta(back, unit="D")
    ev["event_start"] = pd.concat([down_start, ev["job_start"]], axis=1).min(axis=1)

    # end: explicit end > repair-job end > first recovery day > the start day itself
    rec = pd.Series(pd.NaT, index=ev.index, dtype="datetime64[ns]")
    for k in range(RECOVERY_MAX_DAYS, 0, -1):  # reverse so the earliest recovery wins
        ok = hours_at(k) >= RECOVERY_HOURS
        rec[ok] = ev["failure_start_date"][ok] + pd.Timedelta(days=k - 1)
    end = ev["failure_end_date"].fillna(ev["job_end"]).fillna(rec).fillna(ev["failure_start_date"])
    end = end.clip(lower=ev["failure_start_date"])
    end = end.clip(upper=ev["event_start"] + pd.Timedelta(days=MAX_EVENT_DAYS))

    # never overlap the same well's next event
    ev = ev.assign(event_end=end).sort_values(["uwi", "event_start"])
    nxt = ev.groupby("uwi")["event_start"].shift(-1) - pd.Timedelta(days=1)
    ev["event_end"] = ev["event_end"].where(nxt.isna() | (ev["event_end"] <= nxt), nxt)
    ev["event_end"] = ev["event_end"].clip(lower=ev["event_start"])
    ev["event_days"] = (ev["event_end"] - ev["event_start"]).dt.days + 1
    src = np.where(ev["job_end"].notna(), "repair_job", np.where(rec.reindex(ev.index).notna(), "recovery", "single_day"))
    log.info("events: %d; onset moved earlier for %d; end source %s; median length %.0f days",
             len(ev), (ev["event_start"] < ev["failure_start_date"]).sum(),
             pd.Series(src).value_counts().to_dict(), ev["event_days"].median())
    return ev.reset_index(drop=True)


# --------------------------------------------------------------------------- features
def _asof_count(panel: pd.DataFrame, dates: pd.DataFrame, shift_days: int = 0,
                strictly_before: bool = True, direction: str = "backward") -> pd.Series:
    """For each panel row: cumulative number of `dates` events before (t - shift_days)."""
    d = dates.sort_values("date").copy()
    d["cum"] = d.groupby("uwi").cumcount() + 1
    left = panel[["uwi", "production_date"]].copy()
    left["key"] = left["production_date"] - pd.Timedelta(days=shift_days)
    left["_row"] = np.arange(len(left))
    out = pd.merge_asof(left.sort_values("key"), d[["uwi", "date", "cum"]], left_on="key",
                        right_on="date", by="uwi", allow_exact_matches=not strictly_before,
                        direction=direction)
    return out.sort_values("_row")["cum"].fillna(0).to_numpy()


def _asof_last(panel: pd.DataFrame, rows: pd.DataFrame, cols: list[str],
               direction: str = "backward") -> pd.DataFrame:
    """Most recent event strictly before t (or next strictly after t when direction='forward')."""
    left = panel[["uwi", "production_date"]].assign(_row=np.arange(len(panel)))
    r = rows[["uwi", "date"] + cols].sort_values("date")
    out = pd.merge_asof(left.sort_values("production_date"), r, left_on="production_date",
                        right_on="date", by="uwi", allow_exact_matches=False, direction=direction)
    return out.sort_values("_row").reset_index(drop=True)


def add_failure_features(panel: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    on = ev.rename(columns={"event_start": "date"})

    # --- labels
    days = ev["event_days"].to_numpy()
    expanded = pd.DataFrame({
        "uwi": np.repeat(ev["uwi"].to_numpy(), days),
        "production_date": np.repeat(ev["event_start"].to_numpy(), days)
        + pd.to_timedelta(np.concatenate([np.arange(n) for n in days] or [np.array([], dtype=int)]), unit="D"),
        "active_failure_type": np.repeat(ev["failure_group"].to_numpy(), days),
        "active_failure_cause": np.repeat(ev["cause_group"].to_numpy(), days),
    }).drop_duplicates(["uwi", "production_date"])
    panel = panel.merge(expanded, on=["uwi", "production_date"], how="left")
    panel["failure_active"] = panel["active_failure_type"].notna().astype("int8")
    onset_keys = set(zip(ev["uwi"], ev["event_start"]))
    panel["failure_onset"] = pd.Series(list(zip(panel["uwi"], panel["production_date"]))).isin(onset_keys).astype("int8").to_numpy()

    nxt = _asof_last(panel, on, ["failure_group"], direction="forward")
    days_to_next = (nxt["date"] - panel["production_date"].reset_index(drop=True)).dt.days.to_numpy()
    panel["days_to_next_failure"] = days_to_next
    last_day = panel.groupby("uwi")["production_date"].transform("max")
    panel["label_observed"] = panel["production_date"] >= LABEL_START
    for h in HORIZONS:
        y = (days_to_next <= h).astype("float32")
        # unknown if the horizon runs past the well's last observed day with no event seen
        unseen = np.isnan(days_to_next) | (days_to_next > h)
        y[unseen & (panel["production_date"] + pd.Timedelta(days=h) > last_day).to_numpy()] = np.nan
        y[~panel["label_observed"].to_numpy()] = np.nan
        panel[f"fail_next_{h}d"] = y

    # --- history (strictly before t)
    panel["fail_count_prior"] = _asof_count(panel, on)
    for g in FAILURE_GROUPS:
        panel[f"fail_count_prior_{g.lower()}"] = _asof_count(panel, on[on["failure_group"] == g])
    for w in (90, 365):
        panel[f"fail_count_{w}d"] = panel["fail_count_prior"] - _asof_count(panel, on, shift_days=w - 1)
    last = _asof_last(panel, on, ["failure_group"])
    panel["days_since_last_failure"] = (panel["production_date"].reset_index(drop=True) - last["date"]).dt.days.to_numpy()
    panel["last_failure_type"] = last["failure_group"].to_numpy()
    return panel


def add_job_features(panel: pd.DataFrame, j: pd.DataFrame) -> pd.DataFrame:
    jobs = j.rename(columns={"start_date": "date"})
    panel["job_count_prior"] = _asof_count(panel, jobs)
    for g in JOB_GROUPS:
        panel[f"job_count_prior_{g}"] = _asof_count(panel, jobs[jobs["job_group"] == g])
    for w in (90, 365):
        panel[f"job_count_{w}d"] = panel["job_count_prior"] - _asof_count(panel, jobs, shift_days=w - 1)

    t = panel["production_date"].reset_index(drop=True)
    last = _asof_last(panel, jobs, ["job_category", "job_type", "job_subtype", "job_duration_days"])
    panel["days_since_last_job"] = (t - last["date"]).dt.days.to_numpy()
    panel["last_job_category"] = last["job_category"].to_numpy()
    panel["last_job_type"] = last["job_type"].to_numpy()
    panel["last_job_subtype"] = last["job_subtype"].to_numpy()
    panel["last_job_duration_days"] = last["job_duration_days"].to_numpy()
    for g in ["rod_truck", "workover", "pump_change"]:
        lg = _asof_last(panel, jobs[jobs["job_group"] == g], [])
        panel[f"days_since_last_{g}"] = (t - lg["date"]).dt.days.to_numpy()

    cost = jobs[["uwi", "date", "field_est_total"]].sort_values("date").copy()
    cost["cum_cost"] = cost.groupby("uwi")["field_est_total"].cumsum()
    cc = _asof_last(panel, cost, ["cum_cost"])
    panel["job_cost_prior"] = cc["cum_cost"].fillna(0).to_numpy()

    # job in progress on day t (descriptive only: it is the response, not a predictor)
    end = jobs["end_date"].fillna(jobs["date"])
    span = (end - jobs["date"]).dt.days.clip(lower=0, upper=60).to_numpy() + 1
    act = pd.DataFrame({
        "uwi": np.repeat(jobs["uwi"].to_numpy(), span),
        "production_date": np.repeat(jobs["date"].to_numpy(), span)
        + pd.to_timedelta(np.concatenate([np.arange(n) for n in span]), unit="D"),
    }).drop_duplicates()
    act["job_active"] = np.int8(1)
    panel = panel.merge(act, on=["uwi", "production_date"], how="left")
    panel["job_active"] = panel["job_active"].fillna(0).astype("int8")
    return panel


def add_static(panel: pd.DataFrame, wells: pd.DataFrame) -> pd.DataFrame:
    cols = ["uwi", "field_operating_area", "province", "surface_latitude", "surface_longitude",
            "pcp_pump_make", "pcp_pump_model"]
    wells = wells.assign(pcp_pump_make=wells["pcp_pump_make"].str.strip().str.upper())  # "Mantl" vs "MANTL"
    panel = panel.merge(wells[cols], on="uwi", how="left")
    panel["well_age_years"] = ((panel["production_date"] - panel["on_production_date"]).dt.days / 365.25).astype("float32")
    return panel


# --------------------------------------------------------------------------- main
def run(processed: Path, raw: Path) -> None:
    panel = pd.read_parquet(processed / "model.parquet")
    panel = panel.drop(columns=[c for c in panel.columns if c.endswith("_s")])  # rescale later incl. new features
    wells = pd.read_parquet(processed / "wells.parquet")
    f = pd.read_parquet(next(raw.glob("production_failures*.parquet")))
    j = pd.read_parquet(next(raw.glob("well_jobs*.parquet")))

    for d in (panel, wells, f, j):
        d["uwi"] = d["uwi"].astype(str)
    scope = set(panel["uwi"])
    log.info("failure wells in scope: %d / %d, job wells in scope: %d / %d",
             f["uwi"].isin(scope).groupby(f["uwi"]).first().sum(), f["uwi"].nunique(),
             j["uwi"].isin(scope).groupby(j["uwi"]).first().sum(), j["uwi"].nunique())
    f = clean_failures(f[f["uwi"].isin(scope)])
    j = clean_jobs(j[j["uwi"].isin(scope)])

    panel = panel.sort_values(["uwi", "production_date"]).reset_index(drop=True)
    ev = build_events(f, j, panel)
    panel = add_failure_features(panel, ev)
    panel = add_job_features(panel, j)
    panel = add_static(panel, wells)

    obs = panel[panel["label_observed"]]
    log.info("merged: %d rows, %d cols; observed-label days %d, active-failure days %d, onsets %d, "
             "fail_next_30d rate %.3f", len(panel), panel.shape[1], len(obs),
             obs["failure_active"].sum(), obs["failure_onset"].sum(), obs["fail_next_30d"].mean())
    assert not panel.duplicated(["uwi", "production_date"]).any(), "join created duplicate rows"
    panel.to_parquet(processed / "merged.parquet", index=False)
    ev.to_parquet(processed / "failure_events.parquet", index=False)
    j.to_parquet(processed / "jobs_clean.parquet", index=False)
    log.info("wrote merged.parquet, failure_events.parquet, jobs_clean.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.processed, args.raw)


if __name__ == "__main__":
    main()
