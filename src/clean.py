"""Clean daily_production.parquet into a dashboard table and an NLinear model table.

Usage:
    python src/clean.py --input data/raw --output data/processed

Outputs (in --output):
    dashboard.parquet  scoped + cleaned rows, gaps kept, no imputation, unscaled
    model.parquet      regular daily panel per well, short gaps imputed, `split`
                       column, raw features plus scaled copies (`<col>_s`)
    wells.parquet      one row per well: identifiers/location for joining only
    scaler.json        per-feature median/IQR (fit on the train split only)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

log = logging.getLogger("clean")

SEED = 42
PANEL_START = pd.Timestamp("2021-01-01")
MAX_GAP_DAYS = 3
HOLDOUT_MONTHS = 4
HOLDOUT_WELL_FRAC = 0.15
NULL_DROP_FRAC = 0.90
# Named in the cleaning spec; hydraulic_pressure_psi is only ~87% null inside the PCP
# scope, so the threshold alone would keep it.
ALWAYS_DROP = ["hydraulic_pressure_psi", "choke_open_pct", "injected_water_m3"]

# Columns that are identifiers/descriptors: kept only in wells.parquet for joining.
ID_COLS = [
    "formatted_uwi", "well_name", "field_operating_area", "province",
    "surface_latitude", "surface_longitude", "facility_name",
    "pcp_pump_name", "pcp_pump_make", "pcp_pump_model",
]
TEXT_DROP = ["pcp_pump_comment"]
DECIMAL_COLS = ["surface_latitude", "surface_longitude", "pcp_pump_rpm"]

VOLUME_COLS = ["oil_produced_m3", "water_produced_m3", "gas_produced_e3m3"]
ZERO_FILL_COLS = VOLUME_COLS + ["lost_oil_m3"]
SHUT_STATUSES = ["Suspended", "Abandoned"]
DROP_STATUSES = ["Drilling", "Issued", "SOLD", "Injecting"]
LABEL_MAP = {"Oil": "Heavy Oil", "H2O Inj.": "Water Injection"}

NEG_TO_NAN = [
    "hours_down", "hours_on_prod", "lost_oil_m3",
    "tubing_pressure_kpa", "casing_pressure_kpa", "injection_pressure_kpa",
    "oil_produced_m3", "gas_produced_e3m3", "water_produced_m3",
    "injection_volume_m3", "injected_water_m3", "injected_polymer_m3",
    "hydraulic_pressure_psi", "pump_efficiency_pct",
]
# 999999 is treated as a sentinel everywhere; 999 only where it is a known placeholder
# (a real 999 m3 of water or 999 kPa is plausible, so it is not replaced globally).
SENTINEL_999_COLS = [
    "pcp_pump_rod_torque_ftlb", "injected_polymer_m3", "pump_efficiency_pct", "choke_open_pct",
]
# Domain limits picked from clear breaks in the distributions of the scoped data
# (e.g. RPM has nothing between 800 and 1500; casing pressure jumps from 10 MPa to 50 MPa).
PHYSICAL_MAX = {
    "pcp_pump_rpm": 1000,
    "pcp_pump_electric_current_a": 500,
    "pcp_pump_rod_torque_ftlb": 10000,
    "pcp_pump_volume_throughput_m3d": 2000,
    "tubing_pressure_kpa": 20000,
    "casing_pressure_kpa": 20000,
}
THROUGHPUT_RTOL = 0.05

LOG1P_COLS = VOLUME_COLS + ["lost_oil_m3", "pcp_pump_volume_throughput_m3d"]
# Kept unscaled as label material; not model inputs.
NON_FEATURES = {"hours_down"}
CORE_COLS = ["hours_on_prod"] + VOLUME_COLS


# --------------------------------------------------------------------------- load
def _resolve_input(path: Path) -> Path:
    return path / "daily_production.parquet" if path.is_dir() else path


def pcp_wells(path: Path) -> set[str]:
    t = pq.read_table(path, columns=["uwi", "pcp_pump_name"])
    t = t.filter(pc.is_valid(t["pcp_pump_name"]))
    return set(pc.unique(t["uwi"]).to_pylist())


def load(path: Path, wells: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream the file in batches (the full table does not fit in memory as pandas).

    Returns (daily rows for PCP wells with a `null_count` column, well-level id table).
    """
    pf = pq.ParquetFile(path)
    keep = [c for c in pf.schema_arrow.names if c not in ID_COLS + TEXT_DROP]
    well_set = pa.array(sorted(wells))
    parts, id_parts = [], []
    for batch in pf.iter_batches(batch_size=500_000):
        t = pa.Table.from_batches([batch])
        t = t.filter(pc.is_in(t["uwi"], value_set=well_set))
        if t.num_rows == 0:
            continue
        nulls = np.zeros(t.num_rows, dtype=np.int16)
        for col in t.columns:
            nulls += col.is_null().to_numpy(zero_copy_only=False)
        ids = t.select(["uwi"] + ID_COLS).to_pandas()
        id_parts.append(ids.groupby("uwi", sort=False).first())
        t = t.select(keep).append_column("null_count", pa.array(nulls))
        for c in DECIMAL_COLS:  # decimal128 -> float
            if c in t.column_names:
                t = t.set_column(t.column_names.index(c), c, pc.cast(t[c], pa.float64()))
        parts.append(t.to_pandas(strings_to_categorical=True))
    df = pd.concat(parts, ignore_index=True)
    for c in df.select_dtypes("category").columns:
        df[c] = df[c].astype("string")
    ids = pd.concat(id_parts).groupby(level=0).first().reset_index()
    for c in ["surface_latitude", "surface_longitude"]:
        ids[c] = pd.to_numeric(ids[c], errors="coerce").astype(float)
    return df, ids


# --------------------------------------------------------------------------- steps
def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df = (df.sort_values(["uwi", "production_date", "null_count"], kind="stable")
            .drop_duplicates(["uwi", "production_date"], keep="first")
            .drop(columns="null_count"))
    log.info("dedupe: removed %d duplicate rows", n - len(df))
    return df.reset_index(drop=True)


def parse_types(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["production_date", "on_production_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in DECIMAL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def standardise_status(df: pd.DataFrame) -> pd.DataFrame:
    df["primary_fluid_type"] = df["primary_fluid_type"].replace(LABEL_MAP)

    log.info("well_status vs well_daily_status (disagreements):\n%s",
             df.loc[df["well_status"].ne(df["well_daily_status"])
                    & df["well_status"].notna() & df["well_daily_status"].notna()]
               .groupby(["well_status", "well_daily_status"]).size().to_string())

    # flow_direction is null on exactly the null-status rows: fill it within the well
    # so the scope filter does not silently drop those days.
    df["flow_direction"] = df.groupby("uwi")["flow_direction"].transform(lambda s: s.ffill().bfill())

    s = df["well_daily_status"].copy()
    n0 = s.isna().sum()
    s = s.fillna(df["well_status"])
    on = df["hours_on_prod"] > 0
    produced = (df[VOLUME_COLS] > 0).any(axis=1)
    injected = (df[["injection_volume_m3", "injected_water_m3"]] > 0).any(axis=1)
    s = s.mask(s.isna() & on & produced, "Producing")
    s = s.mask(s.isna() & on & injected, "Injecting")
    df["well_daily_status"] = s
    df = df.drop(columns="well_status")
    log.info("status: %d nulls -> %d filled, %d rows dropped", n0, n0 - s.isna().sum(), s.isna().sum())
    return df.dropna(subset=["well_daily_status"])


def scope(df: pd.DataFrame) -> pd.DataFrame:
    """Producing-direction heavy-oil PCP wells only (PCP wells are pre-filtered in load)."""
    n = len(df)
    keep = (
        df["flow_direction"].eq("production")
        & (df["primary_fluid_type"].eq("Heavy Oil") | df["primary_fluid_type"].isna())
        & ~df["well_daily_status"].isin(DROP_STATUSES)
    )
    df = df[keep.fillna(False)]
    log.info("scope: kept %d of %d rows, %d wells", len(df), n, df["uwi"].nunique())
    return df


def replace_impossible(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num] == 999999)
    for c in SENTINEL_999_COLS:
        if c in df.columns:
            df.loc[df[c] == 999, c] = np.nan
    for c in NEG_TO_NAN:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan
    if "choke_open_pct" in df.columns:  # 0-100 scale in this data -> fraction
        ch = df["choke_open_pct"]
        df["choke_open_pct"] = (ch / 100).where((ch >= 0) & (ch <= 100))
    for c, hi in PHYSICAL_MAX.items():
        bad = df[c] > hi
        log.info("physical limit %s > %s: %d -> NaN", c, hi, bad.sum())
        df.loc[bad, c] = np.nan
    return df


def fix_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """pump_efficiency_pct is mostly a fraction (median ~0.69).

    (1, 1.2]  : kept (plausible slight over-reads, 76k rows)
    (1.2, 5]  : NaN  (too high for a fraction, too low to be a percentage)
    (5, 120]  : percentage -> /100
    then anything outside [0, 1.2] -> NaN
    """
    e = df["pump_efficiency_pct"]
    e = e.where(~((e > 5) & (e <= 120)), e / 100)
    df["pump_efficiency_pct"] = e.where((e >= 0) & (e <= 1.2))
    return df


def reconcile_pump(df: pd.DataFrame) -> pd.DataFrame:
    """Throughput is reported as displacement * RPM / 100 (exact on 99.5% of rows).

    Rows that disagree by > 5% are flagged and throughput is recomputed from the
    components; missing throughput or RPM is back-filled from the other two.
    """
    disp, rpm = df["pcp_pump_displacement_m3d100rpm"], df["pcp_pump_rpm"]
    thr = df["pcp_pump_volume_throughput_m3d"]
    expected = disp * rpm / 100
    mismatch = thr.notna() & expected.notna() & ~np.isclose(thr, expected, rtol=THROUGHPUT_RTOL)
    df["pump_reading_flag"] = mismatch
    thr = thr.where(~mismatch & thr.notna(), expected)
    rpm = rpm.fillna((thr * 100 / disp).where(disp > 0))
    df["pcp_pump_volume_throughput_m3d"] = thr.where(thr <= PHYSICAL_MAX["pcp_pump_volume_throughput_m3d"])
    df["pcp_pump_rpm"] = rpm.where(rpm <= PHYSICAL_MAX["pcp_pump_rpm"])
    log.info("pump: %d mismatched readings flagged and recomputed", mismatch.sum())
    return df


def zero_fill_shut(df: pd.DataFrame) -> pd.DataFrame:
    shut = df["well_daily_status"].isin(SHUT_STATUSES)
    for c in ZERO_FILL_COLS:
        df.loc[shut & df[c].isna(), c] = 0.0
    return df


def drop_low_info(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Drop columns > 90% null, measured on pre-cutoff rows to avoid peeking at the test period."""
    frac = df.loc[df["production_date"] < cutoff].isna().mean()
    drop = sorted((set(frac[frac > NULL_DROP_FRAC].index) | set(ALWAYS_DROP)) & set(df.columns))
    log.info("drop >%.0f%% null: %s", NULL_DROP_FRAC * 100, drop)
    return df.drop(columns=drop)


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex each well to a continuous daily range; added rows get is_imputed=True.

    Range: max(on_production_date, 2021-01-01) (or the first observed day if earlier)
    to the well's own last observed day.
    """
    g = df.groupby("uwi")
    first, last = g["production_date"].min(), g["production_date"].max()
    on_prod = g["on_production_date"].first().fillna(PANEL_START).clip(lower=PANEL_START)
    start = pd.concat([on_prod, first], axis=1).min(axis=1)
    days = (last - start).dt.days.to_numpy() + 1
    full = pd.DataFrame({
        "uwi": np.repeat(last.index.to_numpy(), days),
        "production_date": np.repeat(start.to_numpy(), days)
        + pd.to_timedelta(np.concatenate([np.arange(n) for n in days]), unit="D"),
    })
    out = full.merge(df.assign(is_imputed=False), on=["uwi", "production_date"], how="left")
    out["is_imputed"] = out["is_imputed"].isna()
    log.info("panel: %d rows (%d added)", len(out), out["is_imputed"].sum())
    return out.sort_values(["uwi", "production_date"]).reset_index(drop=True)


def impute_short_gaps(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Linearly interpolate NaN runs of <= MAX_GAP_DAYS inside a well.

    A run is only filled when the observed rows on both sides are active
    (Producing with hours_on_prod > 0) and no row inside the run is a known shut-in,
    so values are never carried across a shut-in.
    """
    status = df["well_daily_status"]
    active = (status.eq("Producing").fillna(False) & (df["hours_on_prod"] > 0)).astype(bool)
    shut = (status.isin(SHUT_STATUSES) | df["hours_on_prod"].eq(0)).astype(bool)
    new_well = df["uwi"].ne(df["uwi"].shift())
    uwi = df["uwi"].to_numpy()
    n = len(df)
    pos = np.arange(n)
    filled_any = np.zeros(n, dtype=bool)

    for c in cols:
        na = df[c].isna()
        if not na.any():
            continue
        run = (na.ne(na.shift()) | new_well).cumsum()
        runs = pd.DataFrame({"run": run[na], "pos": pos[na], "shut": shut[na]}).groupby("run").agg(
            first=("pos", "min"), last=("pos", "max"), size=("pos", "size"), shut=("shut", "any"))
        runs = runs[(runs["size"] <= MAX_GAP_DAYS) & ~runs["shut"]]
        prev, nxt = runs["first"].to_numpy() - 1, runs["last"].to_numpy() + 1
        inside = (prev >= 0) & (nxt < n)
        prev, nxt, runs = prev[inside], nxt[inside], runs[inside]
        ok = (uwi[prev] == uwi[runs["first"]]) & (uwi[nxt] == uwi[runs["first"]]) \
            & active.to_numpy()[prev] & active.to_numpy()[nxt]
        runs = runs[ok]
        if runs.empty:
            continue
        mask = np.zeros(n, dtype=bool)
        for f, l in zip(runs["first"], runs["last"]):
            mask[f:l + 1] = True
        interp = df[c].interpolate(limit_area="inside")  # neighbours are in the same well (checked)
        df.loc[mask, c] = interp[mask]
        filled_any |= mask

    # rows filled inside an active stretch are producing days
    df.loc[filled_any & df["well_daily_status"].isna(), "well_daily_status"] = "Producing"
    if "hours_down" in df.columns:
        df["hours_down"] = df["hours_down"].fillna(24 - df["hours_on_prod"])
    df["long_gap"] = df[CORE_COLS].isna().any(axis=1)
    log.info("impute: %d rows filled, %d rows still in long gaps (exclude windows containing them)",
             filled_any.sum(), df["long_gap"].sum())
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """train / test_time (last HOLDOUT_MONTHS, train wells) / test_well (held-out wells, all dates)."""
    cutoff = df["production_date"].max() - pd.DateOffset(months=HOLDOUT_MONTHS)
    wells = np.sort(df["uwi"].unique())
    rng = np.random.default_rng(SEED)
    held = set(rng.choice(wells, size=int(round(len(wells) * HOLDOUT_WELL_FRAC)), replace=False))
    df["split"] = np.where(df["uwi"].isin(held), "test_well",
                           np.where(df["production_date"] >= cutoff, "test_time", "train"))
    log.info("split: cutoff %s, %d held-out wells\n%s", cutoff.date(), len(held),
             df["split"].value_counts().to_string())
    return df, cutoff


def fit_scaler(df: pd.DataFrame, features: list[str]) -> dict:
    train = df.loc[df["split"].eq("train") & ~df["is_imputed"]]
    params = {}
    for c in features:
        x = train[c].dropna()
        if c in LOG1P_COLS:
            x = np.log1p(x)
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        params[c] = {"log1p": c in LOG1P_COLS, "median": float(med), "iqr": float(q3 - q1) or 1.0}
    return params


def apply_scaler(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    for c, p in params.items():
        x = np.log1p(df[c]) if p["log1p"] else df[c]
        df[f"{c}_s"] = ((x - p["median"]) / p["iqr"]).astype("float32")
    return df


def validate(df: pd.DataFrame, params: dict) -> None:
    """Fail loudly instead of writing a silently broken model table."""
    assert df.groupby("uwi")["production_date"].diff().dropna().eq(pd.Timedelta("1D")).all(), "panel not daily"
    assert not df.duplicated(["uwi", "production_date"]).any(), "duplicate keys"
    assert set(df.loc[df.split == "train", "uwi"]).isdisjoint(df.loc[df.split == "test_well", "uwi"]), "well leak"
    for c in params:
        raw, s = df[c], df[f"{c}_s"]
        assert raw.isna().equals(s.isna()), f"{c}: scaled NaN pattern differs from raw"
        assert s.std() > 0, f"{c}: scaled column is constant"
        assert (raw.dropna() >= 0).all() or c not in NEG_TO_NAN, f"{c}: negative values remain"


# --------------------------------------------------------------------------- main
def run(input_path: Path, output_dir: Path) -> None:
    src = _resolve_input(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    wells = pcp_wells(src)
    log.info("PCP-equipped wells: %d", len(wells))
    df, ids = load(src, wells)
    ids.to_parquet(output_dir / "wells.parquet", index=False)

    df = dedupe(df)
    df = parse_types(df)
    df = standardise_status(df)
    df = scope(df)
    df = replace_impossible(df)
    df = fix_efficiency(df)
    df = reconcile_pump(df)
    df = zero_fill_shut(df)

    cutoff = df["production_date"].max() - pd.DateOffset(months=HOLDOUT_MONTHS)
    df = drop_low_info(df, cutoff)
    df.to_parquet(output_dir / "dashboard.parquet", index=False)

    numeric = [c for c in df.select_dtypes("number").columns if c not in ("pump_reading_flag",)]
    panel = build_panel(df)
    panel = impute_short_gaps(panel, numeric)
    panel, _ = split(panel)
    features = [c for c in numeric if c not in NON_FEATURES]
    params = fit_scaler(panel, features)
    panel = apply_scaler(panel, params)
    panel[numeric] = panel[numeric].astype("float32")
    validate(panel, params)

    panel.to_parquet(output_dir / "model.parquet", index=False)
    (output_dir / "scaler.json").write_text(json.dumps(params, indent=2))
    log.info("wrote %s", output_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data/raw"),
                    help="directory containing daily_production.parquet, or the file itself")
    ap.add_argument("--output", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.input, args.output)


if __name__ == "__main__":
    main()
