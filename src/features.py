"""Feature pipeline shared by training (src/train.py), inference (src/predict.py) and evaluation.

Two feature sets:
  v1  exactly the pipeline of src/model.ipynb (the original checkpoint). Do not change it:
      src/predict.py --verify checks it reproduces the notebook's saved predictions.
  v3  the v1 pipeline on merged_v3.parquet: failure history and labels reconstructed from repair jobs
      back to 2021 (src/merge.py --proxy-before), so the model can train on 2021-2026.
  v2  v1 plus: sentinel guard, partly-reported days treated as missing (not shutdowns),
      extra leak-free rolling/delta operating features, recency/count history, and lagged
      downtime-code counts. Every value at day t uses only data from days <= t.

All text read here comes from typed Parquet string columns (never from raw/binary dumps) and
goes through sanitize_text() before use.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

P = Path("data/processed")

TARGET = "fail_next_30d"
LABEL_START = pd.Timestamp("2025-06-01")
MAX_LOOKBACK = 60
DATA_START = {"v1": LABEL_START - pd.Timedelta(days=MAX_LOOKBACK + 35),
              "v2": LABEL_START - pd.Timedelta(days=MAX_LOOKBACK + 100),   # room for 60-day rolling features
              "v3": pd.Timestamp("2021-01-01")}                            # full history (merged_v3, proxy labels)
MERGED = {"v1": "merged.parquet", "v2": "merged.parquet", "v3": "merged_v3.parquet"}
LEAKY = ["fail_next_30d", "fail_next_7d", "days_to_next_failure", "failure_active", "failure_onset",
         "active_failure_type", "active_failure_cause", "job_active", "last_job_duration_days",
         "job_cost_prior", "label_observed", "split"]
HIST = ["fail_count_prior", "fail_count_90d", "fail_count_365d", "fail_count_prior_rods", "fail_count_prior_pump",
        "fail_count_prior_tubing", "days_since_last_failure", "job_count_prior", "job_count_90d", "job_count_365d",
        "job_count_prior_rod_truck", "job_count_prior_workover", "job_count_prior_pump_change",
        "days_since_last_job", "days_since_last_rod_truck", "days_since_last_workover", "days_since_last_pump_change"]
STATIC = ["well_age_years", "surface_latitude", "surface_longitude"]
DAILY = ["hours_on_prod", "hours_down", "oil_produced_m3", "water_produced_m3", "gas_produced_e3m3",
         "tubing_pressure_kpa", "casing_pressure_kpa", "pump_efficiency_pct"]
SETPOINT = ["pcp_pump_rpm", "pcp_pump_rod_torque_ftlb", "pcp_pump_electric_current_a",
            "pcp_pump_volume_throughput_m3d", "pcp_pump_displacement_m3d100rpm"]
CORE = ["hours_on_prod", "oil_produced_m3", "water_produced_m3"]
CATS = ["field_operating_area", "pcp_pump_make"]

V2_EXTRA = ["oil_14v60", "water_14v60", "hours_mean_7d", "hours_down_7d", "downtime_starts_30d",
            "days_since_downtime", "water_cut_7v30", "eff_7v30", "rpm_changes_30d", "torque_changes_30d",
            "fail_count_30d", "job_count_30d", "rod_jobs_per_year",
            "dt_rods_30d", "dt_pump_30d", "dt_tubing_30d", "dt_torque_30d", "dt_external_30d", "dt_shutin_30d"]
# downtime-code groups (matched on sanitised, lower-cased text)
DT_GROUPS = {"rods": r"\brod", "pump": r"pump", "tubing": r"tubing", "torque": r"torq",
             "external": r"power|grid|facility|esd|turn around|weather|road|battery", "shutin": r"shut"}
DT_LAG = 3   # downtime codes count over days t-30 .. t-3: same-day/just-before codes can be the failure itself


# ------------------------------------------------------------------------------------ text
_MOJIBAKE = re.compile("[ÃÂâ][\x80-\xbf€™œžŸ]|�")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _clean_one(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (bytes, bytearray)):
        x = bytes(x).decode("utf-8", errors="replace")
    x = str(x)
    if _MOJIBAKE.search(x):                       # UTF-8 bytes that were decoded as cp1252/latin-1
        for enc in ("cp1252", "latin-1"):
            try:
                y = x.encode(enc).decode("utf-8")
                if len(_MOJIBAKE.findall(y)) < len(_MOJIBAKE.findall(x)):
                    x = y
                    break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    x = unicodedata.normalize("NFKC", x).replace("�", "")
    x = _CONTROL.sub(" ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x or np.nan


def sanitize_text(s: pd.Series) -> pd.Series:
    """Decode bytes, repair common mojibake, NFKC-normalise, drop control chars, collapse spaces.

    Works on unique values only, so it is cheap on millions of rows.
    """
    u = pd.Series(s.dropna().unique())
    return s.map(dict(zip(u, u.map(_clean_one))))


# ------------------------------------------------------------------------------------ model
class NLinearClassifier(nn.Module):
    """NLinear backbone + linear head (identical to src/model.ipynb)."""
    def __init__(self, L, C, d=8, n_cats=(), emb_dim=4, individual=False, dropout=0.1):
        super().__init__()
        self.individual = individual
        self.temporal = nn.ModuleList([nn.Linear(L, d) for _ in range(C)]) if individual else nn.Linear(L, d)
        self.embs = nn.ModuleList([nn.Embedding(n + 1, emb_dim) for n in n_cats])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(C * d + emb_dim * len(n_cats), 1))

    def forward(self, x, cats):
        last = x[:, -1:, :]
        z = (x - last).transpose(1, 2)
        if self.individual:
            z = torch.stack([lin(z[:, i]) for i, lin in enumerate(self.temporal)], dim=1)
        else:
            z = self.temporal(z)
        z = z + last.transpose(1, 2)
        h = torch.cat([z.flatten(1)] + [e(cats[:, i]) for i, e in enumerate(self.embs)], dim=1)
        return self.head(h).squeeze(-1)


def load_model(ck: dict, n_channels: int) -> NLinearClassifier:
    model = NLinearClassifier(ck["lookback"], n_channels, ck["d"], n_cats=[len(ck["vocab"][c]) for c in ck["cats"]],
                              individual=ck["individual"], dropout=ck.get("dropout", 0.1))
    model.load_state_dict(ck["state_dict"])
    return model.eval()


def score(model, X, cats, rows, L, bs=8192, logits=False) -> np.ndarray:
    X_t, C_t = torch.from_numpy(X), torch.from_numpy(cats)
    off = torch.arange(-L + 1, 1)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = torch.as_tensor(rows[i:i + bs])
            z = model(X_t[r[:, None] + off], C_t[r])
            out.append((z if logits else torch.sigmoid(z)).numpy())
    return np.concatenate(out) if out else np.array([], dtype="float32")


# ------------------------------------------------------------------------------------ frame
def roll(s, w, fn, minp, pos, shift=0):
    """Rolling stat down the sorted column; rows whose window reaches into the previous well -> NaN."""
    r = getattr(pd.Series(s).shift(shift).rolling(w, min_periods=minp), fn)().to_numpy(dtype="float32", copy=True)
    r[pos < w - 1 + shift] = np.nan
    return r


def partly_reported_days(m: pd.DataFrame) -> pd.Index:
    """Days where far more producing wells than usual report 0 oil and 0 water (late/missing reporting)."""
    prod = m.well_daily_status.eq("Producing").to_numpy(dtype=bool, na_value=False)
    zero = (m.oil_produced_m3.eq(0) & m.water_produced_m3.eq(0))[prod].groupby(m.production_date[prod]).mean()
    return zero[(zero > 2 * zero.median()) & (zero > 0.4)].index


def build_frame(feature_set: str = "v1") -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Returns (frame sorted by uwi/date, position-in-well, channel list for this feature set)."""
    v2 = feature_set == "v2"
    since = [("production_date", ">=", DATA_START[feature_set].to_pydatetime())]
    m = pd.read_parquet(P / MERGED[feature_set], filters=since,
                        columns=["uwi", "production_date", "split", "well_daily_status", TARGET, "failure_active"]
                        + HIST + STATIC + CATS)
    ops_cols = DAILY + SETPOINT + (["downtime_code_1"] if v2 else [])
    ops = pd.read_parquet(P / "dashboard.parquet", filters=since, columns=["uwi", "production_date"] + ops_cols)
    m = m.merge(ops, on=["uwi", "production_date"], how="left"); del ops
    m = m.sort_values(["uwi", "production_date"]).reset_index(drop=True)
    num = DAILY + SETPOINT + HIST + STATIC
    m[num] = m[num].astype("float32")

    if v2:
        # guard: placeholder values must never reach the model (src/clean.py removes them; this is a backstop)
        ops_num = DAILY + SETPOINT
        m[ops_num] = m[ops_num].mask((m[ops_num] >= 999_998) | (m[ops_num] == 999))
        # partly-reported days: their zeros are missing data, not shutdowns -> NaN (forward fill bridges them)
        bad_days = partly_reported_days(m)
        zero_rows = m.production_date.isin(bad_days) & m.oil_produced_m3.eq(0) & m.water_produced_m3.eq(0)
        m.loc[zero_rows, ["hours_on_prod", "hours_down", "oil_produced_m3", "water_produced_m3", "gas_produced_e3m3"]] = np.nan
        m["incomplete_reporting_day"] = m.production_date.isin(bad_days)
        m.attrs["partly_reported_days"] = [str(d.date()) for d in bad_days]
        raw_rpm, raw_torque = m.pcp_pump_rpm.copy(), m.pcp_pump_rod_torque_ftlb.copy()

    # ---- v1 pipeline (unchanged from src/model.ipynb) ----
    g = m.groupby("uwi", sort=False)
    pos = g.cumcount().to_numpy()
    for c, n in [("pcp_pump_electric_current_a", "amps"), ("pcp_pump_rod_torque_ftlb", "torque"), ("pump_efficiency_pct", "eff")]:
        m[f"has_{n}"] = m[c].notna().astype("float32")
    m[DAILY] = g[DAILY].ffill(limit=3)
    m[SETPOINT] = g[SETPOINT].ffill(limit=60)

    ENG = []
    for n in ["amps", "torque", "eff"]:
        m[f"has_{n}_30d"] = roll(m[f"has_{n}"], 30, "max", 1, pos); ENG.append(f"has_{n}_30d")
    m = m.drop(columns=["has_amps", "has_torque", "has_eff"])
    base = {"torque": "pcp_pump_rod_torque_ftlb", "amps": "pcp_pump_electric_current_a", "rpm": "pcp_pump_rpm",
            "eff": "pump_efficiency_pct", "oil": "oil_produced_m3", "water": "water_produced_m3",
            "hours": "hours_on_prod", "throughput": "pcp_pump_volume_throughput_m3d"}
    bounded = {"eff", "hours"}
    with np.errstate(divide="ignore", invalid="ignore"):
        for n, c in base.items():
            s = m[c]
            m[f"{n}_std_7d"] = roll(s, 7, "std", 4, pos); ENG.append(f"{n}_std_7d")
            m30 = roll(s, 30, "mean", 10, pos)
            for k in ([3, 7] if n in ("torque", "amps", "rpm") else [3]):
                mk = roll(s, k, "mean", max(1, k - 1), pos)
                m[f"{n}_{k}v30"] = mk - m30 if n in bounded else np.where(m30 > 0, mk / m30 - 1, np.nan).astype("float32")
                ENG.append(f"{n}_{k}v30")
        w7, o7 = roll(m.water_produced_m3, 7, "sum", 4, pos), roll(m.oil_produced_m3, 7, "sum", 4, pos)
        m["water_cut_7d"] = np.where(w7 + o7 > 0, w7 / (w7 + o7), np.nan).astype("float32")
    m["downtime_days_30d"] = roll((m.hours_on_prod == 0).astype("float32"), 30, "sum", 20, pos)
    ENG += ["water_cut_7d", "downtime_days_30d"]
    for c in [c for c in HIST if c.startswith("days_since")]:
        m[c] = m[c].fillna(730).clip(upper=730)
    channels = DAILY + SETPOINT + ENG + HIST + STATIC

    if v2:
        _add_v2(m, pos, raw_rpm, raw_torque)
        channels = channels + V2_EXTRA
    assert not set(LEAKY) & set(channels), "a leaky column is a model input"
    return m, pos, channels


def _add_v2(m: pd.DataFrame, pos: np.ndarray, raw_rpm: pd.Series, raw_torque: pd.Series) -> None:
    g = m.groupby("uwi", sort=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        for n, c in [("oil", "oil_produced_m3"), ("water", "water_produced_m3")]:
            m14, m60 = roll(m[c], 14, "mean", 7, pos), roll(m[c], 60, "mean", 30, pos)
            m[f"{n}_14v60"] = np.where(m60 > 0, m14 / m60 - 1, np.nan).astype("float32")
        h = m.hours_on_prod
        m["hours_mean_7d"] = roll(h, 7, "mean", 4, pos)
        m["hours_down_7d"] = roll(m.hours_down, 7, "sum", 4, pos)
        down = (h == 0).astype("float32").to_numpy()
        prev = np.r_[0, down[:-1]]; prev[pos == 0] = 0
        m["downtime_starts_30d"] = roll((down == 1) & (prev == 0), 30, "sum", 20, pos)
        idx = pd.Series(np.where(down == 1, np.arange(len(m)), np.nan))
        last_down = idx.groupby(m.uwi.to_numpy(), sort=False).ffill().to_numpy()
        m["days_since_downtime"] = np.clip(np.nan_to_num(np.arange(len(m)) - last_down, nan=365), 0, 365).astype("float32")

        w7, o7 = roll(m.water_produced_m3, 7, "sum", 4, pos), roll(m.oil_produced_m3, 7, "sum", 4, pos)
        w30, o30 = roll(m.water_produced_m3, 30, "sum", 10, pos), roll(m.oil_produced_m3, 30, "sum", 10, pos)
        m["water_cut_7v30"] = (np.where(w7 + o7 > 0, w7 / (w7 + o7), np.nan)
                               - np.where(w30 + o30 > 0, w30 / (w30 + o30), np.nan)).astype("float32")
        m["eff_7v30"] = roll(m.pump_efficiency_pct, 7, "mean", 4, pos) - roll(m.pump_efficiency_pct, 30, "mean", 10, pos)

    # setpoint changes = operator adjustments (count of new readings that differ from the previous reading)
    for n, raw in [("rpm", raw_rpm), ("torque", raw_torque)]:
        last = raw.groupby(m.uwi.to_numpy(), sort=False).ffill().groupby(m.uwi.to_numpy(), sort=False).shift(1)
        changed = (raw.notna() & last.notna() & (raw != last)).astype("float32")
        m[f"{n}_changes_30d"] = roll(changed, 30, "sum", 10, pos)

    # recency/counts: *_prior counts are already strictly before t, so differences stay leak-free
    for n, c in [("fail_count_30d", "fail_count_prior"), ("job_count_30d", "job_count_prior")]:
        v = m[c].to_numpy(dtype="float32")
        lag = g[c].shift(30).to_numpy(dtype="float32")
        m[n] = np.where(pos >= 30, v - lag, np.nan).astype("float32")
    years = np.minimum(m.well_age_years.to_numpy(),
                       (m.production_date - pd.Timestamp("2021-01-01")).dt.days.to_numpy() / 365.25)
    m["rod_jobs_per_year"] = (m.job_count_prior_rod_truck / np.clip(years, 0.25, None)).astype("float32")

    # downtime-code groups over t-30 .. t-3 (text sanitised first; lag keeps the failure's own codes out)
    code = sanitize_text(m.pop("downtime_code_1")).str.lower()
    for n, pat in DT_GROUPS.items():
        hit = code.str.contains(pat, regex=True).fillna(False).astype("float32")
        m[f"dt_{n}_30d"] = roll(hit, 28, "sum", 14, pos, shift=DT_LAG)


def window_ok(m: pd.DataFrame, pos: np.ndarray, L: int) -> np.ndarray:
    """True where the last L days have hours/oil/water present and stay inside one well."""
    core_gap = m[CORE].isna().any(axis=1).astype("float32")
    return roll(core_gap, L, "sum", L, pos) == 0


def fit_scaler(m: pd.DataFrame, channels: list[str], fit_rows: np.ndarray) -> dict:
    log1p = {"oil_produced_m3", "water_produced_m3", "gas_produced_e3m3", "tubing_pressure_kpa", "casing_pressure_kpa",
             "pcp_pump_volume_throughput_m3d", "pcp_pump_rod_torque_ftlb", "pcp_pump_electric_current_a"}
    scaler = {}
    for c in channels:
        v = m[c].to_numpy("float32")
        lg = c in log1p or c.startswith(("fail_count", "job_count"))
        if lg:
            v = np.log1p(np.clip(v, 0, None))
        ref = v[fit_rows]; ref = ref[~np.isnan(ref)]
        med, iqr = np.percentile(ref, 50), np.subtract(*np.percentile(ref, [75, 25]))
        sc = iqr if iqr > 0 else (ref.std() if ref.std() > 0 else 1.0)
        scaler[c] = {"log1p": lg, "median": float(med), "scale": float(sc)}
    return scaler


def to_matrix(m: pd.DataFrame, channels: list[str], scaler: dict, cats: list[str], vocab: dict):
    missing = set(channels) - set(m.columns)
    assert not missing, f"pipeline is missing channels: {missing}"
    X = np.empty((len(m), len(channels)), dtype="float32")
    for j, c in enumerate(channels):
        s = scaler[c]
        v = m[c].to_numpy("float32")
        if s["log1p"]:
            v = np.log1p(np.clip(v, 0, None))
        X[:, j] = np.nan_to_num(np.clip((v - s["median"]) / s["scale"], -10, 10), nan=0.0)
    if cats:
        codes = np.stack([m[c].map(vocab[c]).fillna(0).astype("int64").to_numpy() for c in cats], axis=1)
    else:
        codes = np.zeros((len(m), 0), dtype="int64")
    return X, codes
