# CLAUDE.md

## Project
Predict production cessation (failures/downtime) of heavy oil PCP wells
using an NLinear model, plus a dashboard. DSMLC F26 Datathon.

## Data (never commit)
- data/raw/: daily_production.parquet (5M rows, 202 MB), well_jobs.parquet,
  production_failures.parquet. All join on `uwi`.
- The daily file is large: read with pyarrow, select columns, process by
  well or in chunks. Don't load everything casually.
- Never `git add` anything in data/ or any .parquet/.csv file.
- Only exception: dashboard/bundle/ (slim dashboard data for Streamlit Cloud, ~14 MB, no free text; built by python src/make_dashboard_bundle.py). The repo is public, so this data is public.

## Cleaning rules (src/clean.py)
- Dedupe on (uwi, production_date).
- Sentinels (999999, 999) and negative volumes/pressures -> NaN.
- pump_efficiency_pct > 1: divide by 100.
- Merge labels: Oil -> Heavy Oil, H2O Inj. -> Water Injection.
- Fill volumes with 0 only when status is Suspended/Abandoned; keep NaN when Producing.
- Dashboard table: keep gaps, no imputation. Model table: regular daily panel,
  impute only gaps <= 3 days.

## Conventions
- Code in src/, exploration in notebooks/, charts saved to reports/figures/.
- Python 3.11+, pandas, pyarrow, torch. Pin new deps in requirements.txt.
- Set random seed = 42. Split by time and by well. No leakage from downtime days into features.
- Clear notebook outputs before committing.

## Commands
- Clean: python src/clean.py --input data/raw --output data/processed
- Merge failures + jobs: python src/merge.py --raw data/raw (labels: failure_active, fail_next_7d/30d; failure records only exist from 2025-06)
- EDA notebook: python notebooks/build_eda.py, then execute notebooks/eda_failure_drivers.ipynb
- Train: run src/model.ipynb with the `Python (datathon-2026)` kernel (conda env datathon-2026 has torch; base Anaconda's OpenMP clashes with torch). NLinear classifier for fail_next_30d; time split with 30-day purge gaps + held-out wells.
- Experiments (validation-only selection): python src/train.py -> models/nlinear_fail30_v2.pt, outputs/experiments/. Features live in src/features.py (v1 = notebook pipeline, must stay bit-identical: python src/predict.py --verify).
- Predict: python src/predict.py --model <ckpt> [--start 2026-04-01] [--threshold 0.5|f1|cost] [--forecast]
- Cost evaluation: python src/evaluate.py [--cost-fn --cost-fp --oil-value --budget] -> outputs/evaluation/. Test (>= 2026-06-27) is scored only; thresholds, calibration, tiers and policy settings are fitted on validation.
- Run torch scripts/tests with the datathon-2026 env python (base Anaconda has no torch).
- Dashboard: python -m streamlit run dashboard/app.py (base Anaconda env; deployed app uses dashboard/requirements.txt and reads dashboard/bundle/ when outputs/ is absent)
- Tests: pytest tests/

## Git
- Work on a feature/* branch, never directly on main. Small, specific commit messages.