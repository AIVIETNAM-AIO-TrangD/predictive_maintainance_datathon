## Findings (summary of the analysis below)

**Setup:**
- 513,757 producing well-days from 1,168 wells, June 2025 to October 2026.
- Overall `fail_next_30d` rate: **12.5%**.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **Historical failures** | **Strong** | Failures in the previous 365 days: 0 → 7%, 4+ → 31%. Last failure ≤30 days ago → 27%; none or more than a year ago → 7%. Failures cluster: 34% of wells had none, while the top 10% of wells had 40% of all failures. |
| **Historical jobs (minor repairs)** | **Strong, but overlaps with failure history** | Last rod-truck job ≤30 days ago → 30%; never → 5%. Rod-truck jobs are mostly the repair after a rod failure, so job history and failure history measure the same thing. The model does as well with either one alone. Job history goes back to 2021, so it covers the period before failure records start. |
| **Mechanical load and RPM** | **Supported, as how hard the well is run** | Risk rises steadily with RPM (lowest decile 5% → highest 23%). It also rises with motor current up to about 70 A and rod torque up to about 850 ft-lb. But RPM, torque, amps and throughput only change on 2–9% of days. They are setpoints or occasional field entries, so they describe the well's operating level, not a warning that builds before failure. |
| **Water cut** | **Strongest single model feature** | Lowest-water-cut decile (about 40%) → 7%; above 99% water cut → 21%. One possible reason is that more fluid means more rod cycles and more corrosion, but check this with someone who knows the field. |
| **Falling pump efficiency / throughput** | **Weak** | Efficiency drifts about 2% below the well's own baseline over the 30 days before failure. Oil rate is about 5% down by day −7, then collapses at onset. The drop is real but small compared with day-to-day noise (AUC about 0.52). Throughput change carries no signal because throughput is a setpoint. |
| **Long downtime** | **Depends on how long** | Risk rises with hours down in the last 30 days up to about 100 h (23%). Wells that were down most of the month have low risk, because they aren't running. Recent unplanned downtime is a warning; long shut-ins are not. |
| **Well and location** | **Large differences, but no extra information** | Winter field has 4.3 failures per well-year against 1.9 overall. Plover Lake Conventional and Bodo Main are below 0.25. Pump makes LSI, Weatherford and PCM run about 2.1–2.3, against 0.8 for Kudu and PCP (small samples). Younger wells fail more. Once operating features are in the model, location adds nothing (AUC 0.760 with location, 0.761 without), so field differences seem to come from how the wells are run. |

**All features together:** gradient boosting scores **AUC 0.76 on held-out wells**. Removing the operating features costs the most (0.76 → 0.71).

**Caveats:**
- Labels cover only 17 months, and failure history is only recorded from mid-2025.
- Failure onset was moved back to the day the well actually stopped (see `src/merge.py`).
- Nearby days from the same well are strongly correlated, so the single-feature AUCs are a ranking, not significance tests.
- Rod failures are 78% of events, and the drivers of pump or tubing failures may differ.
