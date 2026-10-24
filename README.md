# BayWa r.e. Challenge — Wind Farm Outage Energy Loss Estimator

**WATT THE HACK 2026 · Munich Energy & AI Hackathon**

When the communication link to a wind farm drops, the farm may have stopped producing — and its own wind measurements are gone too. For **business interruption insurance claims**, BayWa r.e. needs to estimate how much energy was lost. Today this is done by hand using wind data from a neighbouring farm. **Our job: build a tool that automates it.**

---

## 1. The challenge in one picture

```
 Reference parks (H, O)          Target park (T)               Power curve
 wind speed, direction  ──►  learn relationship  ──►  predicted wind  ──►  lost energy (kWh)
 (available during outage)   (from history)          during outage        + uncertainty
```

1. **Analyse** — learn how wind at the target park relates to wind at the reference parks, using historical data.
2. **Predict** — apply that relationship to reference data during the outage to reconstruct the missing wind speeds.
3. **Calculate** — run predicted wind through the turbine power curve to get the energy yield loss in kWh.

## 2. Scoring

| Weight | Criterion | What it means for us |
|---|---|---|
| **40%** | Model accuracy on hidden test data | Wind speed RMSE/MAE, power RMSE/MAE, deviation of summed energy |
| **30%** | User experience | Clean, intuitive local web GUI |
| **20%** | Solution concept | Clear, defensible methodology |
| **10%** | Presentation | Pitch + demo |

> ⚠️ **Explainability beats accuracy.** The brief says: if there is a trade-off between accuracy and explainability, prioritise explainability.

## 3. Required deliverables

**GUI** — runs locally in a standard web browser and shows:
- [ ] Accuracy metrics (our choice)
- [ ] Predicted wind speed time series
- [ ] Explanation of the prediction method + its accuracy/uncertainty (e.g. correlation analysis)
- [ ] Energy loss calculation in kWh
- [ ] Energy loss visualisation

**Inputs the tool must accept**
- Wind data time series for reference and affected turbines (file upload)
- Power curves of the target turbines
- Outage period — **auto-detected from missing data**, with manual adjustment

**Predictions** for the hidden test period, scored by the judges.

**Bonus (if time permits)**
- [ ] Insurance report — generate an email summarising the outage for the insurer
- [ ] Additional parameters — use temperature, pressure, etc. to improve accuracy
- [ ] Automatic turbine selection — tool picks the best reference turbines from the pool

---

## 4. The data (`data/`)

10 turbines in 3 parks, **90 days (1 Jul – 30 Sep 2026)**, 10-minute averages at hub height. Files are `.xlsx`.

| Park | Turbines | Role |
|---|---|---|
| **T** | T1, T2, T3 | **Targets** — predict these |
| **H** | H1–H5 | Reference |
| **O** | O1, O2 | Reference |

**Columns:** `turbine_name`, `timestamp` (Berlin time), `wind_speed` (m/s), `wind_direction` (°), `ambient_temperature`, `air_pressure`, `wind_speed_stddev`, `test_set` (bool)

**Power curves:** `T1/T2/T3_power_curve.xlsx` — all three are identical. Rated 4,500 (assumed kW) from 11.5 m/s; cut-in ~3 m/s; **no hard cut-out** — output derates gradually above 20 m/s down to 2,529 at 26 m/s.

### Test period (from the organisers' README)

| Period | Dates (inclusive) | Status |
|---|---|---|
| **Hidden test** | **1 – 3 Sep** | T1–T3 wind is NaN → **this is what we predict and get scored on** |
| Shown test | 4 – 9 Sep | Real values available → our official validation week |
| Training | everything else (Jul, Aug, 10–30 Sep) | Free to use |

### Data gotchas

- **All three T turbines are blanked simultaneously** in the hidden window → T2 cannot be used to predict T1. Only H and O parks are usable.
- **The outage is in the middle of the data** → we can train on data from *both* sides of the gap.
- **O park has no `wind_speed_stddev`**, and its `air_pressure` is a fake constant (990.27). Any model using these must handle that.
- **Sensor-fault rows**: some rows have *every* channel = 0 (incl. 0 °C temperature in summer). Treat as missing, not as calm wind.
- **Communication gaps**: multi-hour blocks of NaN in July (e.g. H1 ~16 h on 13–14 Jul; T1 5 h on 7 Aug). Random single NaNs everywhere.
- **NaN ≠ zero.** Never interpret missing values as zero wind or zero power.
- Range is Jul–Sep only → no daylight-saving switch inside the data.

---

## 5. What we know so far (preliminary experiments)

Validated on **27 fake 3-day outages** across the dataset (plus the shown test week):

| Model | Wind RMSE (m/s) | Power RMSE (kW) | Abs. energy-sum error |
|---|---|---|---|
| **Linear: park means + ±30 min lags + O-park direction** | **1.15** | **646** | ~11% |
| Gradient boosting (black box) | 1.16 | 648 | ~10% |
| Linear per 45° direction sector | 1.19 | 667 | ~10% |
| Linear on all 7 reference turbines | 1.21 | 685 | ~12% |

Takeaways:
- **The explainable linear model ties the black box** (GBM wins only 41% of windows). → No accuracy/explainability trade-off; strong concept story.
- Correlation between target and reference parks is only **~0.70** (10-min) / ~0.77 (hourly) → a floor on achievable accuracy. Communicate uncertainty, don't hide it.
- **Variance-ratio MCP correction made things worse** on this data — don't use it.
- **Expected power** (averaging the power curve over the model's error distribution instead of using the point prediction) slightly improves power RMSE and gives natural P10/P50/P90 bands.
- First estimate for the hidden window: **≈ 314 MWh lost** across T1–T3 (T1 ≈ 89, T2 ≈ 126, T3 ≈ 98). The window was windier than average at the reference parks (7.5 vs 6.1 m/s).

---

## 6. Task breakdown

### Workstream A — Data & Model *(owner: ML lead — TBD)*

- [ ] **A1 Loader** — read `.xlsx` and `.csv`; parse timestamps; align all turbines on one 10-min index.
- [ ] **A2 Cleaning** — sensor-fault zeros → NaN; flag frozen/flat-lined values; short-gap interpolation for *reference* inputs only (never for targets).
- [ ] **A3 Outage detection** — find NaN runs in the target series above a minimum length (e.g. ≥ 1 h); return list of `(start, end)`.
- [ ] **A4 Baseline model** — linear regression: park-mean wind speeds (H, O) + lags (−3…+3 steps) + O-park direction (sin/cos interaction) + hour of day.
- [ ] **A5 Validation harness** — blocked backtest: cut fake outages (6 h / 1 d / 3 d) at many positions, report all scored metrics. Also score on the shown test week (4–9 Sep).
- [ ] **A6 Improvements (only keep if backtest improves)** — per-turbine vs. park-level references, direction sectors, temperature/air-density, turbulence (H only), challenger GBM.
- [ ] **A7 Energy calculation** — power-curve interpolation; expected power over residual distribution; energy = Σ P × 10 min; P10/P50/P90.
- [ ] **A8 Reference selection (bonus)** — rank candidate references by backtest error; optionally inverse-error weighted blend.
- [ ] **A9 Final predictions** — fit on all non-hidden data, predict 1–3 Sep for T1–T3, export submission file in the format the judges want.

### Workstream B — GUI & Report *(owner: Adrian)*

- [ ] **B1 App skeleton** (Streamlit + Plotly) running on dummy output from day one.
- [ ] **B2 Upload & setup** — upload turbine + power-curve files; pick target and reference turbines (or "auto").
- [ ] **B3 Outage panel** — auto-detected outage shaded on the timeline; slider/date pickers to adjust.
- [ ] **B4 Method & accuracy panel** — scatter target vs. reference with fit line, R², backtest metrics, plain-language "how this works".
- [ ] **B5 Wind panel** — actual vs. predicted wind with uncertainty band; outage highlighted.
- [ ] **B6 Energy panel** — lost kWh headline (P50 + P10–P90), per-turbine breakdown, cumulative-loss chart.
- [ ] **B7 Insurance email (bonus)** — one-click summary email: farm, period, cause (comms loss), method, lost energy with range, attachments list.
- [ ] **B8 Export** — download predictions + report (CSV/PDF).

### Workstream C — Concept & Pitch *(owner: TBD)*

- [ ] **C1** One-slide method explanation (the pipeline picture above, with numbers).
- [ ] **C2** Why linear: backtest table showing it matches the black box.
- [ ] **C3** Business framing: analyst hours saved per claim; auditable numbers for insurers.
- [ ] **C4** Live demo script (upload → auto-detect → result → email in < 2 min).
- [ ] **C5** Dry run before final presentation.

### Integration contract (A ↔ B)

The GUI only talks to the model through one function, so both sides can work in parallel:

```python
result = estimate_loss(
    target: pd.DataFrame,          # target turbine time series
    references: dict[str, pd.DataFrame],
    power_curve: pd.DataFrame,     # columns: wind_speed, power
    outage: tuple[pd.Timestamp, pd.Timestamp],
)

result.predictions   # DataFrame: timestamp, ws_pred, ws_p10, ws_p90, power_expected
result.energy_kwh    # dict: p10, p50, p90
result.metrics       # dict: ws_rmse, ws_mae, p_rmse, p_mae, energy_dev_pct, r2
result.method        # dict: features used, coefficients, references chosen
```

---

## 7. Suggested repo layout

```
data/            organiser files (xlsx)
src/
  io.py          loading & cleaning           (A1–A2)
  outage.py      outage detection             (A3)
  model.py       model + estimate_loss()      (A4, A6–A8)
  backtest.py    validation harness           (A5)
  energy.py      power curve & energy         (A7)
  report.py      insurance email              (B7)
app.py           Streamlit GUI                (B1–B6, B8)
notebooks/       exploration
requirements.txt
```

## 8. Working agreements

- Branch per task (`a4-baseline`, `b3-outage-panel`, …), small PRs into `main`.
- Never commit large derived files; keep `data/` to the organiser originals.
- Any model change must come with backtest numbers vs. the current baseline.
- Keep the method explainable in one sentence.

## 9. Run locally

```bash
pip install -r requirements.txt
python -m scripts.make_claim_package                                  # demo claim package -> outputs/
python -m scripts.make_claim_package --valuation market --waiting-hours 24   # live day-ahead prices + deductible
python -m scripts.make_claim_package --valuation eeg --price 73.5            # EEG value, 0 in negative-price intervals
streamlit run app.py                                                  # GUI (coming)
```

## 10. Claim deliverables (B7 / B8) — `src/report.py`

One call produces everything the insurer needs. It only reads the `AssessmentResult` contract (`src/schema.py`), so it keeps working when the model is swapped.

```python
from src.assessment import run_assessment   # runs src/model.py estimate_loss() for T1-T3
from src.report import build_email, build_report_pdf, build_claim_package
from src.schema import ClaimInfo

result = run_assessment("data")          # -> AssessmentResult (src/schema.py)
info = ClaimInfo(wind_farm_name="...", insurer_name="...", policy_number="...",
                 price_eur_per_mwh=None, waiting_period_h=None)   # user-supplied facts

email = build_email(result, info)        # {"subject", "body", "full"} -> copy-paste box in the GUI
pkg = build_claim_package(result, info, "outputs")   # .txt + .pdf + .csv + .zip
```

| Output | Contents |
|---|---|
| Email draft (`templates/claim_email.txt.j2`) | What happened, lost MWh (P50 + P10–P90, per turbine), optional EUR value and waiting-period figure, 2-line method + accuracy, metering-data caveat, attachments list |
| PDF report (4 pages) | 1 Summary & outage details · 2 Wind + energy charts · 3 Method, accuracy table, validation scatter, limitations · A Audit trail: model version, regression weights, SHA-256 of every input file |
| CSV | 10-min reconstructed wind (P10/P50/P90), expected power, energy per turbine |

Design rules: missing facts render as visible `[placeholders]` — the tool never invents policy numbers, prices or deductibles; the email is a draft for human review, never sent automatically. See `docs/sample_claim/` for example output.

## 11. Valuation with live day-ahead prices — `src/pricing.py`

The lost energy is valued **interval by interval** at the German day-ahead price (DE-LU), downloaded live for exactly the outage window from the free Energy-Charts API (Fraunhofer ISE; data Bundesnetzagentur | SMARD.de, CC BY 4.0). No API key, works for any date including outages that are still ongoing.

| Mode | Value per lost MWh | Use when |
|---|---|---|
| `market` | day-ahead price of that interval (negative → 0, plant would curtail) | merchant / direct marketing |
| `eeg` | EEG reference value (`--price`), but 0 in negative-price intervals (§51 EEG) | EEG market premium |
| `fixed` | `--price` | PPA / fixed tariff |

- **Not-yet-published hours** (prices for day D appear ~13:00 on D-1): valued provisionally by repeating the last published day's profile, flagged in email + PDF, and updated on re-run.
- **Time zones**: everything is matched in UTC, so the extra hour when summer time ends (e.g. 25 Oct 2026) is counted correctly; hourly prices (before Oct 2025) and quarter-hourly prices (since) both work.
- **Offline**: every download is cached in `data/prices/`; the hidden-window prices are committed so the demo works without internet.
- Output: valuation paragraph in the email, section 4 + price chart in the PDF, `*_valuation.csv` with price and EUR per 10-min interval.
