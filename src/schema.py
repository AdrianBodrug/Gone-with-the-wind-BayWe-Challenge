"""Data contract between the model (Workstream A) and the GUI / report (Workstream B).

Any model implementation only has to return an `AssessmentResult`.
The report, email and GUI read nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class TurbineResult:
    turbine: str
    # 10-min rows inside the outage window. Required columns:
    #   timestamp, ws_pred, ws_p10, ws_p90,
    #   power_kw (expected power), power_p10_kw, power_p90_kw
    predictions: pd.DataFrame
    # Lost energy over the outage window, kWh: {"p10": .., "p50": .., "p90": ..}
    energy_kwh: dict
    # Accuracy on validation data, e.g. ws_rmse, ws_mae, p_rmse, p_mae,
    # energy_dev_pct, r2, n_backtest_windows, abs_energy_err_pct_mean
    metrics: dict
    # Human-readable description of what the model used, e.g.
    #   {"references": [...], "features": [...], "coefficients": {...}}
    method: dict = field(default_factory=dict)
    # Optional validation series for plots: timestamp, ws_actual, ws_pred
    validation: Optional[pd.DataFrame] = None
    # Optional measured wind around the outage for plots: timestamp, ws_actual
    context: Optional[pd.DataFrame] = None


@dataclass
class AssessmentResult:
    turbines: list[TurbineResult]
    outage_start: pd.Timestamp
    outage_end: pd.Timestamp          # inclusive last 10-min stamp
    power_curve: pd.DataFrame         # columns: wind_speed, power (kW)
    references: list[str]             # reference turbines / parks used
    input_files: dict                 # filename -> sha256
    model_name: str
    model_version: str
    method_summary: str               # 2-3 plain-language sentences
    timezone: str = "Europe/Berlin"
    reference_series: Optional[pd.DataFrame] = None  # timestamp + ref wind for plots

    # ---- convenience -------------------------------------------------
    @property
    def duration_h(self) -> float:
        return (self.outage_end - self.outage_start).total_seconds() / 3600 + 1 / 6

    def farm_energy_kwh(self) -> dict:
        """Farm totals. P10/P90 are summed per turbine, which assumes the
        turbines' errors move together (they share the same references) -
        the conservative, wider choice."""
        keys = ("p10", "p50", "p90")
        return {k: float(sum(t.energy_kwh[k] for t in self.turbines)) for k in keys}


@dataclass
class ClaimInfo:
    """Facts only the user can supply. Anything left empty is shown as a
    clearly marked placeholder - the tool never invents contract details."""
    wind_farm_name: str = ""
    wind_farm_location: str = ""
    operator_name: str = ""           # claimant / insured
    insurer_name: str = ""
    policy_number: str = ""
    claim_reference: str = ""
    contact_name: str = ""
    contact_email: str = ""
    contact_phone: str = ""
    turbine_model: str = ""
    suspected_cause: str = (
        "A production stop is suspected; possible causes include cable damage "
        "between the wind farm and the grid connection point or a technical "
        "fault preventing feed-in. The root cause is under investigation."
    )
    repair_status: str = ""
    # Valuation. All three models are always calculated and shown side by side;
    # valuation_mode picks the headline one ("market" | "eeg" | "ppa", default "market").
    #   market: day-ahead price per interval (negative -> 0)
    #   eeg:    day-ahead price + max(0, reference value - monthly wind market value), 0 if negative (§51 EEG)
    #   ppa:    fixed PPA price
    # Leave the prices empty to use the documented market benchmarks (src/pricing.py).
    valuation_mode: Optional[str] = None
    eeg_reference_value: Optional[float] = None  # EUR/MWh, the plant's "anzulegender Wert"
    ppa_price: Optional[float] = None            # EUR/MWh
    price_eur_per_mwh: Optional[float] = None    # legacy: price for the headline mode
    price_basis: str = ""                        # free-text note shown with the headline value
    waiting_period_h: Optional[float] = None    # time deductible under the policy
