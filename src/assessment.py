"""Glue between the model (src/model.py) and the claim deliverables (src/report.py).

    from src.assessment import run_assessment
    result = run_assessment("data")          # -> schema.AssessmentResult

Runs estimate_loss() for every target turbine, optionally scores the shown test week,
and packs everything into the AssessmentResult contract that report.py / the GUI read.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import SHOWN_WEEK, split
from .data_io import clean, load_folder
from .energy import PowerCurve, expected_power
from .model import WINDOW, _block_energy_errors, estimate_loss, make_blocks
from .outage import detect_outages
from .schema import AssessmentResult, TurbineResult

MODEL_NAME = "Bounded linear regression on lag-aligned reference turbines"

EXTRA_LABELS = {
    "hour": "time of day",
    "ref_x_hour": "time-of-day dependent slope",
    "direction": "wind direction",
    "ref_x_dir": "wind-direction dependent slope",
    "dir_harmonics": "wind-direction correction curve (e.g. wake sectors)",
    "turbulence": "turbulence",
    "temp_change_3h": "3-hour temperature change",
}


def _model_version() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        h = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        return f"git {h}" if h else "dev"
    except Exception:
        return "dev"


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _abs_block_energy_error(res, pc: PowerCurve) -> tuple[float, int]:
    """Mean |energy error| per cross-validation 3-day block, from the model's CV history."""
    h = res.history
    y = h["ws_actual"]
    cv = h["ws_cv"]
    resid = (y - cv).dropna().to_numpy()
    block_id = make_blocks(y.notna())
    errs = _block_energy_errors(y, expected_power(cv.to_numpy(), resid, pc), block_id, pc)
    return float(np.mean(np.abs(errs)) * 100), int(len(errs))


def _validation(target, references, power_curve, window, **kwargs):
    """Fake an outage over the shown test week, refit, compare with the real values."""
    start, end = window
    masked = target.copy()
    masked.loc[(masked.index >= start) & (masked.index <= end), "wind_speed"] = np.nan
    res = estimate_loss(masked, references, power_curve, window, **kwargs)
    pc = PowerCurve(power_curve)
    p = res.predictions
    truth = clean(target)["wind_speed"].reindex(p["timestamp"]).to_numpy()
    pred, p_pred = p["ws_pred"].to_numpy(), p["power_expected"].to_numpy()
    ok = ~np.isnan(truth) & ~np.isnan(pred)
    p_true = pc.power(truth[ok])
    e_true = p_true.sum() / 6
    scores = {
        "val_ws_rmse": float(np.sqrt(((pred[ok] - truth[ok]) ** 2).mean())),
        "val_ws_mae": float(np.abs(pred[ok] - truth[ok]).mean()),
        "val_p_rmse": float(np.sqrt(((p_pred[ok] - p_true) ** 2).mean())),
        "val_p_mae": float(np.abs(p_pred[ok] - p_true).mean()),
        "val_energy_dev_pct": float((p_pred[ok].sum() / 6 - e_true) / e_true * 100),
    }
    series = pd.DataFrame({"timestamp": p["timestamp"], "ws_actual": truth, "ws_pred": pred})
    return scores, series


def run_assessment(data_dir="data", targets=None, outage=None, target_park="T",
                   include_validation=True, validation_window=SHOWN_WEEK,
                   **model_kwargs) -> AssessmentResult:
    """Run the model on every target turbine of the data folder and return an AssessmentResult.

    outage: (start, end) inclusive; default = longest gap detected per target
    model_kwargs: passed to estimate_loss (references_to_use, select_extras, window)
    """
    turbines, curves = load_folder(str(data_dir))
    tgt_all, refs = split(turbines, target_park)
    targets = list(targets or sorted(tgt_all))
    files = sorted(Path(data_dir).glob("*_data.*")) + sorted(Path(data_dir).glob("*_power_curve.*"))
    hashes = {f.name: _sha256(f) for f in files if not f.name.startswith("~$")}
    return assess({t: tgt_all[t] for t in targets}, refs, curves, outage, hashes,
                  include_validation=include_validation, validation_window=validation_window,
                  **model_kwargs)


def _validation_possible(d, outage, window) -> bool:
    """Shown-week validation only if it lies in the data, does not touch the outage and the
    target has measurements there (uploaded data may cover other dates)."""
    v0, v1 = (pd.Timestamp(x) for x in window)
    if outage is not None and not (pd.Timestamp(outage[1]) < v0 or pd.Timestamp(outage[0]) > v1):
        return False
    ws = d["wind_speed"] if "wind_speed" in d else d.iloc[:, 0]
    sub = ws[(ws.index >= v0) & (ws.index <= v1)]
    return len(sub) > 144 and sub.notna().mean() > 0.5


def assess(targets: dict, refs: dict, curves: dict, outage=None, input_files=None,
           include_validation=True, validation_window=SHOWN_WEEK, **model_kwargs) -> AssessmentResult:
    """Core: estimate_loss() for each {name: frame} in `targets` against `refs` -> AssessmentResult.

    Used by run_assessment (data folder) and by the web claim endpoint (data sent by the page).
    """
    results, used_refs, starts, ends = [], set(), [], []
    for t, d in targets.items():
        pc_df = curves[t]
        o = outage or detect_outages(d, refs)[0]
        res = estimate_loss(d, refs, pc_df, o, **model_kwargs)
        pc = PowerCurve(pc_df)
        m = res.method
        p = res.predictions

        preds = pd.DataFrame({
            "timestamp": p["timestamp"],
            "ws_pred": p["ws_pred"], "ws_p10": p["ws_p10"], "ws_p90": p["ws_p90"],
            "power_kw": p["power_expected"],
            "power_p10_kw": pc.power(p["ws_p10"].to_numpy()),
            "power_p90_kw": pc.power(p["ws_p90"].to_numpy()),
        })
        abs_err, n_blocks = _abs_block_energy_error(res, pc)
        metrics = {
            "bt_ws_rmse": res.metrics["ws_rmse"], "bt_ws_mae": res.metrics["ws_mae"],
            "bt_p_rmse": res.metrics["p_rmse"], "bt_p_mae": res.metrics["p_mae"],
            "bt_energy_dev_pct": res.metrics["energy_dev_pct"],
            "bt_abs_energy_err_pct": abs_err, "n_backtest_windows": n_blocks,
            "r2": res.metrics["r2"],
        }
        validation = None
        if include_validation and _validation_possible(d, o, validation_window):
            try:
                vs, validation = _validation(d, refs, pc_df, validation_window, **model_kwargs)
                metrics.update(vs)
            except Exception:            # validation is a bonus - never block the claim
                validation = None

        h = res.history
        o0, o1 = pd.Timestamp(o[0]), pd.Timestamp(o[1])
        win = (h.index >= o0 - pd.Timedelta("1D")) & (h.index <= o1 + pd.Timedelta("1D"))
        context = pd.DataFrame({"timestamp": h.index[win], "ws_actual": h["ws_actual"][win].to_numpy()})

        coefs = m["coefficients"]
        method = {
            "references": m["references"],
            "weights": {r: float(coefs[f"ws_{r}"]) for r in m["references"]},
            "lags_min": {r: m["lags_min"][r] for r in m["references"]},
            "extra_features": m["extra_features"],
            "extra_coefficients": {k: float(v) for k, v in coefs.items() if not k.startswith("ws_")},
            "intercept": float(m["intercept"]),
            "n_train_rows": m["n_train"],
            "window_min": m["window_min"],
            "description": m["description"],
            "reference_selection": m["reference_selection"],
            "feature_selection": m["feature_selection"],
        }
        results.append(TurbineResult(turbine=t, predictions=preds, energy_kwh=dict(res.energy_kwh),
                                     metrics=metrics, method=method, validation=validation,
                                     context=context))
        used_refs.update(m["references"])
        starts.append(o0)
        ends.append(o1)

    extras = sorted({e for r in results for e in r.method["extra_features"]})
    extras_txt = ", ".join(EXTRA_LABELS.get(e, e) for e in extras)
    summary = (
        f"For each affected turbine, the wind speed during the outage was reconstructed as a weighted "
        f"sum of wind speeds measured at reference turbines in other wind farms "
        f"({', '.join(sorted(used_refs))}). Each reference is first shifted by the time delay that best "
        f"matches the target turbine and smoothed over {WINDOW * 10} minutes; all weights are forced "
        f"to be non-negative. References"
        + (f" and extra terms ({extras_txt})" if extras else "")
        + " were kept only where they improved accuracy in cross-validation over 3-day blocks of "
          "history. Lost energy is the turbine power curve averaged over the model's historical "
          "error distribution."
    )
    first = next(iter(targets))
    return AssessmentResult(
        turbines=results, outage_start=min(starts), outage_end=max(ends),
        power_curve=curves[first], references=sorted(used_refs), input_files=input_files or {},
        model_name=MODEL_NAME, model_version=_model_version(), method_summary=summary,
    )
