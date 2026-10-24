"""Preliminary baseline model (placeholder for Workstream A).

Linear regression of target wind speed on the mean wind speed of each reference
park (H, O), their values +-30 min around each timestamp, and the O-park wind
direction. Validated on rolling 3-day fake outages. Returns an AssessmentResult
so the report/GUI can be built against real numbers. Workstream A can replace
this file wholesale as long as `run_assessment` still returns an AssessmentResult.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .schema import AssessmentResult, TurbineResult

MODEL_NAME = "Park-mean linear MCP with lags and direction"
MODEL_VERSION = "0.1.0-baseline"
CHANNELS = ["wind_speed", "wind_direction", "ambient_temperature",
            "air_pressure", "wind_speed_stddev"]
LAGS = [-3, -2, -1, 0, 1, 2, 3]   # 10-min steps


# ---------------------------------------------------------------- loading
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_folder(data_dir: str | Path):
    data_dir = Path(data_dir)
    turbines, hashes = {}, {}
    for f in sorted(data_dir.glob("*_data.xlsx")) + sorted(data_dir.glob("*_data.csv")):
        d = pd.read_excel(f) if f.suffix == ".xlsx" else pd.read_csv(f, parse_dates=["timestamp"])
        d = d.set_index("timestamp").sort_index()
        # sensor-fault rows: every channel reads exactly 0 (incl. 0 degC in summer)
        if "ambient_temperature" in d:
            bad = d["ambient_temperature"] == 0
            d.loc[bad, [c for c in CHANNELS if c in d]] = np.nan
        turbines[f.name.split("_")[0]] = d
        hashes[f.name] = sha256(f)
    curves = {}
    for f in sorted(data_dir.glob("*_power_curve.xlsx")):
        curves[f.name.split("_")[0]] = pd.read_excel(f)[["wind_speed", "power"]]
        hashes[f.name] = sha256(f)
    return turbines, curves, hashes


def power_from_curve(ws, pc: pd.DataFrame):
    return np.interp(ws, pc["wind_speed"], pc["power"], left=0.0, right=float(pc["power"].iloc[-1]))


def detect_outages(series: pd.Series, min_steps: int = 6):
    """Runs of NaN at least `min_steps` long -> list of (start, end_inclusive)."""
    na = series.isna().to_numpy()
    out, i = [], 0
    while i < len(na):
        if na[i]:
            j = i
            while j + 1 < len(na) and na[j + 1]:
                j += 1
            if j - i + 1 >= min_steps:
                out.append((series.index[i], series.index[j]))
            i = j + 1
        else:
            i += 1
    return out


# ---------------------------------------------------------------- features
def build_features(turbines: dict, references: list[str]):
    idx = next(iter(turbines.values())).index
    ws = pd.DataFrame({k: turbines[k]["wind_speed"] for k in references}).reindex(idx)
    ws = ws.interpolate(limit=6).ffill().bfill()
    parks = sorted({r[0] for r in references})
    park_mean = {p: ws[[r for r in references if r[0] == p]].mean(axis=1) for p in parks}
    # direction of the park with the most complete direction data
    dir_park = "O" if "O" in parks else parks[0]
    dirs = pd.DataFrame({k: turbines[k]["wind_direction"] for k in references if k[0] == dir_park})
    theta = np.deg2rad(dirs.mean(axis=1).interpolate(limit=6).ffill().bfill())
    X = pd.DataFrame(index=idx)
    for p, s in park_mean.items():
        for L in LAGS:
            X[f"{p}_lag{L:+d}"] = s.shift(L)
        X[f"{p}_roll70min"] = s.rolling(7, center=True, min_periods=1).mean()
        X[f"{p}_x_sin_dir"] = s * np.sin(theta)
        X[f"{p}_x_cos_dir"] = s * np.cos(theta)
    X["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    X = X.bfill().ffill()
    ref_series = pd.DataFrame({f"park_{p}_mean_ws": s for p, s in park_mean.items()})
    return X, ref_series


def _fit(X, y, mask):
    m = mask & y.notna().to_numpy()
    mod = LinearRegression().fit(X[m], y[m])
    resid = y[m] - mod.predict(X[m])
    return mod, resid


def _expected_power(ws_pred: np.ndarray, resid: pd.Series, pc, n=400):
    """Power curve averaged over the model's historical error distribution."""
    r = np.random.default_rng(0).choice(resid.to_numpy(), n)
    grid = np.clip(np.asarray(ws_pred)[:, None] + r[None, :], 0, None)
    return power_from_curve(grid, pc).mean(axis=1)


def _scores(y, yhat, mask, pc, resid=None):
    """Accuracy on `mask`. With `resid`, power uses the same expected-power
    estimator as the final loss figure, so backtest errors are comparable."""
    m = mask & y.notna().to_numpy()
    e = yhat[m] - y[m]
    pt = power_from_curve(y[m], pc)
    pp = _expected_power(yhat[m].to_numpy(), resid, pc) if resid is not None else power_from_curve(yhat[m], pc)
    return dict(ws_rmse=float(np.sqrt((e ** 2).mean())), ws_mae=float(e.abs().mean()),
                p_rmse=float(np.sqrt(((pp - pt) ** 2).mean())), p_mae=float(np.abs(pp - pt).mean()),
                energy_dev_pct=float(100 * (pp.sum() - pt.sum()) / pt.sum()), n=int(m.sum()))


# ---------------------------------------------------------------- main entry
def run_assessment(data_dir, targets=("T1", "T2", "T3"), references=None,
                   outage=None, validation_window=("2026-09-04", "2026-09-09 23:50")) -> AssessmentResult:
    turbines, curves, hashes = load_folder(data_dir)
    if references is None:
        references = [k for k in turbines if k[0] != targets[0][0]]
    X, ref_series = build_features(turbines, references)
    idx = X.index

    if outage is None:  # longest gap shared by all targets
        common = pd.concat([turbines[t]["wind_speed"] for t in targets], axis=1).isna().all(axis=1)
        gaps = detect_outages(pd.Series(np.where(common, np.nan, 1.0), index=idx))
        outage = max(gaps, key=lambda g: g[1] - g[0])
    o_start, o_end = pd.Timestamp(outage[0]), pd.Timestamp(outage[1])
    in_outage = (idx >= o_start) & (idx <= o_end)
    ctx = (idx >= o_start - pd.Timedelta("1D")) & (idx <= o_end + pd.Timedelta("1D"))
    v0, v1 = (pd.Timestamp(t) for t in validation_window)
    in_val = (idx >= v0) & (idx <= v1)

    results = []
    for t in targets:
        y = turbines[t]["wind_speed"].reindex(idx)
        pc = curves.get(t, next(iter(curves.values())))

        # 1) backtest: rolling 3-day fake outages
        rows, rel_err = [], []
        for b in pd.date_range(idx[0], idx[-1], freq="3D"):
            test = (idx >= b) & (idx < b + pd.Timedelta("3D"))
            if (test & in_outage).any() or y[test].notna().sum() < 300:
                continue
            mod, res = _fit(X, y, ~(test | in_outage))
            s = _scores(y, pd.Series(mod.predict(X), index=idx), test, pc, res)
            rows.append(s)
            rel_err.append(s["energy_dev_pct"] / 100)
        bt = pd.DataFrame(rows)
        rel_err = np.array(rel_err)

        # 2) official validation week (train without outage + validation week)
        mod_v, res_v = _fit(X, y, ~(in_outage | in_val))
        yv = pd.Series(mod_v.predict(X), index=idx)
        val = _scores(y, yv, in_val, pc, res_v)
        validation = pd.DataFrame({"timestamp": idx[in_val], "ws_actual": y[in_val].to_numpy(),
                                   "ws_pred": yv[in_val].to_numpy()})

        # 3) final model on all data except the outage
        mod, resid = _fit(X, y, ~in_outage)
        ws_pred = np.clip(mod.predict(X[in_outage]), 0, None)
        q10, q90 = np.quantile(resid, [0.1, 0.9])
        p_exp = _expected_power(ws_pred, resid, pc)
        pred = pd.DataFrame({
            "timestamp": idx[in_outage],
            "ws_pred": ws_pred,
            "ws_p10": np.clip(ws_pred + q10, 0, None),
            "ws_p90": ws_pred + q90,
            "power_kw": p_exp,
            "power_p10_kw": power_from_curve(np.clip(ws_pred + q10, 0, None), pc),
            "power_p90_kw": power_from_curve(ws_pred + q90, pc),
        })
        e50 = float(p_exp.sum() / 6)  # kW over 10 min -> kWh
        # energy range from backtest relative energy errors: actual = pred / (1 + err)
        e10 = float(e50 / (1 + np.quantile(rel_err, 0.9)))
        e90 = float(e50 / (1 + np.quantile(rel_err, 0.1)))
        m = ~in_outage & y.notna().to_numpy()
        r2 = float(mod.score(X[m], y[m]))

        # explainable summary: total weight on each park across lags
        coefs = pd.Series(mod.coef_, index=X.columns)
        park_weights = {p: float(coefs[[c for c in X.columns if c.startswith(p + "_lag") or c == f"{p}_roll70min"]].sum())
                        for p in sorted({r[0] for r in references})}
        results.append(TurbineResult(
            turbine=t, predictions=pred,
            energy_kwh={"p10": e10, "p50": e50, "p90": e90},
            metrics={
                "val_ws_rmse": val["ws_rmse"], "val_ws_mae": val["ws_mae"],
                "val_p_rmse": val["p_rmse"], "val_p_mae": val["p_mae"],
                "val_energy_dev_pct": val["energy_dev_pct"],
                "bt_ws_rmse": float(bt.ws_rmse.mean()), "bt_ws_mae": float(bt.ws_mae.mean()),
                "bt_p_rmse": float(bt.p_rmse.mean()), "bt_p_mae": float(bt.p_mae.mean()),
                "bt_abs_energy_err_pct": float(bt.energy_dev_pct.abs().mean()),
                "n_backtest_windows": int(len(bt)), "r2": r2,
            },
            method={"park_weights": park_weights, "intercept": float(mod.intercept_),
                    "n_features": int(X.shape[1]), "n_train_rows": int(m.sum())},
            validation=validation,
            context=pd.DataFrame({"timestamp": idx[ctx], "ws_actual": y[ctx].to_numpy()}),
        ))

    ref_series = ref_series.reset_index().rename(columns={"index": "timestamp"})
    parks = sorted({r[0] for r in references})
    summary = (
        f"Wind speeds at the affected turbines were reconstructed from {len(references)} reference "
        f"turbines in park(s) {', '.join(parks)} outside the affected wind farm. A linear regression, "
        f"fitted on all available history outside the outage, relates each target turbine's wind speed "
        f"to the park-average reference wind speed (including values up to 30 minutes before and after "
        f"each interval) and the reference wind direction. Lost energy is the turbine power curve applied "
        f"to the reconstructed wind, averaged over the model's historical error distribution."
    )
    return AssessmentResult(
        turbines=results, outage_start=o_start, outage_end=o_end,
        power_curve=curves[targets[0]], references=list(references), input_files=hashes,
        model_name=MODEL_NAME, model_version=MODEL_VERSION, method_summary=summary,
        reference_series=ref_series,
    )
