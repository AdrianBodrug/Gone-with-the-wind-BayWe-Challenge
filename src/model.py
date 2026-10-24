"""Reference-park regression and estimate_loss() (README tasks A4, A6-A8).

Method in one sentence: the target's wind speed is a weighted sum (weights >= 0) of
lag-aligned, 2-hour-smoothed wind speeds of reference turbines from other parks, plus a few
extra terms kept only if they lower the cross-validated error.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .data_io import align, clean, fill_reference
from .energy import PowerCurve, energy_kwh, expected_power

WINDOW = 13          # smoothing window in 10-min steps (130 min, centred; odd keeps it symmetric)
MAX_LAG = 6          # search reference lags of -60..+60 min
BLOCK_DAYS = 3       # cross-validation block length, same as the hidden test outage
ELIM_TOL = 0.002     # backward elimination: drop a reference if CV RMSE gets no worse than +tol m/s
MIN_GAIN = 0.002     # forward selection: keep an extra feature group if it lowers CV RMSE by >= 0.2 %

EXTRA_GROUPS = {
    "hour":           ["hour_sin", "hour_cos"],
    "ref_x_hour":     ["ref_x_hour_sin", "ref_x_hour_cos"],
    "direction":      ["dir_sin", "dir_cos"],
    "ref_x_dir":      ["ref_x_dir_sin", "ref_x_dir_cos"],
    "turbulence":     ["turbulence"],
    "temp_change_3h": ["temp_change_3h"],
}


@dataclass
class LossResult:
    predictions: pd.DataFrame   # timestamp, ws_pred, ws_p10, ws_p90, power_expected
    energy_kwh: dict            # p10, p50, p90
    metrics: dict               # ws_rmse, ws_mae, p_rmse, p_mae, energy_dev_pct, r2, ...
    method: dict                # references, lags, coefficients, selection history, ...
    history: pd.DataFrame = field(repr=False)   # whole period: actual, CV / fitted prediction


class BoundedLinear:
    """Linear regression where reference weights are >= 0 and all other coefficients are free."""

    def __init__(self, nonneg_cols):
        self.nonneg_cols = set(nonneg_cols)

    def fit(self, X, y, cols):
        self.cols = list(cols)
        A = np.column_stack([np.ones(len(X)), X])
        lb = [-np.inf] + [0 if c in self.nonneg_cols else -np.inf for c in self.cols]
        sol = lsq_linear(A, y, bounds=(lb, [np.inf] * len(lb)))
        self.intercept_, self.coef_ = sol.x[0], sol.x[1:]
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_


# ---------------------------------------------------------------- building blocks

def find_lag(y, x, mask, max_lag=MAX_LAG):
    """Best shift (steps) of reference x for target y; negative = target sees the wind first."""
    out = {}
    for L in range(-max_lag, max_lag + 1):
        pair = pd.concat([y, x.shift(L)], axis=1)[mask].dropna()
        out[L] = pair.corr().iloc[0, 1]
    s = pd.Series(out)
    return int(s.idxmax()), float(s.max())


def make_blocks(train, block_days=BLOCK_DAYS):
    """Consecutive training days cut into blocks of block_days; -1 outside training."""
    idx = train.index
    days = idx[train.to_numpy()].normalize().unique()
    block_id = pd.Series(-1, index=idx)
    day_of = idx.normalize()
    for b, start in enumerate(range(0, len(days), block_days)):
        block_id[day_of.isin(days[start:start + block_days]) & train.to_numpy()] = b
    return block_id


def smooth(x, window):
    if window > 1:
        x = x.rolling(window, center=True, min_periods=window // 2 + 1).mean()
    return x


def reference_features(refs, lags, window):
    """Lag-shifted, smoothed wind speed of every reference: columns ws_<name>."""
    return pd.DataFrame({f"ws_{r}": smooth(d["wind_speed"].shift(lags[r]), window)
                         for r, d in refs.items()})


def extra_features(refs, ref_avg, window):
    """Candidate extra features (see EXTRA_GROUPS), built from all references."""
    idx = ref_avg.index
    F = pd.DataFrame(index=idx)
    hour = idx.hour + idx.minute / 60
    F["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    F["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    F["ref_x_hour_sin"] = ref_avg * F["hour_sin"]          # slope changes with time of day
    F["ref_x_hour_cos"] = ref_avg * F["hour_cos"]
    rad = np.deg2rad(pd.DataFrame({r: d["wind_direction"] for r, d in refs.items()}))
    F["dir_sin"] = np.sin(rad).mean(axis=1)
    F["dir_cos"] = np.cos(rad).mean(axis=1)
    F["ref_x_dir_sin"] = ref_avg * F["dir_sin"]            # slope changes with wind direction
    F["ref_x_dir_cos"] = ref_avg * F["dir_cos"]
    turb = pd.DataFrame({r: d["wind_speed_stddev"] for r, d in refs.items()}).mean(axis=1)
    F["turbulence"] = turb.rolling(window, center=True, min_periods=1).mean()
    temp = pd.DataFrame({r: d["ambient_temperature"] for r, d in refs.items()}).mean(axis=1)
    F["temp_change_3h"] = temp.diff(18)                     # warming/cooling ~ day/night stability
    return F


def cv_predict(F, cols, y, block_id, nonneg):
    """Leave-one-block-out predictions (clipped at 0); NaN where not predicted."""
    X = F[cols].to_numpy(float)
    yv = y.to_numpy(float)
    bid = block_id.to_numpy()
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(yv)
    pred = np.full(len(yv), np.nan)
    for b in np.unique(bid[bid >= 0]):
        test = (bid == b) & ok
        fit = (bid >= 0) & (bid != b) & ok
        if test.sum() == 0:
            continue
        m = BoundedLinear(nonneg).fit(X[fit], yv[fit], cols)
        pred[test] = m.predict(X[test])
    return pd.Series(np.clip(pred, 0, None), index=y.index)


def rmse(y, pred):
    ev = pred.notna() & y.notna()
    return float(np.sqrt(((pred[ev] - y[ev]) ** 2).mean()))


def backward_eliminate(F, candidates, y, block_id, tol=ELIM_TOL):
    """Drop references one at a time while CV RMSE gets no worse than +tol m/s."""
    current = list(candidates)
    score = lambda cols: rmse(y, cv_predict(F, cols, y, block_id, cols))
    best = score(current)
    history = [{"step": "start", "refs": [c[3:] for c in current], "cv_rmse": best}]
    while len(current) > 1:
        trials = {c: score([m for m in current if m != c]) for c in current}
        drop = min(trials, key=trials.get)
        if trials[drop] > best + tol:
            break
        current.remove(drop)
        best = min(best, trials[drop])
        history.append({"step": f"removed {drop[3:]}", "refs": [c[3:] for c in current],
                        "cv_rmse": trials[drop]})
    return current, history


def forward_select(F, ref_cols, groups, y, block_id, min_gain=MIN_GAIN):
    """Add extra feature groups one at a time while CV RMSE improves by >= min_gain (relative)."""
    cols, remaining = list(ref_cols), dict(groups)
    score = lambda c: rmse(y, cv_predict(F, c, y, block_id, ref_cols))
    best = score(cols)
    history = [{"step": "references only", "cv_rmse": best}]
    added = []
    while remaining:
        trials = {g: score(cols + c) for g, c in remaining.items()}
        g = min(trials, key=trials.get)
        if (best - trials[g]) / best < min_gain:
            break
        cols += remaining.pop(g)
        best = trials[g]
        added.append(g)
        history.append({"step": f"added {g}", "cv_rmse": best})
    return cols, added, history


def _metrics(y, ws_pred, p_pred, pc):
    """Scores of a prediction against the measured target, on rows where both exist."""
    ev = ws_pred.notna() & y.notna()
    yt, yp, pp = y[ev].to_numpy(), ws_pred[ev].to_numpy(), np.asarray(p_pred)[ev.to_numpy()]
    pt = pc.power(yt)
    e_true = energy_kwh(pt)
    return {
        "ws_rmse": float(np.sqrt(((yp - yt) ** 2).mean())),
        "ws_mae": float(np.abs(yp - yt).mean()),
        "p_rmse": float(np.sqrt(((pp - pt) ** 2).mean())),
        "p_mae": float(np.abs(pp - pt).mean()),
        "energy_dev_pct": float((energy_kwh(pp) - e_true) / e_true * 100),
        "energy_dev_pct_point": float((energy_kwh(pc.power(yp)) - e_true) / e_true * 100),
        "r2": float(1 - ((yp - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()),
        "n_eval": int(ev.sum()),
    }


def _block_energy_errors(y, p_pred, block_id, pc, min_frac=0.5):
    """Relative energy error per CV block (only blocks with >= min_frac of steps evaluated)."""
    df = pd.DataFrame({"pt": pc.power(y.to_numpy()), "pp": p_pred, "b": block_id.to_numpy()},
                      index=y.index)
    df = df[(df["b"] >= 0) & df["pt"].notna() & df["pp"].notna()]
    steps = BLOCK_DAYS * 144
    errs = []
    for _, g in df.groupby("b"):
        if len(g) >= min_frac * steps and g["pt"].sum() > 0:
            errs.append(g["pp"].sum() / g["pt"].sum() - 1)
    return np.array(errs)


def _py(x):
    """numpy scalars -> plain Python so the frontend can json.dumps the result."""
    return x.item() if isinstance(x, np.generic) else x


# ---------------------------------------------------------------- public API

def estimate_loss(target, references, power_curve, outage, *,
                  references_to_use=None, select_extras=True, window=WINDOW):
    """Reconstruct the target's wind speed during `outage` and estimate the lost energy.

    target:      DataFrame of one turbine (timestamp index or column, wind_speed, ...)
    references:  {name: DataFrame} of turbines outside the target park
    power_curve: DataFrame with columns wind_speed, power (kW)
    outage:      (start, end), both inclusive
    references_to_use: skip automatic reference selection and use these names
    select_extras: run forward selection of extra features (False = references only)
    """
    pc = PowerCurve(power_curve)
    frames = align({"__target__": target, **references})
    idx = frames["__target__"].index
    y = clean(frames.pop("__target__"))["wind_speed"]
    refs = {n: fill_reference(clean(d)) for n, d in frames.items()}

    start, end = pd.Timestamp(outage[0]), pd.Timestamp(outage[1])
    in_outage = pd.Series((idx >= start) & (idx <= end), index=idx)
    y = y.where(~in_outage)                       # never learn from inside the outage
    train = y.notna()
    block_id = make_blocks(train)
    n_blocks = int(block_id.max()) + 1
    if n_blocks < 2:
        raise ValueError("Not enough training data outside the outage for cross-validation.")

    # 1. lag of every reference
    lags, corrs = {}, {}
    for r, d in refs.items():
        lags[r], corrs[r] = find_lag(y, d["wind_speed"], train)
    R = reference_features(refs, lags, window)

    # 2. which references to use
    if references_to_use:
        unknown = set(references_to_use) - set(refs)
        if unknown:
            raise ValueError(f"Unknown references {sorted(unknown)}; available: {list(refs)}")
        ref_cols, elim_hist = [f"ws_{r}" for r in references_to_use], []
    else:
        ref_cols, elim_hist = backward_eliminate(R, list(R.columns), y, block_id)

    # 3. extra features
    F = pd.concat([R, extra_features(refs, R[ref_cols].mean(axis=1), window)], axis=1)
    groups = {g: c for g, c in EXTRA_GROUPS.items()
              if F.loc[train, c].notna().all(axis=1).mean() > 0.5}   # e.g. no stddev in O park
    if select_extras:
        cols, added, fwd_hist = forward_select(F, ref_cols, groups, y, block_id)
    else:
        cols, added, fwd_hist = list(ref_cols), [], []

    # 4. cross-validated predictions -> metrics and uncertainty
    cv = cv_predict(F, cols, y, block_id, ref_cols)
    residuals = (y - cv).dropna().to_numpy()
    cv_power = expected_power(cv.to_numpy(), residuals, pc)
    metrics = _metrics(y, cv, cv_power, pc)
    block_err = _block_energy_errors(y, cv_power, block_id, pc)

    # 5. final fit on all training rows, predict everywhere the features exist
    X = F[cols].to_numpy(float)
    has = ~np.isnan(X).any(axis=1)
    fit = has & train.to_numpy()
    model = BoundedLinear(ref_cols).fit(X[fit], y.to_numpy()[fit], cols)
    ws_fit = np.full(len(idx), np.nan)
    ws_fit[has] = np.clip(model.predict(X[has]), 0, None)
    ws_fit = pd.Series(ws_fit, index=idx)

    # 6. outage outputs
    ws_out = ws_fit[in_outage]
    q10, q90 = np.quantile(residuals, [0.1, 0.9])
    p_exp = expected_power(ws_out.to_numpy(), residuals, pc)
    predictions = pd.DataFrame({
        "timestamp": ws_out.index,
        "ws_pred": ws_out.to_numpy(),
        "ws_p10": np.clip(ws_out.to_numpy() + q10, 0, None),
        "ws_p90": ws_out.to_numpy() + q90,
        "power_expected": p_exp,
    })
    p50 = energy_kwh(p_exp)
    # true ~ estimate / (1 + past relative error): large past over-estimates give the low end
    e_lo, e_hi = np.clip(np.quantile(block_err, [0.1, 0.9]), -0.9, None) if len(block_err) else (0, 0)
    energy = {"p10": p50 / (1 + e_hi), "p50": p50, "p90": p50 / (1 + e_lo)}

    history = pd.DataFrame({
        "ws_actual": y.where(~in_outage),
        "ws_cv": cv,
        "ws_fit": ws_fit,
        "ref_mean": R[ref_cols].mean(axis=1),
        "in_outage": in_outage,
    })
    history.index.name = "timestamp"

    chosen = [c[3:] for c in ref_cols]
    method = {
        "description": ("Weighted sum (weights >= 0) of lag-aligned, "
                        f"{window * 10}-min smoothed wind speeds of {', '.join(chosen)}"
                        + (f", plus {', '.join(added)}" if added else "")
                        + f"; fitted on {int(fit.sum())} 10-min steps outside the outage."),
        "references": chosen,
        "references_available": list(refs),
        "lags_min": {r: lags[r] * 10 for r in refs},
        "correlations": {r: round(corrs[r], 4) for r in refs},
        "extra_features": added,
        "features": cols,
        "coefficients": {c: _py(v) for c, v in zip(cols, model.coef_)},
        "intercept": _py(model.intercept_),
        "window_min": window * 10,
        "n_train": int(fit.sum()),
        "n_cv_blocks": n_blocks,
        "n_outage_steps": int(in_outage.sum()),
        "n_missing_steps": int(ws_out.isna().sum()),
        "residual_p10_p90": [_py(q10), _py(q90)],
        "block_energy_error_p10_p90_pct": [_py(e_lo * 100), _py(e_hi * 100)],
        "reference_selection": elim_hist,
        "feature_selection": fwd_hist,
    }
    return LossResult(predictions, {k: _py(v) for k, v in energy.items()}, metrics, method, history)
