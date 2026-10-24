"""Reference-park regression and estimate_loss() (README tasks A4, A6-A8).

Method in one sentence: the target's wind speed is a weighted sum (weights >= 0) of
lag-aligned, smoothed wind speeds of reference turbines from other parks, plus a few extra
terms; smoothing, references and extra terms are picked step by step on the test week.

Selection (per target turbine), all scored on the selection period - the organisers' shown
test week 4-9 Sep, with the model fitted without that week:
  0. smoothing window 30 / 70 / 130 min (all references)
  1. references, forward: add the reference that lowers the score most
  2. extra features, forward: same for direction, time of day, turbulence, weather, ...
  each stage stops when the best step improves the score by less than MIN_GAIN.
  score = mean of wind RMSE, wind MAE, power RMSE, power MAE relative to "all references".
If the outage overlaps the test week, or the data do not cover it, the 6 days right after
(else before) the outage are used instead. The final model is refitted on all data outside
the outage. Reported metrics are those of the selection period. Leave-one-3-day-block-out
predictions are only used for the uncertainty (error spread, energy range).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .data_io import align, clean, fill_reference
from .energy import PowerCurve, energy_kwh, expected_power

WINDOW = 13                  # default smoothing in 10-min steps (130 min); the selection may pick another
WINDOWS = (3, 7, 13)         # smoothing options: 30 / 70 / 130 min, centred
MAX_LAG = 6                  # search reference lags of -60..+60 min
BLOCK_DAYS = 3               # block length for the uncertainty estimate, same as the hidden test outage
MIN_GAIN = 0.005             # keep a selection step only if the score improves by >= 0.5 %
SELECTION_DAYS = 6
TEST_WEEK = (pd.Timestamp("2026-09-04 00:00"), pd.Timestamp("2026-09-10 00:00"))   # shown test week, end exclusive
HARMONICS = (2, 3, 4)
WEATHER = ("temperature", "temp_anomaly", "pressure", "pressure_change_3h", "air_density", "turb_intensity")

EXTRA_GROUPS = {
    "hour":           ["hour_sin", "hour_cos"],
    "ref_x_hour":     ["ref_x_hour_sin", "ref_x_hour_cos"],
    "direction":      ["dir_sin", "dir_cos"],
    "ref_x_dir":      ["ref_x_dir_sin", "ref_x_dir_cos"],
    "dir_harmonics":  [f"h_{f}{k}" for k in HARMONICS for f in ("sin", "cos")],
    "turbulence":     ["turbulence"],
    "temp_change_3h": ["temp_change_3h"],
    **{w: [w] for w in WEATHER},
}


@dataclass
class LossResult:
    predictions: pd.DataFrame   # timestamp, ws_pred, ws_p10, ws_p90, power_expected
    energy_kwh: dict            # p10, p50, p90
    metrics: dict               # selection period: ws_rmse, ws_mae, p_rmse, p_mae, energy_dev_kwh/pct, r2, ...
    method: dict                # references, lags, coefficients, selection path, ...
    history: pd.DataFrame = field(repr=False)      # whole period: actual, block-out / fitted prediction
    validation: pd.DataFrame = field(default=None, repr=False)   # selection period: actual vs predicted


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


def extra_features(refs, ref_avg, window, dir_refs=None):
    """Candidate extra features (see EXTRA_GROUPS), built from all references.

    dir_refs: references whose mean wind direction drives the direction harmonics
    (a smooth correction curve that can follow wake dips); default all references.
    Weather terms are standardised (mean 0, std 1).
    """
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
    d = rad[list(dir_refs or refs)]
    th = np.arctan2(np.sin(d).mean(axis=1), np.cos(d).mean(axis=1))
    for k in HARMONICS:
        F[f"h_sin{k}"] = ref_avg * np.sin(k * th)
        F[f"h_cos{k}"] = ref_avg * np.cos(k * th)
    col = lambda c: pd.DataFrame({r: dd[c] for r, dd in refs.items()}).mean(axis=1)
    turb, temp, pres = col("wind_speed_stddev"), col("ambient_temperature"), col("air_pressure")
    roll = lambda s: s.rolling(window, center=True, min_periods=1).mean()
    F["turbulence"] = roll(turb)
    F["temp_change_3h"] = temp.diff(18)                     # warming/cooling ~ day/night stability
    weather = pd.DataFrame({
        "temperature": roll(temp),
        "temp_anomaly": roll(temp - temp.rolling(144, center=True, min_periods=72).mean()),
        "pressure": roll(pres),
        "pressure_change_3h": pres.diff(18),
        "air_density": roll(pres * 100 / (287.05 * (temp + 273.15))),
        "turb_intensity": roll(turb / col("wind_speed")),
    })
    return pd.concat([F, (weather - weather.mean()) / weather.std()], axis=1)


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


def fit_predict(F, cols, y, fit_mask, nonneg):
    """Fit on fit_mask rows, predict wherever the features exist (clipped at 0)."""
    X = F[cols].to_numpy(float)
    has = ~np.isnan(X).any(axis=1)
    fit = has & fit_mask.to_numpy() & y.notna().to_numpy()
    if fit.sum() < 100:
        raise ValueError("Not enough training data for the model fit.")
    model = BoundedLinear(nonneg).fit(X[fit], y.to_numpy()[fit], cols)
    pred = np.full(len(y), np.nan)
    pred[has] = np.clip(model.predict(X[has]), 0, None)
    return model, pd.Series(pred, index=y.index), int(fit.sum())


def selection_period(y, in_outage, days=SELECTION_DAYS):
    """Mask + label of the period used for selection and metrics.

    The shown test week if the target was measured there and it does not overlap the outage,
    otherwise the `days` days right after the outage (else right before).
    """
    idx = y.index
    tw = (idx >= TEST_WEEK[0]) & (idx < TEST_WEEK[1])
    if not (tw & in_outage.to_numpy()).any() and y[tw].notna().mean() > 0.5:
        return pd.Series(tw, index=idx), "test_week", TEST_WEEK
    out_idx = idx[in_outage.to_numpy()]
    step = idx[1] - idx[0]
    for a, b, kind in [(out_idx[-1] + step, out_idx[-1] + step + pd.Timedelta(days=days), "after_outage"),
                       (out_idx[0] - pd.Timedelta(days=days), out_idx[0], "before_outage")]:
        m = (idx >= a) & (idx < b)
        if m.any() and y[m].notna().mean() > 0.5:
            return pd.Series(m, index=idx), kind, (a, b)
    raise ValueError("No measured period next to the outage to select the model on.")


def period_metrics(y, pred, mask, pc, residuals):
    """Judges' metrics on `mask`: wind RMSE/MAE, power RMSE/MAE (power curve on the predicted
    wind) and the deviation of the summed energy (expected power)."""
    ev = (mask & pred.notna() & y.notna()).to_numpy()
    yt, yp = y.to_numpy()[ev], pred.to_numpy()[ev]
    pt, pp, pe = pc.power(yt), pc.power(yp), expected_power(yp, residuals, pc)
    e_true = energy_kwh(pt)
    return {
        "ws_rmse": float(np.sqrt(np.mean((yp - yt) ** 2))), "ws_mae": float(np.mean(np.abs(yp - yt))),
        "p_rmse": float(np.sqrt(np.mean((pp - pt) ** 2))), "p_mae": float(np.mean(np.abs(pp - pt))),
        "energy_pred_kwh": energy_kwh(pe), "energy_true_kwh": e_true,
        "energy_dev_kwh": energy_kwh(pe) - e_true, "energy_dev_kwh_point": energy_kwh(pp) - e_true,
        "energy_dev_pct": float((energy_kwh(pe) - e_true) / e_true * 100) if e_true else np.nan,
        "r2": float(1 - ((yp - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()),
        "n_eval": int(ev.sum()),
    }


def _score(m, base):
    return float(np.mean([m[k] / base[k] for k in ("ws_rmse", "ws_mae", "p_rmse", "p_mae")]))


def _block_energy_errors(y, p_pred, block_id, pc, min_frac=0.5):
    """Relative energy error per block (only blocks with >= min_frac of steps evaluated)."""
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
                  references_to_use=None, select_extras=True, window=None, min_gain=MIN_GAIN):
    """Reconstruct the target's wind speed during `outage` and estimate the lost energy.

    target:      DataFrame of one turbine (timestamp index or column, wind_speed, ...)
    references:  {name: DataFrame} of turbines outside the target park
    power_curve: DataFrame with columns wind_speed, power (kW)
    outage:      (start, end), both inclusive
    references_to_use: restrict the candidate pool to these names (default: all references)
    select_extras: run the forward selection of extra features (False = references only)
    window:      fixed smoothing in 10-min steps (default: selected from WINDOWS)
    """
    pc = PowerCurve(power_curve)
    frames = align({"__target__": target, **references})
    idx = frames["__target__"].index
    y = clean(frames.pop("__target__"))["wind_speed"]
    refs = {n: fill_reference(clean(d)) for n, d in frames.items()}

    start, end = pd.Timestamp(outage[0]), pd.Timestamp(outage[1])
    in_outage = pd.Series((idx >= start) & (idx <= end), index=idx)
    y = y.where(~in_outage)                                  # never learn from inside the outage
    train = y.notna()
    val, val_kind, (val_a, val_b) = selection_period(y, in_outage)
    fit_sel = train & ~val                                   # selection fits never see the selection period

    pool = list(references_to_use or refs)
    unknown = set(pool) - set(refs)
    if unknown:
        raise ValueError(f"Unknown references {sorted(unknown)}; available: {list(refs)}")
    pool = [r for r in pool if refs[r]["wind_speed"][fit_sel].notna().mean() >= 0.5]   # enough overlapping data
    if not pool:
        raise ValueError("No reference turbine has enough data overlapping the target.")

    lags, corrs = {}, {}
    for r in refs:
        lags[r], corrs[r] = find_lag(y, refs[r]["wind_speed"], fit_sel)

    def evaluate(F, cols, nonneg):
        _, pred, _ = fit_predict(F, cols, y, fit_sel, nonneg)
        resid = (y - pred)[fit_sel].dropna().to_numpy()
        if pred[val & y.notna()].notna().mean() < 0.5:     # candidate cannot predict the period
            return None
        return period_metrics(y, pred, val, pc, resid)

    # 0. smoothing window (all references)
    windows = [window] if window else list(WINDOWS)
    R_by_w = {w: reference_features(refs, lags, w) for w in windows}
    all_cols = [f"ws_{r}" for r in pool]
    win_m = {w: evaluate(R_by_w[w], all_cols, all_cols) for w in windows}
    ref_w = WINDOW if WINDOW in win_m else windows[-1]
    window = min(windows, key=lambda w: _score(win_m[w], win_m[ref_w]))
    base = win_m[window]
    path = [{"stage": "smoothing", "step": f"{w * 10} min", "score": _score(win_m[w], win_m[ref_w]),
             **{k: win_m[w][k] for k in ("ws_rmse", "ws_mae", "p_rmse", "p_mae", "energy_dev_pct")}} for w in windows]
    R = R_by_w[window]

    # 1. references, forward
    def step(stage, name, m):
        return {"stage": stage, "step": name, "score": _score(m, base),
                **{k: m[k] for k in ("ws_rmse", "ws_mae", "p_rmse", "p_mae", "energy_dev_pct")}}
    chosen, rest, cur = [], list(pool), None
    while rest:
        trials = {r: evaluate(R, [f"ws_{c}" for c in chosen + [r]], [f"ws_{c}" for c in chosen + [r]]) for r in rest}
        trials = {r: m for r, m in trials.items() if m}
        if not trials:
            break
        r = min(trials, key=lambda k: _score(trials[k], base))
        if cur is not None and _score(cur, base) - _score(trials[r], base) < min_gain:
            break
        chosen.append(r); rest.remove(r); cur = trials[r]
        path.append(step("reference", r, cur))
    ref_cols = [f"ws_{r}" for r in chosen]

    # 2. extra features, forward (direction curve / slope terms built on the chosen references)
    F = pd.concat([R, extra_features(refs, R[ref_cols].mean(axis=1), window, chosen)], axis=1)
    groups = {g: c for g, c in EXTRA_GROUPS.items()
              if set(c) <= set(F.columns) and F.loc[fit_sel, c].notna().all(axis=1).mean() > 0.5}
    cols, added = list(ref_cols), []
    while select_extras and groups:
        trials = {g: evaluate(F, cols + c, ref_cols) for g, c in groups.items()}
        trials = {g: m for g, m in trials.items() if m}
        if not trials:
            break
        g = min(trials, key=lambda k: _score(trials[k], base))
        if _score(cur, base) - _score(trials[g], base) < min_gain:
            break
        cols += groups.pop(g); added.append(g); cur = trials[g]
        path.append(step("feature", g, cur))

    # 3. metrics on the selection period (model fitted without it)
    _, sel_pred, _ = fit_predict(F, cols, y, fit_sel, ref_cols)
    sel_resid = (y - sel_pred)[fit_sel].dropna().to_numpy()
    metrics = period_metrics(y, sel_pred, val, pc, sel_resid)
    label = f"{val_a:%d.%m.}–{val_b - (idx[1] - idx[0]):%d.%m.%Y}"
    metrics.update(basis=val_kind, basis_period=label)
    v = val & sel_pred.notna() & y.notna()
    validation = pd.DataFrame({"ws_actual": y[v], "ws_pred": sel_pred[v],
                               "power_actual": pc.power(y[v].to_numpy()), "power_pred": pc.power(sel_pred[v].to_numpy()),
                               "power_expected": expected_power(sel_pred[v].to_numpy(), sel_resid, pc)})
    validation.index.name = "timestamp"

    # 4. uncertainty: leave-one-3-day-block-out over all data outside the outage
    block_id = make_blocks(train)
    cv = cv_predict(F, cols, y, block_id, ref_cols) if block_id.max() >= 1 else pd.Series(np.nan, index=idx)
    residuals = (y - cv).dropna().to_numpy()
    if len(residuals) < 100:
        residuals = sel_resid
    block_err = _block_energy_errors(y, expected_power(cv.to_numpy(), residuals, pc), block_id, pc)

    # 5. final fit on all data outside the outage
    model, ws_fit, n_fit = fit_predict(F, cols, y, train, ref_cols)
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

    history = pd.DataFrame({"ws_actual": y.where(~in_outage), "ws_cv": cv, "ws_fit": ws_fit,
                            "ref_mean": R[ref_cols].mean(axis=1), "in_outage": in_outage})
    history.index.name = "timestamp"

    coef = dict(zip(cols, model.coef_))
    total_w = sum(max(coef[c], 0) for c in ref_cols) or 1.0
    ref_table = sorted([{"ref": r, "lag_min": lags[r] * 10, "corr": round(corrs[r], 4), "used": r in chosen,
                         "order": chosen.index(r) + 1 if r in chosen else None,
                         "weight": float(coef.get(f"ws_{r}", 0.0)),
                         "share_pct": float(coef.get(f"ws_{r}", 0.0) / total_w * 100),
                         **({} if r in pool else {"note": "too little overlapping data"})}
                        for r in refs], key=lambda q: (q["order"] or 99, -q["corr"]))
    method = {
        "description": ("Weighted sum (weights >= 0) of lag-aligned, "
                        f"{window * 10}-min smoothed wind speeds of {', '.join(chosen)}"
                        + (f", plus {', '.join(added)}" if added else "")
                        + f"; selected on {label}, fitted on {n_fit} 10-min steps outside the outage."),
        "references": chosen,
        "references_available": list(refs),
        "lags_min": {r: lags[r] * 10 for r in refs},
        "correlations": {r: round(corrs[r], 4) for r in refs},
        "extra_features": added,
        "features": cols,
        "coefficients": {c: _py(v) for c, v in coef.items()},
        "intercept": _py(model.intercept_),
        "window_min": window * 10,
        "n_train": n_fit,
        "n_cv_blocks": int(block_id.max()) + 1,
        "n_outage_steps": int(in_outage.sum()),
        "n_missing_steps": int(ws_out.isna().sum()),
        "residual_p10_p90": [_py(q10), _py(q90)],
        "block_energy_error_p10_p90_pct": [_py(e_lo * 100), _py(e_hi * 100)],
        "selection_basis": {"kind": val_kind, "period": label, "min_gain_pct": min_gain * 100},
        "selection_path": path,
        "reference_selection": ref_table,
        "reference_table": ref_table,
        "feature_selection": path,
    }
    return LossResult(predictions, {k: _py(v) for k, v in energy.items()}, metrics, method, history, validation)
