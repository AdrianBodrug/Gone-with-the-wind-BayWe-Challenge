"""Adapter between the WindGap web page (templates/windgap.html) and estimate_loss().

The page POSTs (see its "Integration" tab)
    {target, window: {start, end}, reference: "auto" | [names], power_curve: [[ws, kw], ...],
     data: {t0, step_min, series: {name: [wind speed or null, ...]}}}       # data is optional
and gets back
    {predicted_wind, power_kw, low_kw, high_kw, reference, rank, model, metrics, reference_table}
with one value per 10-min step in the window (end exclusive).

The page only sends wind speeds. Turbines whose series match the server's data folder are
enriched with their other channels (direction, temperature, turbulence) from there, so the
direction and turbulence features still work; new turbines (e.g. an uploaded T4) run on
wind speed alone.
"""
import hashlib
import json
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

from .data_io import clean, load_folder
from .energy import PowerCurve, energy_kwh
from .model import estimate_loss

EXTRA_CHANNELS = ["wind_direction", "ambient_temperature", "air_pressure", "wind_speed_stddev"]
SHIFTS_H = (0, -1, 1, -2, 2)          # tolerate uploads read in local time instead of wall-clock time
MATCH_TOL = 0.011                     # the page rounds wind speeds to 2 decimals
CACHE_SIZE = 16

FEATURE_LABELS = {
    "hour": "Tageszeit", "ref_x_hour": "Tageszeit × Wind", "direction": "Windrichtung",
    "ref_x_dir": "Windrichtung × Wind", "dir_harmonics": "Richtungskurve (Nachlauf)",
    "turbulence": "Turbulenz", "temp_change_3h": "Temperaturänderung 3 h",
}

_cache = OrderedDict()


def park(name):
    """H3 -> H, T12 -> T (same rule as the page)."""
    return re.sub(r"\d+$", "", name) or name


class Store:
    """Full turbine files and power curves from the server's data folder."""

    def __init__(self, data_dir="data"):
        self.data_dir = str(data_dir)
        self.turbines, self.curves = load_folder(str(data_dir))


def _naive(ts):
    t = pd.Timestamp(ts)
    return (t.tz_convert(None) if t.tzinfo else t).round("min")


def _match_share(s, full):
    """Share of overlapping steps where an uploaded series equals the stored wind speed."""
    ref = full.reindex(s.index)
    both = s.notna() & ref.notna()
    if both.sum() < 100:
        return 0.0
    return float(((s[both] - ref[both]).abs() <= MATCH_TOL).mean())


def _series(payload):
    data = payload["data"]
    step = pd.Timedelta(minutes=float(data.get("step_min", 10)))
    series = data["series"]
    idx = pd.date_range(_naive(data["t0"]), periods=max(len(v) for v in series.values()), freq=step)
    ws = {n: pd.Series([np.nan if x is None else float(x) for x in v], index=idx[:len(v)])
          for n, v in series.items()}
    return ws, step


def _frames(payload, store):
    """{name: DataFrame} on the stored time convention, plus the applied shift and step."""
    if not payload.get("data"):
        return {n: d.copy() for n, d in store.turbines.items()}, pd.Timedelta(0), pd.Timedelta("10min")
    ws, step = _series(payload)
    known = [n for n in ws if n in store.turbines]
    shift = pd.Timedelta(0)
    if known:
        hits = {h: sum(_match_share(ws[n].shift(freq=pd.Timedelta(hours=h)), store.turbines[n]["wind_speed"]) >= 0.9
                       for n in known) for h in SHIFTS_H}
        best = max(SHIFTS_H, key=lambda h: (hits[h], h == 0))
        if hits[best]:
            shift = pd.Timedelta(hours=best)
    frames = {}
    for n, s in ws.items():
        s = s.shift(freq=shift)
        full = store.turbines.get(n)
        if full is not None and _match_share(s, full["wind_speed"]) >= 0.9:
            df = full.reindex(s.index)                  # same turbine: full precision + all channels
            df.loc[s.isna(), "wind_speed"] = np.nan     # keep gaps the page sends (e.g. a masked backtest)
        else:
            df = pd.DataFrame({"wind_speed": s, **{c: np.nan for c in EXTRA_CHANNELS}})
        df["turbine_name"] = n
        frames[n] = df
    return frames, shift, step


def _clean_json(x):
    """NaN/inf -> None and numpy -> Python, so the browser's JSON.parse accepts it."""
    if isinstance(x, dict):
        return {str(k): _clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean_json(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _scores(pred_ws, power_kw, truth, pc):
    """Same keys as the page's own metr()."""
    ok = ~np.isnan(truth) & ~np.isnan(pred_ws)
    if not ok.any():
        return None
    t, p, pw = truth[ok], pred_ws[ok], power_kw[ok]
    pt = pc.power(t)
    e_pred, e_act = energy_kwh(pw), energy_kwh(pt)
    return {"n": int(ok.sum()), "wind_rmse": float(np.sqrt(((p - t) ** 2).mean())),
            "wind_mae": float(np.abs(p - t).mean()), "power_rmse": float(np.sqrt(((pw - pt) ** 2).mean())),
            "power_mae": float(np.abs(pw - pt).mean()), "energy_pred_kwh": e_pred, "energy_act_kwh": e_act,
            "energy_dev_kwh": e_pred - e_act, "energy_dev_pct": (e_pred - e_act) / e_act * 100 if e_act else None}


def _equation(method):
    coefs, lags = method["coefficients"], method["lags_min"]
    terms = [f"{coefs[f'ws_{r}']:.2f} · {r}" + (f" ({lags[r]:+d} min)" if lags[r] else "")
             for r in sorted(method["references"], key=lambda r: -coefs[f"ws_{r}"])]
    extras = [FEATURE_LABELS.get(g, g) for g in method["extra_features"]]
    return f"{method['intercept']:.2f} + " + " + ".join(terms) + (" + Korrekturen: " + ", ".join(extras) if extras else "")


def analyze(payload, store):
    """Run estimate_loss() for one page request and return the page's response dict."""
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    target = payload["target"]
    frames, shift, step = _frames(payload, store)
    if target not in frames:
        raise ValueError(f"Zielturbine {target} nicht in den Daten.")
    refs = {n: d for n, d in frames.items() if park(n) != park(target)}
    if not refs:
        raise ValueError("Keine Referenzturbinen außerhalb des Zielparks gefunden.")
    pool = payload.get("reference", "auto")
    pool = None if pool in (None, "auto") else [r for r in pool if r in refs] or None

    start = _naive(payload["window"]["start"]) + shift
    end_excl = _naive(payload["window"]["end"]) + shift
    window = pd.date_range(start, end_excl - step, freq=step)
    if payload.get("power_curve"):
        pc_df = pd.DataFrame(payload["power_curve"], columns=["wind_speed", "power"]).astype(float)
    elif target in store.curves:
        pc_df = store.curves[target]
    else:
        raise ValueError(f"Keine Leistungskurve für {target}.")
    pc = PowerCurve(pc_df)

    res = estimate_loss(frames[target], refs, pc_df, (window[0], window[-1]), references_to_use=pool)
    p = res.predictions.set_index("timestamp").reindex(window)
    missing = p["ws_pred"].isna()
    if missing.mean() > 0.5:      # e.g. 3-4 Aug: rows are absent in every file, references included
        raise ValueError(f"Im gewählten Fenster fehlen auch die Referenzdaten (nur {1 - missing.mean():.0%} "
                         "der Zeitschritte rekonstruierbar) – keine Schätzung möglich.")
    ws_pred = p["ws_pred"].interpolate(limit_direction="both").to_numpy()   # reference gaps: bridge
    power = np.where(missing, pc.power(ws_pred), p["power_expected"].to_numpy())
    e = res.energy_kwh
    lo_f, hi_f = (e["p10"] / e["p50"], e["p90"] / e["p50"]) if e["p50"] else (1.0, 1.0)

    m, mt = res.method, res.metrics
    truth = clean(frames[target])["wind_speed"].reindex(window).to_numpy()
    test = _scores(ws_pred, power, truth, pc) if np.isfinite(truth).mean() > 0.5 else None
    table = m["reference_table"]
    out = {
        "predicted_wind": ws_pred.tolist(),
        "power_kw": power.tolist(),
        "low_kw": (power * lo_f).tolist(),        # scaled so the sums equal the P10 / P90 energy
        "high_kw": (power * hi_f).tolist(),
        "reference": m["references"],
        "rank": [{"n": r["ref"], "r": r["corr"], "weight": r["weight"], "used": r["used"]} for r in table],
        "reference_table": table,
        "filled_steps": int(missing.sum()),
        "energy_kwh": e,
        "model": {
            "type": "bounded-linear", "intercept": m["intercept"],
            "slope": float(sum(m["coefficients"][f"ws_{r}"] for r in m["references"])),
            "r": float(np.sqrt(max(mt["r2"], 0))), "equation": _equation(m),
            "description": m["description"], "coefficients": m["coefficients"],
            "extra_features": m["extra_features"], "feature_selection": m["feature_selection"],
            "n_train": m["n_train"], "cv_blocks": m["n_cv_blocks"], "window_min": m["window_min"],
            "time_shift_h": shift.total_seconds() / 3600,
        },
        "metrics": {
            "validation": {"n": mt["n_eval"], "wind_rmse": mt["ws_rmse"], "wind_mae": mt["ws_mae"],
                           "power_rmse": mt["p_rmse"], "power_mae": mt["p_mae"],
                           "energy_dev_kwh": mt["energy_dev_kwh"], "energy_dev_pct": mt["energy_dev_pct"]},
            "test": test,
        },
    }
    out = _clean_json(out)
    print(f"analyze: {target}, {len(window)} steps, {len(m['references'])} references, "
          f"{m['n_train']} training rows, {m['n_cv_blocks']} CV blocks, {mt['n_eval']} validation rows, {out['filled_steps']} filled steps") 
    print(f"Model: {m['description']}, slope {out['model']['slope']:.2f}, r {out['model']['r']:.2f}, "
          f"features {', '.join(m['extra_features']) if m['extra_features'] else 'none'}, equation: {out['model']['equation']}")
    print("================================================================")
    _cache[key] = out
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return out
