"""CSV export of an outage estimate: one file per turbine, a park total and a metrics summary.

    POST /api/export   (page "CSV exportieren" button) -> ZIP
    python -m src.export --start 2026-09-01 --end 2026-09-04 --out outputs/export_hidden

Every prediction file has one row per 10-min step of the outage window (end exclusive):
predicted wind and power with their P10/P90 range, cumulative energy and, if the window has
measurements (backtest), the measured values.

metrics_summary.csv: accuracy on the selection period - the shown test week 4-9 Sep, on which
references and features are selected, with the model fitted without that week (see src/model.py) -
wind RMSE/MAE (m/s), power RMSE/MAE (kW, power curve on the predicted wind) and the deviation of
the summed energy yield (expected power), per turbine and in total, plus the outage loss estimate
(P10/P50/P90). TOTAL pools all turbine steps for RMSE/MAE and sums the energy over the turbines.
"""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import clean
from .energy import energy_kwh
from .webapi import Store, park, run_turbine


def _prediction_table(run):
    res, window, pc = run["res"], run["window"], run["pc"]
    p = res.predictions.set_index("timestamp").reindex(window)
    missing = p["ws_pred"].isna()
    fill = lambda s: s.interpolate(limit_direction="both")              # bridge short reference gaps
    ws = fill(p["ws_pred"])
    power = pd.Series(np.where(missing, pc.power(ws.to_numpy()), p["power_expected"].to_numpy()), index=window)
    e = res.energy_kwh
    lo_f, hi_f = (e["p10"] / e["p50"], e["p90"] / e["p50"]) if e["p50"] else (1.0, 1.0)
    truth = clean(run["frame"])["wind_speed"].reindex(window)
    return pd.DataFrame({
        "timestamp": window,
        "wind_speed_pred_ms": ws.to_numpy(),
        "wind_speed_p10_ms": fill(p["ws_p10"]).to_numpy(),
        "wind_speed_p90_ms": fill(p["ws_p90"]).to_numpy(),
        "power_pred_kw": power.to_numpy(),
        "power_p10_kw": (power * lo_f).to_numpy(),                       # sums equal the P10 / P90 energy
        "power_p90_kw": (power * hi_f).to_numpy(),
        "energy_cum_kwh": (power.cumsum() / 6).to_numpy(),
        "wind_speed_measured_ms": truth.to_numpy(),
        "power_measured_kw": pc.power(truth.to_numpy()),
    })


def _selection_errors(res):
    """Errors on the model's selection period (test week, model fitted without it)."""
    v = res.validation
    if v is None or v.empty:
        return None
    return ((v["ws_pred"] - v["ws_actual"]).to_numpy(), (v["power_pred"] - v["power_actual"]).to_numpy(),
            energy_kwh(v["power_expected"].to_numpy()), energy_kwh(v["power_actual"].to_numpy()))


def _metrics_row(name, period, errs, loss):
    row = {"turbine": name, "test_period": period}
    if errs is None:
        row["note"] = "no measurements in the selection period"
    else:
        werr, perr, e_pred, e_true = errs
        row.update({"n_steps": len(werr),
                    "wind_rmse_ms": float(np.sqrt(np.mean(werr ** 2))), "wind_mae_ms": float(np.mean(np.abs(werr))),
                    "power_rmse_kw": float(np.sqrt(np.mean(perr ** 2))), "power_mae_kw": float(np.mean(np.abs(perr))),
                    "energy_pred_kwh": e_pred, "energy_measured_kwh": e_true, "energy_dev_kwh": e_pred - e_true,
                    "energy_dev_pct": (e_pred - e_true) / e_true * 100 if e_true else np.nan})
    row.update({"outage_loss_p10_kwh": loss["p10"], "outage_loss_p50_kwh": loss["p50"], "outage_loss_p90_kwh": loss["p90"]})
    return row


def _metrics(runs):
    """Accuracy on the selection period (model fitted without it) + the outage loss estimate."""
    errs = {t: _selection_errors(r["res"]) for t, r in runs.items()}
    period = {t: r["res"].metrics.get("basis_period", "") for t, r in runs.items()}
    rows = [_metrics_row(t, period[t], errs[t], runs[t]["res"].energy_kwh) for t in runs]
    if len(runs) > 1:
        loss = {k: sum(runs[t]["res"].energy_kwh[k] for t in runs) for k in ("p10", "p50", "p90")}
        ok = [e for e in errs.values() if e is not None]
        total = None if len(ok) < len(errs) else (np.concatenate([e[0] for e in ok]), np.concatenate([e[1] for e in ok]),
                                                  sum(e[2] for e in ok), sum(e[3] for e in ok))
        rows.append(_metrics_row("TOTAL", period[next(iter(runs))], total, loss))
    return pd.DataFrame(rows)


def _park_table(tables):
    ts = list(tables.values())
    stack = lambda col: np.column_stack([t[col].to_numpy() for t in ts])
    return pd.DataFrame({
        "timestamp": ts[0]["timestamp"],
        "wind_speed_pred_mean_ms": stack("wind_speed_pred_ms").mean(axis=1),
        "power_pred_kw": stack("power_pred_kw").sum(axis=1),
        "power_p10_kw": stack("power_p10_kw").sum(axis=1),
        "power_p90_kw": stack("power_p90_kw").sum(axis=1),
        "energy_cum_kwh": stack("energy_cum_kwh").sum(axis=1),
        "wind_speed_measured_mean_ms": stack("wind_speed_measured_ms").mean(axis=1),   # NaN if any turbine missing
        "power_measured_kw": stack("power_measured_kw").sum(axis=1),
    })


def export_tables(payload, store):
    """{filename: DataFrame} for the page request (targets = list of turbines, else target)."""
    targets = payload.get("targets") or [payload["target"]]
    curves = payload.get("power_curves") or {}
    runs = {}
    for t in targets:
        p = {k: payload.get(k) for k in ("window", "reference", "data")}
        p["target"] = t
        p["power_curve"] = curves.get(t) or (payload.get("power_curve") if t == payload.get("target") else None)
        runs[t] = run_turbine(p, store)
    tables = {t: _prediction_table(r) for t, r in runs.items()}
    out = {f"{t}_predictions.csv": tab for t, tab in tables.items()}
    if len(tables) > 1:
        out[f"park_{park(targets[0])}_total_predictions.csv"] = _park_table(tables)

    out["metrics_summary.csv"] = _metrics(runs)
    return out, runs[targets[0]]["window"]


def _csv(df):
    num = df.select_dtypes("number").columns
    return df.assign(**{c: df[c].round(4) for c in num}).to_csv(index=False, date_format="%Y-%m-%d %H:%M")


def export_zip(payload, store):
    """(zip bytes, filename) with all CSVs."""
    tables, window = export_tables(payload, store)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            z.writestr(name, _csv(df))
    return buf.getvalue(), f"windgap_export_{window[0]:%Y%m%d_%H%M}_{window[-1]:%Y%m%d_%H%M}.zip"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--start", required=True, help="first 10-min step, e.g. 2026-09-01 (Berlin time as in the files)")
    ap.add_argument("--end", required=True, help="end, exclusive, e.g. 2026-09-04")
    ap.add_argument("--targets", default="T1,T2,T3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    store = Store(args.data)
    payload = {"targets": args.targets.split(","), "target": args.targets.split(",")[0],
               "window": {"start": args.start, "end": args.end}, "reference": "auto"}
    tables, _ = export_tables(payload, store)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        (out / name).write_text(_csv(df))
        print("wrote", out / name, f"({len(df)} rows)")


if __name__ == "__main__":
    main()
