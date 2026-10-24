"""Validation on the shown test week and final hidden-window predictions (README tasks A5, A9).

    python -m src.backtest --data data --out predictions_hidden.csv
"""
import argparse

import numpy as np
import pandas as pd

from .data_io import clean, load_folder
from .energy import PowerCurve, energy_kwh
from .model import estimate_loss
from .outage import detect_outages

SHOWN_WEEK = (pd.Timestamp("2026-09-04 00:00"), pd.Timestamp("2026-09-09 23:50"))


def split(turbines, target_park="T"):
    targets = {n: d for n, d in turbines.items() if n.startswith(target_park)}
    refs = {n: d for n, d in turbines.items() if not n.startswith(target_park)}
    return targets, refs


def score_window(target, references, power_curve, window, **kwargs):
    """Fake an outage over `window`, predict it, and score against the real values."""
    start, end = window
    masked = target.copy()
    masked.loc[(masked.index >= start) & (masked.index <= end), "wind_speed"] = np.nan
    res = estimate_loss(masked, references, power_curve, window, **kwargs)

    pc = PowerCurve(power_curve)
    truth = clean(target)["wind_speed"].reindex(res.predictions["timestamp"]).to_numpy()
    pred = res.predictions["ws_pred"].to_numpy()
    p_pred = res.predictions["power_expected"].to_numpy()
    ok = ~np.isnan(truth) & ~np.isnan(pred)
    p_true = pc.power(truth[ok])
    e_true = energy_kwh(p_true)
    return {
        "ws_rmse": float(np.sqrt(((pred[ok] - truth[ok]) ** 2).mean())),
        "ws_mae": float(np.abs(pred[ok] - truth[ok]).mean()),
        "p_rmse": float(np.sqrt(((p_pred[ok] - p_true) ** 2).mean())),
        "energy_true_kwh": e_true,
        "energy_pred_kwh": energy_kwh(p_pred[ok]),
        "energy_dev_pct": (energy_kwh(p_pred[ok]) - e_true) / e_true * 100,
        "in_p10_p90": res.energy_kwh["p10"] <= e_true <= res.energy_kwh["p90"],
    }


def shown_week_score(turbines, power_curves, **kwargs):
    targets, refs = split(turbines)
    rows = {t: score_window(d, refs, power_curves[t], SHOWN_WEEK, **kwargs) for t, d in targets.items()}
    return pd.DataFrame(rows).T


def predict_hidden(turbines, power_curves, **kwargs):
    """Run estimate_loss on each target's longest detected outage."""
    targets, refs = split(turbines)
    results = {}
    for t, d in targets.items():
        outage = detect_outages(d, refs)[0]
        results[t] = estimate_loss(d, refs, power_curves[t], outage, **kwargs)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="predictions_hidden.csv")
    ap.add_argument("--no-extras", action="store_true", help="references-only model")
    ap.add_argument("--skip-shown", action="store_true", help="skip the shown-week validation")
    ap.add_argument("--refs", default="auto",
                    help="'auto' (backward elimination), 'all', or a comma list like H1,O1")
    args = ap.parse_args()
    kwargs = {"select_extras": not args.no_extras}

    turbines, curves = load_folder(args.data)
    if args.refs == "all":
        kwargs["references_to_use"] = list(split(turbines)[1])
    elif args.refs != "auto":
        kwargs["references_to_use"] = [r.strip() for r in args.refs.split(",")]
    pd.set_option("display.width", 160)

    if not args.skip_shown:
        print("=== Shown test week (4-9 Sep) ===")
        print(shown_week_score(turbines, curves, **kwargs).round(3).to_string(), "\n")

    print("=== Hidden outage ===")
    results = predict_hidden(turbines, curves, **kwargs)
    out = []
    for t, r in results.items():
        p = r.predictions
        print(f"{t}: {p['timestamp'].min()} -> {p['timestamp'].max()}  "
              f"lost {r.energy_kwh['p50'] / 1000:.1f} MWh "
              f"(P10 {r.energy_kwh['p10'] / 1000:.1f} - P90 {r.energy_kwh['p90'] / 1000:.1f})")
        print(f"    {r.method['description']}")
        print("    CV: " + ", ".join(f"{k}={v:.3f}" for k, v in r.metrics.items() if k != "n_eval"))
        out.append(p.assign(turbine=t))
    total = sum(r.energy_kwh["p50"] for r in results.values())
    print(f"Total lost: {total / 1000:.1f} MWh")

    cols = ["timestamp", "turbine", "ws_pred", "ws_p10", "ws_p90", "power_expected"]
    pd.concat(out)[cols].to_csv(args.out, index=False)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
