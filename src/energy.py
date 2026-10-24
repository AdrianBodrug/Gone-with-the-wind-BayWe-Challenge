"""Power curve & energy calculation (README task A7)."""
import numpy as np

N_QUANTILES = 41


class PowerCurve:
    """Wind speed (m/s) -> power (kW), linear interpolation between the table points."""

    def __init__(self, df):
        self.ws = np.r_[0.0, df["wind_speed"].to_numpy(float)]   # add a 0 m/s -> 0 kW point
        self.kw = np.r_[0.0, df["power"].to_numpy(float)]

    def power(self, v):
        v = np.asarray(v, float)
        p = np.interp(v, self.ws, self.kw)
        p[v > self.ws[-1]] = 0.0          # beyond the last table point: turbine off
        p[np.isnan(v)] = np.nan           # missing wind stays missing
        return p


def energy_kwh(power_kw):
    """Energy (kWh) of a series of 10-min average powers: power x 1/6 h, summed."""
    return float(np.nansum(power_kw) / 6)


def expected_power(ws_pred, residuals, pc, n=N_QUANTILES):
    """Average power over the model's error distribution instead of P(point prediction).

    Regression predictions are smoother than reality and the power curve is convex in the
    main operating range, so P(ws_pred) underestimates energy. Here each step's power is the
    mean of P(ws_pred + r) over n quantiles r of the cross-validated residuals.
    """
    ws_pred = np.asarray(ws_pred, float)
    r = np.quantile(np.asarray(residuals, float), (np.arange(n) + 0.5) / n)
    samples = np.clip(ws_pred[:, None] + r[None, :], 0, None)
    p = pc.power(samples.ravel()).reshape(samples.shape).mean(axis=1)
    p[np.isnan(ws_pred)] = np.nan
    return p
