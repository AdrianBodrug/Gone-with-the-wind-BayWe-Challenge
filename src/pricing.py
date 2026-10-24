"""Day-ahead prices and monetary valuation of the lost energy.

Prices are fetched live for exactly the outage window from the Energy-Charts API
(Fraunhofer ISE; data Bundesnetzagentur | SMARD.de, CC BY 4.0). No API key needed.
Every successful download is cached in data/prices/ so the tool still works offline.

    from src.pricing import value_loss
    v = value_loss(result, mode="market")                 # time-matched day-ahead prices
    v = value_loss(result, mode="eeg", price=73.5)        # EEG reference value, 0 in negative hours
    v = value_loss(result, mode="fixed", price=80.0)      # PPA / fixed tariff

Day-ahead prices for day D are published around 13:00 CET on day D-1. For an outage that
runs into hours without published prices (e.g. an outage still ongoing), those hours are
valued *provisionally* by repeating the last published day's price profile, and flagged.
"""
from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .schema import AssessmentResult

API_URL = "https://api.energy-charts.info/price"
BIDDING_ZONE = "DE-LU"
LOCAL_TZ = "Europe/Berlin"
LAST_FETCH_ERROR: Optional[str] = None   # why the last live download failed (shown in reports)
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "prices"
ATTRIBUTION = ("Day-ahead prices: Bundesnetzagentur | SMARD.de, via Energy-Charts API "
               "(Fraunhofer ISE), licence CC BY 4.0")
MODES = {
    "market": "day-ahead market price (EPEX Spot, DE-LU) for each interval",
    "eeg": "EEG reference value (anzulegender Wert); intervals with negative day-ahead price "
           "valued at zero (§51 EEG)",
    "fixed": "fixed remuneration (e.g. PPA price)",
}


# ================================================================ time handling
def to_utc(local_naive) -> pd.DatetimeIndex:
    """Naive Europe/Berlin timestamps -> UTC. DST-safe: the repeated hour on the
    last Sunday of October is resolved by order, the skipped hour in March shifted."""
    idx = pd.DatetimeIndex(local_naive)
    if idx.tz is not None:
        return idx.tz_convert("UTC")
    try:
        loc = idx.tz_localize(LOCAL_TZ, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        loc = idx.tz_localize(LOCAL_TZ, ambiguous=np.zeros(len(idx), bool), nonexistent="shift_forward")
    return loc.tz_convert("UTC")


# ================================================================ fetching
def _parse(payload: dict) -> pd.Series:
    ts = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    s = pd.Series(payload["price"], index=ts, dtype="float64", name="eur_per_mwh").dropna()
    return s[~s.index.duplicated()].sort_index()


def _ssl_context() -> ssl.SSLContext:
    """Use certifi's CA bundle when available - python.org builds on macOS ship without
    system certificates ('CERTIFICATE_VERIFY_FAILED')."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _download(start_utc: pd.Timestamp, end_utc: pd.Timestamp, bzn: str, timeout: float) -> dict:
    # request whole local days with one day of margin; the API takes local dates
    d0 = (start_utc.tz_convert(LOCAL_TZ) - pd.Timedelta("1D")).strftime("%Y-%m-%d")
    d1 = (end_utc.tz_convert(LOCAL_TZ) + pd.Timedelta("1D")).strftime("%Y-%m-%d")
    url = f"{API_URL}?{urllib.parse.urlencode({'bzn': bzn, 'start': d0, 'end': d1})}"
    req = urllib.request.Request(url, headers={"User-Agent": "baywa-outage-loss-tool/0.1"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        payload = json.loads(r.read().decode())
    payload["_request"] = {"url": url, "fetched_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    return payload


def _load_cache(bzn: str) -> pd.Series:
    parts = []
    for f in sorted(CACHE_DIR.glob(f"{bzn}_*.json")):
        try:
            parts.append(_parse(json.loads(f.read_text())))
        except Exception:
            continue
    if not parts:
        return pd.Series(dtype="float64", name="eur_per_mwh")
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_day_ahead(start_utc, end_utc, bzn: str = BIDDING_ZONE, timeout: float = 20,
                    use_cache: bool = True, offline: bool = False) -> tuple[pd.Series, str]:
    """Prices (EUR/MWh, UTC index, interval start) covering [start_utc, end_utc].

    Returns (prices, source) with source 'live' or 'cache'. Live results are cached.
    """
    global LAST_FETCH_ERROR
    start_utc, end_utc = pd.Timestamp(start_utc), pd.Timestamp(end_utc)
    if not offline:
        try:
            payload = _download(start_utc, end_utc, bzn, timeout)
            s = _parse(payload)
            if use_cache and len(s):
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                name = f"{bzn}_{s.index[0]:%Y%m%dT%H%M}_{s.index[-1]:%Y%m%dT%H%M}.json"
                (CACHE_DIR / name).write_text(json.dumps(payload))
            LAST_FETCH_ERROR = None
            return s, "live"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as e:
            LAST_FETCH_ERROR = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            import warnings
            warnings.warn(f"Live day-ahead price download failed ({LAST_FETCH_ERROR}); using local cache.")
    return _load_cache(bzn), "cache"


# ================================================================ matching
def match_prices(times_local, prices: pd.Series) -> pd.Series:
    """Price for each (naive local) 10-min timestamp: the price interval that contains it.
    Works for hourly (before Oct 2025) and quarter-hourly (from Oct 2025) prices.
    NaN where no published price covers the timestamp."""
    t_utc = to_utc(times_local)
    out = pd.Series(np.nan, index=t_utc)
    if prices.empty:
        return out
    starts = prices.index
    dur = pd.Series(starts[1:] - starts[:-1], index=starts[:-1])
    dur = dur.reindex(starts).fillna(dur.iloc[-1] if len(dur) else pd.Timedelta("15min"))
    dur = dur.clip(upper=pd.Timedelta("1h"))
    pos = starts.searchsorted(t_utc, side="right") - 1
    ok = pos >= 0
    p_idx = np.clip(pos, 0, len(starts) - 1)
    inside = ok & (t_utc < starts[p_idx] + dur.to_numpy()[p_idx])
    out[:] = np.where(inside, prices.to_numpy()[p_idx], np.nan)
    return out


PROVISIONAL_DAYS = 1


def provisional_profile(t_utc: pd.DatetimeIndex, prices: pd.Series, days: int = PROVISIONAL_DAYS) -> np.ndarray:
    """Placeholder price for intervals not yet published: the local quarter-hour-of-day profile
    of the last `days` published days (default: repeat the most recent published day).
    Day-ahead prices are strongly correlated day to day, so persistence is a more realistic
    placeholder than a weekly average, e.g. on a windy weekend with low prices."""
    if prices.empty:
        return np.full(len(t_utc), np.nan)
    recent = prices[prices.index >= prices.index[-1] - pd.Timedelta(days=days)]
    loc = recent.index.tz_convert(LOCAL_TZ)
    prof = recent.groupby(loc.hour * 4 + loc.minute // 15).mean()
    q = t_utc.tz_convert(LOCAL_TZ)
    key = q.hour * 4 + q.minute // 15
    return prof.reindex(key).fillna(recent.mean()).to_numpy()


# ================================================================ valuation
@dataclass
class Valuation:
    mode: str
    basis: str
    eur: dict                       # p10, p50, p90
    eur_after_waiting: Optional[float]
    energy_mwh: float
    avg_value_eur_per_mwh: float    # EUR per lost MWh (capture price)
    avg_market_price: Optional[float]
    share_provisional: float        # share of lost energy valued with provisional prices
    n_negative_intervals: int
    energy_negative_mwh: float
    price_source: str               # live / cache / none / n.a.
    price_coverage_until: Optional[pd.Timestamp]   # last published price interval (local)
    attribution: str = ATTRIBUTION
    series: pd.DataFrame = field(default=None, repr=False)

    @property
    def is_provisional(self) -> bool:
        return self.share_provisional > 0.0005


def value_loss(result: AssessmentResult, mode: str = "market", price: Optional[float] = None,
               waiting_period_h: Optional[float] = None, prices: Optional[pd.Series] = None,
               offline: bool = False) -> Optional[Valuation]:
    """Monetary value of the lost energy.

    mode 'market': sum over intervals of lost MWh x day-ahead price (negative prices -> 0,
                   the plant would have curtailed). 'eeg': `price` per MWh, but 0 in intervals
                   with negative day-ahead price. 'fixed': `price` per MWh.
    P10/P90 scale with the energy range. Returns None if nothing can be valued.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}")
    if mode in ("eeg", "fixed") and not price:
        return None

    # aggregate turbines in UTC so the repeated local hour at the end of DST is not merged
    frames = [pd.Series(t.predictions["power_kw"].to_numpy(), index=to_utc(t.predictions["timestamp"]))
              for t in result.turbines]
    farm = pd.concat(frames).groupby(level=0).sum().sort_index()
    t_utc = pd.DatetimeIndex(farm.index)
    local = t_utc.tz_convert(LOCAL_TZ).tz_localize(None)
    e_mwh = (farm / 6 / 1000).to_numpy()

    need_prices = mode in ("market", "eeg")
    source, last_pub = "n.a.", None
    p = np.full(len(farm), np.nan)
    provisional = np.zeros(len(farm), bool)
    if need_prices:
        if prices is None:
            prices, source = fetch_day_ahead(t_utc.min(), t_utc.max(), offline=offline)
        else:
            source = "supplied"
        p = match_prices(t_utc, prices).to_numpy().copy()
        provisional = np.isnan(p)
        if provisional.any():
            p[provisional] = provisional_profile(t_utc[provisional], prices)
        if np.isnan(p).all():
            return None if mode == "market" else _fixed(result, farm, e_mwh, price, waiting_period_h, mode,
                                                        source="none")
        if not prices.empty:
            last_pub = prices.index[-1].tz_convert(LOCAL_TZ).tz_localize(None)

    if mode == "market":
        unit = np.clip(p, 0, None)
    elif mode == "eeg":
        unit = np.where(p < 0, 0.0, price)
    else:
        unit = np.full(len(farm), float(price))
    value = e_mwh * unit

    farm_e = result.farm_energy_kwh()
    ratio = {k: farm_e[k] / farm_e["p50"] if farm_e["p50"] else 1.0 for k in ("p10", "p50", "p90")}
    total = float(np.nansum(value))
    after = None
    if waiting_period_h is not None:
        cut = to_utc([result.outage_start])[0] + pd.Timedelta(hours=waiting_period_h)
        after = float(np.nansum(value[t_utc >= cut]))
    neg = (p < 0) & ~np.isnan(p) if need_prices else np.zeros(len(farm), bool)
    e_total = float(e_mwh.sum())
    series = pd.DataFrame({"timestamp": local, "timestamp_utc": t_utc, "lost_mwh": e_mwh, "price_eur_per_mwh": p,
                           "provisional": provisional, "value_eur": value})
    return Valuation(
        mode=mode, basis=MODES[mode], eur={k: total * r for k, r in ratio.items()},
        eur_after_waiting=after, energy_mwh=e_total,
        avg_value_eur_per_mwh=total / e_total if e_total else float("nan"),
        avg_market_price=float(np.nanmean(p)) if need_prices else None,
        share_provisional=float(e_mwh[provisional].sum() / e_total) if e_total else 0.0,
        n_negative_intervals=int(neg.sum()), energy_negative_mwh=float(e_mwh[neg].sum()),
        price_source=source, price_coverage_until=last_pub, series=series,
    )


def _fixed(result, farm, e_mwh, price, waiting_period_h, mode, source):
    v = value_loss(result, "fixed", price, waiting_period_h)
    v.mode, v.basis, v.price_source = mode, MODES[mode] + " (no day-ahead prices available – "\
        "negative-price rule not applied)", source
    return v


def describe(v: Valuation, price: Optional[float] = None) -> str:
    """One paragraph for the email / report."""
    f = lambda x: f"{x:,.0f}"
    if v.mode == "market":
        head = (f"Valued at the German day-ahead market price (EPEX Spot, DE-LU) for each interval of the "
                f"outage, the lost energy corresponds to approximately EUR {f(v.eur['p50'])} "
                f"(range EUR {f(v.eur['p10'])} – {f(v.eur['p90'])}), i.e. an average of "
                f"{v.avg_value_eur_per_mwh:,.2f} EUR per lost MWh.")
    elif v.mode == "eeg":
        head = (f"Valued at the EEG reference value of {price:,.2f} EUR/MWh, the lost energy corresponds to "
                f"approximately EUR {f(v.eur['p50'])} (range EUR {f(v.eur['p10'])} – {f(v.eur['p90'])}).")
    else:
        head = (f"Valued at the fixed remuneration of {price:,.2f} EUR/MWh, the lost energy corresponds to "
                f"approximately EUR {f(v.eur['p50'])} (range EUR {f(v.eur['p10'])} – {f(v.eur['p90'])}).")
    parts = [head]
    if v.n_negative_intervals:
        rule = "§51 EEG" if v.mode == "eeg" else "the plant would have curtailed"
        parts.append(f"{v.n_negative_intervals} intervals with negative day-ahead prices "
                     f"({v.energy_negative_mwh:,.1f} MWh) are valued at zero ({rule}).")
    if v.is_provisional:
        until = f"{v.price_coverage_until:%d %b %Y %H:%M}" if v.price_coverage_until is not None else "n/a"
        parts.append(f"Day-ahead prices are published only up to {until}; {v.share_provisional:.0%} of the "
                     f"lost energy is therefore valued provisionally by repeating the last published day's "
                     f"price profile and will be updated once prices are published.")
    if v.eur_after_waiting is not None:
        parts.append(f"Beyond the waiting period: EUR {f(v.eur_after_waiting)} (P50).")
    return " ".join(parts)
