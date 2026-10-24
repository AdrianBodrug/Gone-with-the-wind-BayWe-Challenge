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
    "eeg": "EEG market premium: day-ahead price + max(0, reference value − monthly market value "
           "of onshore wind); zero in negative-price intervals (§51 EEG)",
    "ppa": "fixed pay-as-produced PPA price",
}
MODE_ALIASES = {"fixed": "ppa"}
MODE_LABELS = {"market": "Day-ahead market", "eeg": "EEG market premium", "ppa": "PPA (fixed price)"}

# ---- defaults when the operator has not given its contract figures --------------------
# EEG reference value ("anzulegender Wert"): onshore wind plants in operation in 2026 were
# mostly awarded in the 2023-2024 tenders, which cleared just below the 7.35 ct/kWh cap
# (Aug 2024 volume-weighted average: 7.33 ct/kWh, Bundesnetzagentur). New awards are far
# lower (Aug 2026: 4.79 ct/kWh). Site-quality correction factors are not applied.
EEG_DEFAULT_EUR_MWH = 73.30
EEG_DEFAULT_SOURCE = ("typical for plants now in operation: Bundesnetzagentur onshore wind tender "
                      "Aug 2024, volume-weighted average award 7.33 ct/kWh")
# PPA: a PPA is fixed when signed, so the relevant benchmark is the price at which German
# onshore wind pay-as-produced PPAs were concluded, not today's spot average.
PPA_DEFAULT_EUR_MWH = 73.13
PPA_DEFAULT_SOURCE = ("German onshore wind pay-as-produced PPA benchmark, national average "
                      "73.13 EUR/MWh (Veyt, 2025)")
# Official monthly market values of onshore wind ("Marktwert Wind an Land", EUR/MWh),
# published by the TSOs on netztransparenz.de. Months not listed are estimated live.
OFFICIAL_WIND_MARKET_VALUE = {
    (2026, 8): 115.45,
    (2026, 9): 109.40,
}
WIND_MV_SOURCE_OFFICIAL = "official monthly market value (netztransparenz.de)"
WIND_MV_SOURCE_ESTIMATE = "estimated from Energy-Charts onshore wind generation x day-ahead price"


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


# ================================================================ wind market value
def _get_json(endpoint: str, params: dict, timeout: float = 30) -> dict:
    url = f"https://api.energy-charts.info/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "baywa-outage-loss-tool/0.1"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        return json.loads(r.read().decode())


def wind_market_value(year: int, month: int, offline: bool = False) -> tuple[Optional[float], str]:
    """Monthly market value of German onshore wind (EUR/MWh) for the EEG premium.

    Official value if known; otherwise generation-weighted day-ahead price of onshore wind for
    that month (month-to-date if running), computed live and cached. Note: the live estimate ran
    ~7 % above the official value for Sep 2026 (117.6 vs 109.4) - treat it as indicative.
    """
    if (year, month) in OFFICIAL_WIND_MARKET_VALUE:
        return OFFICIAL_WIND_MARKET_VALUE[(year, month)], WIND_MV_SOURCE_OFFICIAL
    cache = CACHE_DIR / f"wind_mv_{year}{month:02d}.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("complete"):
            return c["value"], WIND_MV_SOURCE_ESTIMATE
    if offline:
        return None, "unavailable (offline)"
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthEnd(1))
    try:
        rng = {"start": f"{start:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}"}
        gen = _get_json("public_power", {"country": "de", **rng})
        pr = _get_json("price", {"bzn": BIDDING_ZONE, **rng})
        wind = next(t["data"] for t in gen["production_types"] if t["name"] == "Wind onshore")
        g = pd.Series(wind, index=gen["unix_seconds"], dtype="float64")
        p = pd.Series(pr["price"], index=pr["unix_seconds"], dtype="float64")
        p = p.reindex(g.index).ffill(limit=3)        # hourly prices before Oct 2025
        ok = g.notna() & p.notna()
        value = float((g[ok] * p[ok]).sum() / g[ok].sum())
        complete = pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None) > end + pd.Timedelta("1D")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"value": value, "complete": bool(complete)}))
        return value, WIND_MV_SOURCE_ESTIMATE + ("" if complete else " (month to date)")
    except Exception as e:  # network / format problems -> caller falls back
        return None, f"unavailable ({type(e).__name__})"


# ================================================================ valuation
@dataclass
class Valuation:
    mode: str
    label: str
    basis: str
    eur: dict                       # p10, p50, p90
    eur_after_waiting: Optional[float]
    energy_mwh: float
    avg_value_eur_per_mwh: float    # EUR per lost MWh
    avg_market_price: Optional[float]
    share_provisional: float        # share of lost energy valued with provisional prices
    n_negative_intervals: int
    energy_negative_mwh: float
    price_source: str               # live / cache / supplied / n.a.
    price_coverage_until: Optional[pd.Timestamp]
    unit_price: Optional[float] = None          # EEG reference value or PPA price used
    unit_price_source: str = ""                  # "operator" or the default's source
    market_values: dict = field(default_factory=dict)   # "YYYY-MM" -> (EUR/MWh, source)
    premium_eur: float = 0.0                     # EEG market premium part of the value
    attribution: str = ATTRIBUTION
    series: pd.DataFrame = field(default=None, repr=False)

    @property
    def is_provisional(self) -> bool:
        return self.share_provisional > 0.0005


def _farm_series(result: AssessmentResult):
    # aggregate turbines in UTC so the repeated local hour at the end of DST is not merged
    frames = [pd.Series(t.predictions["power_kw"].to_numpy(), index=to_utc(t.predictions["timestamp"]))
              for t in result.turbines]
    farm = pd.concat(frames).groupby(level=0).sum().sort_index()
    t_utc = pd.DatetimeIndex(farm.index)
    return t_utc, t_utc.tz_convert(LOCAL_TZ).tz_localize(None), (farm / 6 / 1000).to_numpy()


def _interval_prices(t_utc, prices, offline):
    source = "supplied"
    if prices is None:
        prices, source = fetch_day_ahead(t_utc.min(), t_utc.max(), offline=offline)
    p = match_prices(t_utc, prices).to_numpy().copy()
    provisional = np.isnan(p)
    if provisional.any() and not prices.empty:
        p[provisional] = provisional_profile(t_utc[provisional], prices)
    last_pub = prices.index[-1].tz_convert(LOCAL_TZ).tz_localize(None) if not prices.empty else None
    return p, provisional, source, last_pub, prices


def value_loss(result: AssessmentResult, mode: str = "market", price: Optional[float] = None,
               waiting_period_h: Optional[float] = None, prices: Optional[pd.Series] = None,
               offline: bool = False, _cache: Optional[dict] = None) -> Optional[Valuation]:
    """Monetary value of the lost energy (P50, with P10/P90 scaled by the energy range).

    market: lost MWh x day-ahead price per interval (negative -> 0: the plant would curtail).
    eeg:    day-ahead price + market premium max(0, reference value - monthly onshore-wind market
            value), zero in negative-price intervals (§51 EEG). `price` = reference value
            (default EEG_DEFAULT_EUR_MWH).
    ppa:    `price` per MWh (default PPA_DEFAULT_EUR_MWH).
    Returns None if the needed prices are unavailable.
    """
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}")
    t_utc, local, e_mwh = _farm_series(result)
    unit_src = "operator"
    if mode == "eeg" and not price:
        price, unit_src = EEG_DEFAULT_EUR_MWH, EEG_DEFAULT_SOURCE
    if mode == "ppa" and not price:
        price, unit_src = PPA_DEFAULT_EUR_MWH, PPA_DEFAULT_SOURCE

    p = np.full(len(e_mwh), np.nan)
    provisional = np.zeros(len(e_mwh), bool)
    source, last_pub, premium = "n.a.", None, np.zeros(len(e_mwh))
    mvs = {}
    if mode in ("market", "eeg"):
        cached = (_cache or {}).get("prices")
        if cached is None:
            cached = _interval_prices(t_utc, prices, offline)
            if _cache is not None:
                _cache["prices"] = cached
        p, provisional, source, last_pub, _ = cached
        if np.isnan(p).all():
            return None

    if mode == "market":
        unit = np.clip(p, 0, None)
    elif mode == "eeg":
        months = pd.PeriodIndex(local, freq="M")
        mv = np.full(len(e_mwh), np.nan)
        for m in months.unique():
            val, src = wind_market_value(m.year, m.month, offline=offline)
            if val is None:     # fall back to the value of the lost energy itself
                sel = months == m
                w = e_mwh[sel] * np.clip(p[sel], 0, None)
                val, src = float(w.sum() / e_mwh[sel].sum()), "proxy: value of the lost energy in this window"
            mvs[str(m)] = (val, src)
            mv[months == m] = val
        prem = np.clip(price - mv, 0, None)
        nonneg = p >= 0
        unit = np.where(nonneg, p + prem, 0.0)
        premium = np.where(nonneg, prem, 0.0) * e_mwh
    else:
        unit = np.full(len(e_mwh), float(price))
    value = e_mwh * unit

    farm_e = result.farm_energy_kwh()
    ratio = {k: farm_e[k] / farm_e["p50"] if farm_e["p50"] else 1.0 for k in ("p10", "p50", "p90")}
    total = float(np.nansum(value))
    after = None
    if waiting_period_h is not None:
        cut = to_utc([result.outage_start])[0] + pd.Timedelta(hours=waiting_period_h)
        after = float(np.nansum(value[t_utc >= cut]))
    has_p = ~np.isnan(p)
    neg = has_p & (p < 0)
    e_total = float(e_mwh.sum())
    series = pd.DataFrame({"timestamp": local, "timestamp_utc": t_utc, "lost_mwh": e_mwh,
                           "price_eur_per_mwh": p, "provisional": provisional, "value_eur": value})
    return Valuation(
        mode=mode, label=MODE_LABELS[mode], basis=MODES[mode],
        eur={k: total * r for k, r in ratio.items()}, eur_after_waiting=after, energy_mwh=e_total,
        avg_value_eur_per_mwh=total / e_total if e_total else float("nan"),
        avg_market_price=float(np.nanmean(p)) if has_p.any() else None,
        share_provisional=float(e_mwh[provisional].sum() / e_total) if (e_total and has_p.any()) else 0.0,
        n_negative_intervals=int(neg.sum()), energy_negative_mwh=float(e_mwh[neg].sum()),
        price_source=source, price_coverage_until=last_pub,
        unit_price=price if mode != "market" else None, unit_price_source=unit_src if mode != "market" else "",
        market_values=mvs, premium_eur=float(premium.sum()), series=series,
    )


def value_all(result: AssessmentResult, eeg_price: Optional[float] = None, ppa_price: Optional[float] = None,
              waiting_period_h: Optional[float] = None, prices: Optional[pd.Series] = None,
              offline: bool = False) -> dict:
    """All three valuations with a single price download: {'market', 'eeg', 'ppa'} -> Valuation|None."""
    cache: dict = {}
    return {
        "market": value_loss(result, "market", None, waiting_period_h, prices, offline, cache),
        "eeg": value_loss(result, "eeg", eeg_price, waiting_period_h, prices, offline, cache),
        "ppa": value_loss(result, "ppa", ppa_price, waiting_period_h, prices, offline, cache),
    }


def describe(v: Valuation, price: Optional[float] = None) -> str:
    """One paragraph for the email / report."""
    f = lambda x: f"{x:,.0f}"
    rng = f"approximately EUR {f(v.eur['p50'])} (range EUR {f(v.eur['p10'])} – {f(v.eur['p90'])})"
    default = v.unit_price_source not in ("", "operator")
    if v.mode == "market":
        head = (f"Valued at the German day-ahead market price (EPEX Spot, DE-LU) for each interval of the "
                f"outage, the lost energy corresponds to {rng}, i.e. an average of "
                f"{v.avg_value_eur_per_mwh:,.2f} EUR per lost MWh.")
    elif v.mode == "eeg":
        mv_txt = "; ".join(f"{k}: {val:,.2f} EUR/MWh" for k, (val, _) in v.market_values.items())
        head = (f"Under the EEG market premium, the operator receives the market price plus a premium equal to "
                f"the reference value ({v.unit_price:,.2f} EUR/MWh{', assumed' if default else ''}) minus the "
                f"monthly market value of onshore wind ({mv_txt}), if positive. ")
        head += ("Because the market value exceeded the reference value, no premium was due and the lost "
                 "revenue equals the market revenue: " if v.premium_eur < 0.5 else
                 f"This includes EUR {f(v.premium_eur)} of market premium. The lost revenue is ")
        head += f"{rng}."
    else:
        head = (f"Valued at a fixed PPA price of {v.unit_price:,.2f} EUR/MWh"
                f"{' (market benchmark; the contract price should be confirmed)' if default else ''}, "
                f"the lost energy corresponds to {rng}.")
    parts = [head]
    if v.n_negative_intervals and v.mode != "ppa":
        rule = "§51 EEG" if v.mode == "eeg" else "the plant would have curtailed"
        parts.append(f"{v.n_negative_intervals} intervals with negative day-ahead prices "
                     f"({v.energy_negative_mwh:,.1f} MWh) are valued at zero ({rule}).")
    if v.is_provisional and v.mode != "ppa":
        until = f"{v.price_coverage_until:%d %b %Y %H:%M}" if v.price_coverage_until is not None else "n/a"
        parts.append(f"Day-ahead prices are published only up to {until}; {v.share_provisional:.0%} of the "
                     f"lost energy is therefore valued provisionally by repeating the last published day's "
                     f"price profile and will be updated once prices are published.")
    if v.eur_after_waiting is not None:
        parts.append(f"Beyond the waiting period: EUR {f(v.eur_after_waiting)} (P50).")
    return " ".join(parts)


def comparison_line(vals: dict, primary: str) -> str:
    """'For comparison: EEG market premium EUR x; PPA (fixed price) EUR y.'"""
    others = [f"{v.label} EUR {v.eur['p50']:,.0f}" for k, v in vals.items() if k != primary and v is not None]
    return ("For comparison, the other remuneration models give: " + "; ".join(others) + ".") if others else ""
