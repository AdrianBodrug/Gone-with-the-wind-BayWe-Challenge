"""Loading & cleaning of turbine time series and power curves (README tasks A1-A2)."""
import glob
import os

import numpy as np
import pandas as pd

VARS = ["wind_speed", "wind_direction", "ambient_temperature", "air_pressure", "wind_speed_stddev"]
FREQ = "10min"


def _read_table(src):
    """Read .xlsx or .csv from a path or a file-like object (e.g. a Streamlit upload)."""
    name = getattr(src, "name", src) if not isinstance(src, str) else src
    if str(name).lower().endswith(".csv"):
        return pd.read_csv(src)
    return pd.read_excel(src)


def load_turbine(src):
    """One turbine file -> DataFrame indexed by timestamp, with all VARS columns present."""
    df = _read_table(src)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    for v in VARS:
        if v not in df.columns:
            df[v] = np.nan
    return df


def load_power_curve(src):
    """Power curve file -> DataFrame with columns wind_speed (m/s), power (kW), sorted by wind speed."""
    df = _read_table(src)
    return df[["wind_speed", "power"]].astype(float).sort_values("wind_speed").reset_index(drop=True)


def turbine_name(df, default="?"):
    if "turbine_name" in df.columns and df["turbine_name"].notna().any():
        return str(df["turbine_name"].dropna().iloc[0])
    return default


def load_folder(path):
    """Load every *_data and *_power_curve file in a folder.

    Returns (turbines, power_curves), both dicts keyed by turbine name (taken from the
    turbine_name column, not the file name).
    """
    turbines, curves = {}, {}
    for f in sorted(glob.glob(os.path.join(path, "*_data.*"))):
        if os.path.basename(f).startswith("~$"):
            continue
        df = load_turbine(f)
        turbines[turbine_name(df, os.path.basename(f).split("_")[0])] = df
    for f in sorted(glob.glob(os.path.join(path, "*_power_curve.*"))):
        if os.path.basename(f).startswith("~$"):
            continue
        curves[os.path.basename(f).split("_")[0]] = load_power_curve(f)
    return turbines, curves


def as_indexed(df):
    """Accept a frame with either a DatetimeIndex or a 'timestamp' column."""
    if "timestamp" in df.columns:
        df = df.assign(timestamp=pd.to_datetime(df["timestamp"]))
        df = df.drop_duplicates("timestamp").set_index("timestamp")
    df = df.sort_index()
    for v in VARS:
        if v not in df.columns:
            df[v] = np.nan
    return df


def align(frames, freq=FREQ):
    """Reindex all frames onto one full 10-min index; absent rows become NaN rows."""
    frames = {n: as_indexed(d) for n, d in frames.items()}
    start = min(d.index.min() for d in frames.values())
    end = max(d.index.max() for d in frames.values())
    idx = pd.date_range(start, end, freq=freq)
    return {n: d.reindex(idx) for n, d in frames.items()}


def clean(d, stuck_steps=6):
    """Set to NaN values that are:
    a) a 0.0 m/s average (logger error: every channel, incl. temperature, is 0),
    b) physically impossible,
    c) a stuck wind sensor (same value for >= stuck_steps steps),
    d) a constant air pressure (placeholder, O park).
    """
    d = d.copy()

    zero_row = d["wind_speed"] == 0
    d.loc[zero_row, VARS] = np.nan

    bad_ws = (d["wind_speed"] < 0) | (d["wind_speed"] > 40)
    d.loc[bad_ws, ["wind_speed", "wind_speed_stddev"]] = np.nan
    d.loc[(d["wind_direction"] < 0) | (d["wind_direction"] > 360), "wind_direction"] = np.nan

    ws = d["wind_speed"]
    run_len = ws.groupby((ws != ws.shift()).cumsum()).transform("size")
    d.loc[ws.notna() & (run_len >= stuck_steps), "wind_speed"] = np.nan

    if d["air_pressure"].nunique(dropna=True) <= 1:
        d["air_pressure"] = np.nan
    return d


def fill_short_gaps(s, max_gap=3):
    """Linear interpolation only inside gaps of <= max_gap steps (30 min). References only."""
    isna = s.isna()
    run_len = isna.groupby((isna != isna.shift()).cumsum()).transform("size")
    filled = s.interpolate(method="time", limit_area="inside")
    return s.where(~(isna & (run_len <= max_gap)), filled)


def fill_reference(d, max_gap=3):
    """Short-gap fill for a reference turbine; direction is filled via sin/cos (circular)."""
    d = d.copy()
    for v in ["wind_speed", "ambient_temperature", "air_pressure", "wind_speed_stddev"]:
        d[v] = fill_short_gaps(d[v], max_gap)
    rad = np.deg2rad(d["wind_direction"])
    sin_f = fill_short_gaps(np.sin(rad), max_gap)
    cos_f = fill_short_gaps(np.cos(rad), max_gap)
    d["wind_direction"] = np.rad2deg(np.arctan2(sin_f, cos_f)) % 360
    return d
