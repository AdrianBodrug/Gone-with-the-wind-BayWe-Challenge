"""Outage detection from missing target data (README task A3)."""
import pandas as pd

from .data_io import align


def gap_table(s):
    """One row per consecutive run of NaN: start, end, steps, hours."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    g = s.index.to_series()[isna].groupby(run_id[isna]).agg(["first", "last", "size"])
    g.columns = ["start", "end", "steps"]
    g["hours"] = g["steps"] / 6
    return g.reset_index(drop=True)


def detect_outages(target, references=None, min_steps=6):
    """NaN runs of the target wind speed of at least min_steps (default 1 h), longest first.

    If references are given, runs where the references are missing as well are dropped
    (e.g. rows absent from every file): there is nothing to predict from there.
    Returns a list of (start, end) timestamps, both inclusive.
    """
    frames = align({"__target__": target, **(references or {})})
    ws = frames["__target__"]["wind_speed"]
    gaps = gap_table(ws)
    gaps = gaps[gaps["steps"] >= min_steps]
    if references:
        ref_ok = pd.concat([d["wind_speed"] for n, d in frames.items() if n != "__target__"],
                           axis=1).notna().any(axis=1)
        covered = [ref_ok[s:e].mean() for s, e in zip(gaps["start"], gaps["end"])]
        gaps = gaps[[c >= 0.5 for c in covered]]
    gaps = gaps.sort_values(["steps", "start"], ascending=[False, True])
    return list(zip(gaps["start"], gaps["end"]))
