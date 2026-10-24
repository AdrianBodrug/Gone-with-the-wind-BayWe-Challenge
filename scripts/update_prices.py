"""Refresh data/prices/day_ahead_DE-LU.csv with the latest German day-ahead prices.

    python -m scripts.update_prices                 # 2024-01-01 .. tomorrow
    python -m scripts.update_prices --start 2025-01-01

The app never downloads anything itself - it only reads the CSV. Run this when you need
newer prices (day D is published around 13:00 on D-1). Source: Energy-Charts API
(Fraunhofer ISE), data Bundesnetzagentur | SMARD.de, CC BY 4.0.
"""
import argparse
import json
import ssl
import urllib.parse
import urllib.request

import pandas as pd

from src.pricing import PRICE_FILE

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2024-01-01")
ap.add_argument("--end", default=(pd.Timestamp.now() + pd.Timedelta("2D")).strftime("%Y-%m-%d"))
args = ap.parse_args()

try:
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    ctx = ssl.create_default_context()
url = "https://api.energy-charts.info/price?" + urllib.parse.urlencode(
    {"bzn": "DE-LU", "start": args.start, "end": args.end})
req = urllib.request.Request(url, headers={"User-Agent": "baywa-outage-loss-tool"})
payload = json.loads(urllib.request.urlopen(req, timeout=60, context=ctx).read())

s = pd.Series(payload["price"], index=pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)).dropna()
s = s[~s.index.duplicated()].sort_index()
df = pd.DataFrame({"timestamp_utc": s.index.strftime("%Y-%m-%dT%H:%M:%SZ"), "eur_per_mwh": s.to_numpy()})
PRICE_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(PRICE_FILE, index=False)
print(f"{len(df)} prices {df.timestamp_utc.iloc[0]} -> {df.timestamp_utc.iloc[-1]} written to {PRICE_FILE}")
