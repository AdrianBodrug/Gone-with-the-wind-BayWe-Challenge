"""Insurance-claim endpoint for the WindGap page: model results -> email draft + PDF + CSV.

The page POSTs to /api/claim the same analysis request it sends to /api/analyze
(target, window, reference pool, optional data) plus the claim form:

    {target, window: {start, end}, reference, power_curves: {name: [[ws, kw], ...]},
     data: {...optional, as /api/analyze},
     claim: {wind_farm_name, insurer_name, policy_number, ..., valuation_mode,
             eeg_reference_value, ppa_price, waiting_period_h}}

The server reruns estimate_loss() (src/model.py) for every turbine of the target's park that
is out in the window, builds the claim package with src/report.py and returns

    {id, turbines, energy_mwh (park total), window, valuation (market / eeg / ppa), email, files: {report}}

Model runs are cached by (data, target, window, references) so re-generating after editing the
form text is instant; only the email/PDF are rebuilt.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from .assessment import _sha256, assess
from .report import build_claim_package
from .schema import ClaimInfo
from .webapi import _frames, _naive, park

ROOT = Path(__file__).resolve().parents[1]
CLAIM_FIELDS = {f.name for f in fields(ClaimInfo)}
NUMERIC = {"eeg_reference_value", "ppa_price", "price_eur_per_mwh", "waiting_period_h"}
_models: "OrderedDict[str, object]" = OrderedDict()
_CACHE = 6
_files: "OrderedDict[str, dict]" = OrderedDict()   # claim id -> {filename: bytes}; nothing kept on disk
_FILES_KEEP = 10


def _claim_info(raw: dict) -> ClaimInfo:
    kw = {}
    for k, v in (raw or {}).items():
        if k not in CLAIM_FIELDS or v is None or (isinstance(v, str) and not v.strip()):
            continue                                   # empty -> ClaimInfo default / placeholder
        if k in NUMERIC:
            v = float(str(v).replace(",", "."))
        kw[k] = v.strip() if isinstance(v, str) else v
    return ClaimInfo(**kw)


def _curves(payload, store) -> dict:
    curves = dict(store.curves)
    for n, pts in (payload.get("power_curves") or {}).items():
        curves[n] = pd.DataFrame(pts, columns=["wind_speed", "power"]).astype(float)
    if payload.get("power_curve"):
        curves[payload["target"]] = pd.DataFrame(payload["power_curve"], columns=["wind_speed", "power"]).astype(float)
    return curves


def _input_hashes(payload, store, frames) -> dict:
    """Fingerprints for the audit appendix: server files for turbines from data/, the
    series as sent by the page for anything uploaded."""
    data_dir = Path(getattr(store, "data_dir", ROOT / "data"))
    out = {}
    for f in sorted(data_dir.glob("*_data.*")) + sorted(data_dir.glob("*_power_curve.*")):
        if f.name.split("_")[0] in frames and not f.name.startswith("~$"):
            out[f.name] = _sha256(f)
    series = (payload.get("data") or {}).get("series", {})
    for n, v in series.items():
        if n not in store.turbines:
            out[f"{n} (uploaded via web page)"] = hashlib.sha256(json.dumps(v).encode()).hexdigest()
    return out


def _model_key(payload) -> str:
    keep = {k: payload.get(k) for k in ("target", "window", "reference", "data", "power_curves", "power_curve")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


def run_model(payload, store):
    """AssessmentResult for every turbine of the target's park that is out in the window."""
    key = _model_key(payload)
    if key in _models:
        _models.move_to_end(key)
        return _models[key]

    target = payload["target"]
    frames, shift, step = _frames(payload, store)
    if target not in frames:
        raise ValueError(f"Zielturbine {target} nicht in den Daten.")
    curves = _curves(payload, store)
    start = _naive(payload["window"]["start"]) + shift
    end_excl = _naive(payload["window"]["end"]) + shift
    window = pd.date_range(start, end_excl - step, freq=step)
    if len(window) < 2:
        raise ValueError("Ausfallfenster zu kurz.")

    refs = {n: d for n, d in frames.items() if park(n) != park(target)}
    if not refs:
        raise ValueError("Keine Referenzturbinen außerhalb des Zielparks gefunden.")
    # the claim covers every turbine of the park that has a power curve and is out in the window
    targets = {}
    for n, d in sorted(frames.items()):
        if park(n) != park(target) or n not in curves:
            continue
        missing = d["wind_speed"].reindex(window).isna().mean()
        if n == target or missing > 0.5:
            targets[n] = d
    pool = payload.get("reference", "auto")
    pool = None if pool in (None, "auto") else [r for r in pool if r in refs] or None

    result = assess(targets, refs, curves, (window[0], window[-1]), _input_hashes(payload, store, frames),
                    references_to_use=pool)
    _models[key] = result
    if len(_models) > _CACHE:
        _models.popitem(last=False)
    return result


def claim(payload, store) -> dict:
    result = run_model(payload, store)
    info = _claim_info(payload.get("claim"))
    cid = uuid.uuid4().hex[:12]
    stem = re.sub(r"[^A-Za-z0-9]+", "_", info.wind_farm_name or "claim").strip("_")[:40] or "claim"
    with tempfile.TemporaryDirectory() as tmp:          # build, read into memory, delete
        pkg = build_claim_package(result, info, tmp, stem=stem)
        report_name = Path(pkg["report"]).name
        _files[cid] = {report_name: Path(pkg["report"]).read_bytes()}
    if len(_files) > _FILES_KEEP:
        _files.popitem(last=False)

    vals = {}
    for k, v in (pkg.get("valuation_obj") or {}).items():
        if v is not None:
            vals[k] = {"label": v.label, "eur": {q: round(x) for q, x in v.eur.items()},
                       "eur_per_mwh": round(v.avg_value_eur_per_mwh, 2),
                       "unit_price": v.unit_price, "default_price": v.unit_price_source not in ("", "operator"),
                       "premium_eur_per_mwh": round(v.premium_eur / v.energy_mwh, 2) if (k == "eeg" and v.energy_mwh) else None,
                       "market_value": next((val for val, _ in v.market_values.values()), None) if k == "eeg" else None,
                       "provisional_share": round(v.share_provisional, 3),
                       "after_waiting_eur": None if v.eur_after_waiting is None else round(v.eur_after_waiting)}
    farm = result.farm_energy_kwh()
    rated = float(result.power_curve["power"].max())
    out = {
        "id": cid,
        "turbines": [t.turbine for t in result.turbines],
        "rated_mw": rated / 1000,
        "energy_mwh": {k: v / 1000 for k, v in farm.items()},
        "window": {"start": str(result.outage_start), "end_inclusive": str(result.outage_end),
                   "hours": result.duration_h},
        "valuation": vals,
        "email": {"subject": pkg["email_subject"], "body": pkg["email_body"]},
        # the PDF also travels inline, so the page works when a second request may hit another
        # server instance (Vercel); locally the page can still use the /api/claim/<id>/<file> link
        "files": {"report": f"/api/claim/{cid}/{report_name}", "report_name": report_name,
                  "report_b64": base64.b64encode(_files[cid][report_name]).decode()},
    }
    return _clean(out)


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


SAFE = re.compile(r"^[0-9a-f]{12}$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9_.\-]+$")
CTYPES = {".pdf": "application/pdf", ".zip": "application/zip", ".csv": "text/csv; charset=utf-8",
          ".txt": "text/plain; charset=utf-8"}


def claim_file(cid: str, name: str):
    """(bytes, content_type) for a report generated in this server session, or None."""
    if not SAFE.match(cid) or not SAFE_FILE.match(name):
        return None
    data = _files.get(cid, {}).get(name)
    if data is None:
        return None
    return data, CTYPES.get(Path(name).suffix, "application/octet-stream")
