"""Claim deliverables: copy-paste email draft, PDF evidence report, data export.

Usage
-----
    from src.report import build_claim_package
    files = build_claim_package(result, claim_info, out_dir="outputs")

Everything here reads only an `AssessmentResult` + `ClaimInfo` (see schema.py),
so it works with any model that honours that contract.
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fpdf import FPDF  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

from .pricing import MODE_ALIASES, Valuation, comparison_line, describe, value_all  # noqa: E402
from .schema import AssessmentResult, ClaimInfo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

# palette
NAVY = (16, 42, 67)
TEAL = (0, 128, 128)
GREY = (90, 98, 110)
LIGHT = (241, 244, 247)
C_NAVY, C_TEAL, C_ORANGE, C_GREY = "#102a43", "#008080", "#d9822b", "#9aa5b1"
TURBINE_COLORS = ["#008080", "#d9822b", "#5e5ab8", "#2f8f4e", "#b84a62"]


# ================================================================ helpers
def _ph(value, label):
    """Value or a visible placeholder - never invent missing facts."""
    return value if value not in (None, "") else f"[{label}]"


def _mwh(kwh):
    return f"{kwh / 1000:,.1f}"


def _eur(x):
    return f"{x:,.0f}"


def _energy_after_waiting_period(result: AssessmentResult, hours: float) -> float:
    cutoff = result.outage_start + pd.Timedelta(hours=hours)
    return float(sum(t.predictions.loc[t.predictions.timestamp >= cutoff, "power_kw"].sum() / 6
                     for t in result.turbines))


def _fmt(x, f):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)


EXTRA_SHORT = {"turbulence": "turbulence", "temp_change_3h": "temp. change 3 h", "dir_sin": "dir. sin",
               "dir_cos": "dir. cos", "hour_sin": "hour sin", "hour_cos": "hour cos",
               "ref_x_dir_sin": "wind×dir sin", "ref_x_dir_cos": "wind×dir cos",
               "ref_x_hour_sin": "wind×hour sin", "ref_x_hour_cos": "wind×hour cos"}


def _extras_text(ex: dict, max_len: int = 62) -> str:
    """Compact extra-term list for the appendix table (direction harmonics grouped)."""
    if not ex:
        return "none"
    parts, n_h = [], sum(1 for k in ex if k.startswith("h_"))
    for k, v in ex.items():
        if not k.startswith("h_"):
            parts.append(f"{EXTRA_SHORT.get(k, k)} {v:+.2f}")
    if n_h:
        parts.append(f"direction curve ({n_h} terms)")
    txt = ", ".join(parts)
    return txt if len(txt) <= max_len else txt[: max_len - 1] + "…"


def _mean_metric(result, key):
    return float(np.mean([t.metrics[key] for t in result.turbines]))


def primary_mode(info: ClaimInfo) -> str:
    return MODE_ALIASES.get(info.valuation_mode or "market", info.valuation_mode or "market")


def compute_valuation(result: AssessmentResult, info: ClaimInfo, offline: bool = False) -> dict:
    """All three valuations {'market','eeg','ppa'} (one price download). Values may be None."""
    mode = primary_mode(info)
    eeg = info.eeg_reference_value or (info.price_eur_per_mwh if mode == "eeg" else None)
    ppa = info.ppa_price or (info.price_eur_per_mwh if mode == "ppa" else None)
    return value_all(result, eeg, ppa, info.waiting_period_h, offline=offline)


def build_context(result: AssessmentResult, info: ClaimInfo,
                  report_filename="energy_loss_report.pdf",
                  data_filename="outage_timeseries.csv",
                  valuation: dict | None = None) -> dict:
    farm = result.farm_energy_kwh()
    s, e = result.outage_start, result.outage_end + pd.Timedelta(minutes=10)
    tz_label = "local time, Europe/Berlin" if result.timezone == "Europe/Berlin" else result.timezone
    ctx = dict(
        farm=_ph(info.wind_farm_name, "wind farm name"),
        location=info.wind_farm_location,
        insurer=_ph(info.insurer_name, "insurer"),
        policy_number=_ph(info.policy_number, "policy number"),
        claim_reference=info.claim_reference,
        operator=_ph(info.operator_name, "operator / insured company"),
        contact_name=_ph(info.contact_name, "contact name"),
        contact_email=info.contact_email, contact_phone=info.contact_phone,
        turbine_model=info.turbine_model,
        cause=info.suspected_cause, repair_status=info.repair_status,
        start_date=s.strftime("%d.%m.%Y"), end_date=(e - pd.Timedelta(minutes=10)).strftime("%d.%m.%Y"),
        start_long=s.strftime("%d %B %Y at %H:%M"), end_long=e.strftime("%d %B %Y at %H:%M"),
        timezone_label=tz_label, duration_h=f"{result.duration_h:.1f}".rstrip("0").rstrip("."),
        turbine_list=", ".join(t.turbine for t in result.turbines),
        e_p10=_mwh(farm["p10"]), e_p50=_mwh(farm["p50"]), e_p90=_mwh(farm["p90"]),
        turbines=[dict(name=t.turbine, p10=_mwh(t.energy_kwh["p10"]), p50=_mwh(t.energy_kwh["p50"]),
                       p90=_mwh(t.energy_kwh["p90"])) for t in result.turbines],
        waiting_period_h=None, price=None, price_basis=info.price_basis,
        method_summary=result.method_summary,
        bt_ws_rmse=f"{_mean_metric(result, 'bt_ws_rmse'):.2f}",
        bt_energy_err=f"{_mean_metric(result, 'bt_abs_energy_err_pct'):.0f}",
        n_windows=int(min(t.metrics["n_backtest_windows"] for t in result.turbines)),
        report_filename=report_filename, data_filename=data_filename,
    )
    after_wp = None
    if info.waiting_period_h is not None:
        after_wp = _energy_after_waiting_period(result, info.waiting_period_h)
        ctx.update(waiting_period_h=f"{info.waiting_period_h:g}", e_after_wp=_mwh(after_wp))
    vals = valuation if valuation is not None else compute_valuation(result, info)
    mode = primary_mode(info)
    if vals.get(mode) is None:                      # e.g. no prices -> fall back to PPA
        mode = next((k for k in ("market", "eeg", "ppa") if vals.get(k) is not None), mode)
    v = vals.get(mode)
    ctx.update(valuation=v, valuations=vals, valuation_text=None,
               valuation_missing="day-ahead prices could not be retrieved; please value manually or retry when online")
    if v is not None:
        ctx.update(valuation_text=(describe(v) + " " + comparison_line(vals, mode)).strip(),
                   value_p10=_eur(v.eur["p10"]), value_p50=_eur(v.eur["p50"]), value_p90=_eur(v.eur["p90"]),
                   valuation_basis=v.basis[0].upper() + v.basis[1:] + (f" – {info.price_basis}" if info.price_basis else ""),
                   price=f"{info.price_eur_per_mwh:,.2f}" if info.price_eur_per_mwh else None)
        if v.eur_after_waiting is not None:
            ctx["value_after_wp"] = _eur(v.eur_after_waiting)
    return ctx


# ================================================================ email
def build_email(result: AssessmentResult, info: ClaimInfo, **kw) -> dict:
    """Return {'subject', 'body', 'full'} - plain text, ready to copy-paste."""
    env = Environment(loader=FileSystemLoader(TEMPLATES), keep_trailing_newline=True,
                      trim_blocks=False, lstrip_blocks=False)
    text = env.get_template("claim_email.txt.j2").render(**build_context(result, info, **kw))
    first, _, body = text.partition("\n")
    return {"subject": first.removeprefix("Subject: ").strip(), "body": body.strip() + "\n", "full": text}


# ================================================================ data export
def export_timeseries(result: AssessmentResult) -> pd.DataFrame:
    frames = []
    for t in result.turbines:
        d = t.predictions.copy()
        d.insert(0, "turbine", t.turbine)
        d["energy_kwh"] = d["power_kw"] / 6
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    return out.rename(columns={
        "ws_pred": "wind_speed_pred_ms", "ws_p10": "wind_speed_p10_ms", "ws_p90": "wind_speed_p90_ms",
        "power_kw": "expected_power_kw", "power_p10_kw": "power_at_ws_p10_kw",
        "power_p90_kw": "power_at_ws_p90_kw"}).round(4)


# ================================================================ charts
def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(C_GREY)
    ax.spines["bottom"].set_color(C_GREY)
    ax.tick_params(colors="#3e4c59", labelsize=8)
    ax.grid(axis="y", color="#e4e7eb", lw=0.8)
    ax.set_axisbelow(True)


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_wind(result: AssessmentResult) -> io.BytesIO:
    n = len(result.turbines)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.9 * n + 0.4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, t, col in zip(axes, result.turbines, TURBINE_COLORS):
        ax.axvspan(result.outage_start, result.outage_end, color="#fdecdc", lw=0)
        if t.context is not None:
            ax.plot(t.context.timestamp, t.context.ws_actual, color=C_NAVY, lw=0.9, label="Measured")
        p = t.predictions
        ax.fill_between(p.timestamp, p.ws_p10, p.ws_p90, color=col, alpha=0.18, lw=0, label="P10–P90")
        ax.plot(p.timestamp, p.ws_pred, color=col, lw=1.1, label="Reconstructed")
        ax.set_ylabel(f"{t.turbine}\nm/s", fontsize=8, color="#3e4c59")
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#fdecdc"))
    labels.append("Outage")
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    fig.tight_layout(h_pad=0.6, rect=(0, 0, 1, 0.97))
    return _png(fig)


def chart_energy(result: AssessmentResult) -> io.BytesIO:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    farm = result.farm_energy_kwh()
    total_ts = None
    for t, col in zip(result.turbines, TURBINE_COLORS):
        p = t.predictions
        a1.plot(p.timestamp, p.power_kw, color=col, lw=0.9, label=t.turbine)
        s = p.set_index("timestamp").power_kw / 6
        total_ts = s if total_ts is None else total_ts.add(s, fill_value=0)
    a1.set_ylabel("Expected power (kW)", fontsize=8)
    a1.legend(fontsize=7, frameon=False, ncol=len(result.turbines), loc="upper left")
    a1.xaxis.set_major_locator(mdates.DayLocator())
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _style(a1)
    cum = total_ts.cumsum() / 1000
    lo, hi = cum * farm["p10"] / farm["p50"], cum * farm["p90"] / farm["p50"]
    a2.fill_between(cum.index, lo, hi, color=C_TEAL, alpha=0.18, lw=0)
    a2.plot(cum.index, cum, color=C_TEAL, lw=1.4)
    a2.annotate(f"{cum.iloc[-1]:,.0f} MWh", (cum.index[-1], cum.iloc[-1]), fontsize=8, color=C_NAVY,
                ha="right", va="bottom", xytext=(0, 4), textcoords="offset points", weight="bold")
    a2.set_ylabel("Cumulative lost energy (MWh)", fontsize=8)
    a2.xaxis.set_major_locator(mdates.DayLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _style(a2)
    fig.tight_layout(w_pad=2)
    return _png(fig)


def chart_validation(result: AssessmentResult) -> io.BytesIO | None:
    tv = [t for t in result.turbines if t.validation is not None]
    if not tv:
        return None
    fig, axes = plt.subplots(1, len(tv), figsize=(8, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    hi = max(float(np.nanmax(t.validation[["ws_actual", "ws_pred"]].to_numpy())) for t in tv) * 1.05
    for ax, t, col in zip(axes, tv, TURBINE_COLORS):
        v = t.validation.dropna()
        ax.scatter(v.ws_actual, v.ws_pred, s=3, color=col, alpha=0.45, lw=0)
        ax.plot([0, hi], [0, hi], color=C_GREY, lw=0.8, ls="--")
        r = np.corrcoef(v.ws_actual, v.ws_pred)[0, 1]
        ax.set_title(f"{t.turbine}   r = {r:.2f}", fontsize=8, color=C_NAVY)
        ax.set_xlim(0, hi), ax.set_ylim(0, hi)
        ax.set_xlabel("Measured (m/s)", fontsize=8)
        _style(ax)
    axes[0].set_ylabel("Reconstructed (m/s)", fontsize=8)
    fig.tight_layout()
    return _png(fig)


# ================================================================ PDF
class _ReportPDF(FPDF):
    def __init__(self, title, subtitle):
        super().__init__(format="A4")
        self.doc_title, self.doc_sub = title, subtitle
        self.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
        self.add_font("DejaVuMono", "", str(FONT_DIR / "DejaVuSansMono.ttf"))
        self.set_auto_page_break(True, margin=18)
        self.set_margins(16, 16, 16)

    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 11, "F")
        self.set_xy(16, 3.2)
        self.set_font("DejaVu", "B", 8)
        self.set_text_color(255, 255, 255)
        self.cell(120, 5, self.doc_title)
        self.set_font("DejaVu", "", 8)
        self.cell(0, 5, self.doc_sub, align="R")
        self.set_y(18)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 7)
        self.set_text_color(*GREY)
        self.cell(0, 5, "Estimate based on reference wind data – see section 5 for limitations.", align="L")
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}", align="R")

    # building blocks
    def h1(self, txt):
        self.set_font("DejaVu", "B", 17)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 8, txt)
        self.ln(1)

    def h2(self, txt):
        self.ln(3)
        self.set_font("DejaVu", "B", 11.5)
        self.set_text_color(*NAVY)
        self.cell(0, 7, txt, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*TEAL)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.l_margin + 18, self.get_y())
        self.ln(2.5)

    def para(self, txt, size=9, color=(31, 41, 51)):
        self.set_font("DejaVu", "", size)
        self.set_text_color(*color)
        self.multi_cell(0, 4.6, txt, align="L")
        self.ln(1.5)

    def kv_table(self, rows, w1=52):
        self.set_font("DejaVu", "", 8.8)
        for i, (k, v) in enumerate(rows):
            if self.get_y() + 12 > self.page_break_trigger:   # keep key + value on one page
                self.add_page()
            self.set_fill_color(*(LIGHT if i % 2 == 0 else (255, 255, 255)))
            y = self.get_y()
            self.set_text_color(*GREY)
            self.multi_cell(w1, 6, k, fill=True, new_x="RIGHT", new_y="TOP", align="L")
            self.set_text_color(31, 41, 51)
            self.multi_cell(0, 6, str(v), fill=True, new_x="LMARGIN", new_y="NEXT", align="L")
            self.set_y(max(self.get_y(), y + 6))

    def grid_table(self, header, rows, widths, align=None, mono_cols=(), size=8.5):
        align = align or ["L"] + ["R"] * (len(header) - 1)
        self.set_font("DejaVu", "B", size)
        self.set_fill_color(*NAVY)
        self.set_text_color(255, 255, 255)
        for h, w, a in zip(header, widths, align):
            self.cell(w, 6.5, h, fill=True, align=a)
        self.ln()
        for i, r in enumerate(rows):
            self.set_fill_color(*(LIGHT if i % 2 == 0 else (255, 255, 255)))
            self.set_text_color(31, 41, 51)
            for j, (c, w, a) in enumerate(zip(r, widths, align)):
                self.set_font("DejaVuMono" if j in mono_cols else "DejaVu", "", size - (1.2 if j in mono_cols else 0))
                self.cell(w, 6, str(c), fill=True, align=a)
            self.ln()
        self.ln(2)

    def image_buf(self, buf, w=178):
        self.image(buf, x=self.l_margin, w=w)
        self.ln(2)

    def caption(self, txt):
        self.set_font("DejaVu", "", 7.5)
        self.set_text_color(*GREY)
        self.multi_cell(0, 3.8, txt, align="L")
        self.ln(2)


def chart_prices(v: Valuation) -> io.BytesIO:
    s = v.series
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax2 = ax.twinx()
    ax2.fill_between(s.timestamp, s.lost_mwh * 6, color=C_TEAL, alpha=0.18, lw=0, step="post")
    ax2.set_ylabel("Lost farm power (MW)", fontsize=8, color=C_TEAL)
    ax2.tick_params(colors=C_TEAL, labelsize=8)
    for sp in ("top",):
        ax2.spines[sp].set_visible(False)
    pub, prov = s[~s.provisional], s[s.provisional]
    ax.step(pub.timestamp, pub.price_eur_per_mwh, where="post", color=C_NAVY, lw=1.1, label="Day-ahead price")
    if len(prov):
        ax.step(prov.timestamp, prov.price_eur_per_mwh, where="post", color=C_ORANGE, lw=1.1, ls="--",
                label="Provisional (not yet published)")
    ax.axhline(0, color=C_GREY, lw=0.6)
    ax.set_ylabel("EUR/MWh", fontsize=8)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _style(ax)
    fig.tight_layout()
    return _png(fig)


def _valuation_section(pdf, v: Valuation, ctx):
    pdf.add_page()
    pdf.h2("4  Valuation")
    vals = ctx["valuations"]
    pdf.para("Which value applies depends on how the wind farm sells its electricity. All three common "
             "models are shown; the headline figure uses the " + v.label.lower() + " model.", size=9)
    rows = []
    for k in ("market", "eeg", "ppa"):
        x = vals.get(k)
        if x is None:
            rows.append([MODE_LABEL_SHORT[k], "prices unavailable", "", "", ""])
            continue
        if k == "market":
            basis = "day-ahead price per interval"
        elif k == "eeg":
            mv = ", ".join(f"{val:,.1f}" for val, _ in x.market_values.values())
            basis = f"ref. value {x.unit_price:,.2f} vs market value {mv}"
        else:
            basis = f"fixed {x.unit_price:,.2f} EUR/MWh"
        rows.append([MODE_LABEL_SHORT[k] + (" ★" if k == v.mode else ""), basis, _eur(x.eur["p50"]),
                     f"{_eur(x.eur['p10'])} – {_eur(x.eur['p90'])}", f"{x.avg_value_eur_per_mwh:,.2f}"])
    pdf.grid_table(["Model", "Price basis", "EUR P50", "EUR P10 – P90", "EUR/MWh"], rows,
                   [30, 66, 22, 38, 22], align=["L", "L", "R", "R", "R"], size=7.8)
    pdf.caption("★ = headline model. EUR/MWh = average value per lost MWh. EEG: under the market premium model the operator sells at the market and "
                "receives max(0, reference value − monthly market value of onshore wind) on top; no payment in "
                "negative-price intervals (§51 EEG). PPA: fixed pay-as-produced price.")
    pdf.para(ctx["valuation_text"], size=8.8)
    notes = []
    for k in ("eeg", "ppa"):
        x = vals.get(k)
        if x is not None and x.unit_price_source not in ("", "operator"):
            notes.append(f"{PRICE_NAME[k]} not provided by the operator – benchmark used: "
                         f"{x.unit_price:,.2f} EUR/MWh ({x.unit_price_source}).")
        if k == "eeg" and x is not None:
            for m, (val, src) in x.market_values.items():
                notes.append(f"Market value onshore wind {m}: {val:,.2f} EUR/MWh – {src}.")
    if notes:
        pdf.set_font("DejaVu", "B", 8.5)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 5, "Assumptions", new_x="LMARGIN", new_y="NEXT")
        pdf.para("\n".join("• " + n for n in notes), size=8)
    if v.avg_market_price is not None or (vals.get("market") is not None):
        m = vals.get("market") or v
        rows = [["Average day-ahead price in window", f"{m.avg_market_price:,.2f} EUR/MWh"],
                ["Negative-price intervals", f"{m.n_negative_intervals} ({m.energy_negative_mwh:,.1f} MWh)"],
                ["Provisionally priced share", f"{m.share_provisional:.0%}"],
                ["Price data", {"live": "downloaded live", "cache": "local cache (offline)",
                                "supplied": "supplied"}.get(m.price_source, m.price_source)
                 + (f"; published until {m.price_coverage_until:%d %b %Y %H:%M}" if m.price_coverage_until is not None else "")]]
        if v.eur_after_waiting is not None:
            rows.append([f"Headline value beyond waiting period", f"EUR {_eur(v.eur_after_waiting)}"])
        pdf.kv_table(rows, w1=62)
        pdf.ln(2)
        pdf.image_buf(chart_prices(m))
        pdf.caption("Figure 4 – Day-ahead price per interval (line) and lost farm power (shaded). Each 10-minute "
                    "slice of lost energy is valued at the price of the interval it falls in. " + m.attribution + ".")


MODE_LABEL_SHORT = {"market": "Market", "eeg": "EEG premium", "ppa": "PPA"}
PRICE_NAME = {"eeg": "EEG reference value", "ppa": "PPA price"}


def build_report_pdf(result: AssessmentResult, info: ClaimInfo, path: str | Path,
                     valuation: dict | None = None) -> Path:
    path = Path(path)
    ctx = build_context(result, info, valuation=valuation)
    farm = result.farm_energy_kwh()
    generated = datetime.now().strftime("%d %b %Y %H:%M")
    pdf = _ReportPDF("Energy Yield Loss Assessment", f"{ctx['farm']}  ·  generated {generated}")
    pdf.alias_nb_pages()

    # ------------------------------------------------ page 1: summary
    pdf.add_page()
    pdf.h1("Energy yield loss during communication outage")
    pdf.para(f"{ctx['farm']}  ·  {ctx['start_date']} – {ctx['end_date']}  ·  turbines {ctx['turbine_list']}",
             size=10, color=GREY)

    # headline box
    y0 = pdf.get_y() + 2
    pdf.set_fill_color(*LIGHT)
    pdf.rect(pdf.l_margin, y0, 178, 34, "F")
    pdf.set_fill_color(*TEAL)
    pdf.rect(pdf.l_margin, y0, 1.6, 34, "F")
    pdf.set_xy(pdf.l_margin + 7, y0 + 4)
    pdf.set_font("DejaVu", "", 8.5)
    pdf.set_text_color(*GREY)
    pdf.cell(80, 5, "ESTIMATED LOST ENERGY (P50)")
    pdf.cell(0, 5, "LIKELY RANGE (P10 – P90)")
    pdf.set_xy(pdf.l_margin + 7, y0 + 10)
    pdf.set_font("DejaVu", "B", 26)
    pdf.set_text_color(*NAVY)
    pdf.cell(80, 12, f"{ctx['e_p50']} MWh")
    pdf.set_font("DejaVu", "B", 15)
    pdf.set_text_color(*TEAL)
    pdf.cell(0, 12, f"{ctx['e_p10']} – {ctx['e_p90']} MWh")
    pdf.set_xy(pdf.l_margin + 7, y0 + 24)
    pdf.set_font("DejaVu", "", 8.5)
    pdf.set_text_color(31, 41, 51)
    line = f"Over {ctx['duration_h']} h for {len(result.turbines)} turbines."
    v = ctx["valuation"]
    if v is not None:
        line += (f"  Value: EUR {ctx['value_p50']} (EUR {ctx['value_p10']} – {ctx['value_p90']}), "
                 + {"market": "day-ahead market", "eeg": "EEG market premium", "ppa": "PPA price"}[v.mode]
                 + (" – partly provisional" if v.is_provisional else "") + ".")
    else:
        line += "  Monetary value: choose a valuation basis to calculate."
    pdf.multi_cell(165, 4.5, line)
    pdf.set_y(y0 + 38)

    pdf.h2("1  Outage details")
    rows = [
        ("Wind farm", ctx["farm"] + (f", {ctx['location']}" if ctx["location"] else "")),
        ("Operator / insured", ctx["operator"]),
        ("Insurer · policy", f"{ctx['insurer']} · {ctx['policy_number']}"),
        ("Outage start", f"{ctx['start_long']} ({ctx['timezone_label']})"),
        ("Outage end", f"{ctx['end_long']} ({ctx['timezone_label']})"),
        ("Duration", f"{ctx['duration_h']} hours"),
        ("Affected turbines", ctx["turbine_list"] + (f" ({ctx['turbine_model']})" if ctx["turbine_model"] else "")),
        ("Suspected cause", ctx["cause"]),
    ]
    if ctx["repair_status"]:
        rows.append(("Status", ctx["repair_status"]))
    if v is not None:
        rows.append(("Valuation basis", ctx["valuation_basis"]))
    if ctx["waiting_period_h"]:
        rows.append(("Waiting period", f"{ctx['waiting_period_h']} h → {ctx['e_after_wp']} MWh (P50) beyond the waiting period"
                     + (f", EUR {ctx['value_after_wp']}" if ctx.get("value_after_wp") else "")))
    pdf.kv_table(rows)

    pdf.h2("2  Lost energy per turbine")
    trs = [[t.turbine, _mwh(t.energy_kwh["p10"]), _mwh(t.energy_kwh["p50"]), _mwh(t.energy_kwh["p90"]),
            f"{t.predictions.ws_pred.mean():.1f}", f"{t.predictions.power_kw.mean() / 1000:.2f}"]
           for t in result.turbines]
    trs.append(["Wind farm", ctx["e_p10"], ctx["e_p50"], ctx["e_p90"], "", ""])
    pdf.grid_table(["Turbine", "P10 MWh", "P50 MWh", "P90 MWh", "Avg wind m/s", "Avg power MW"],
                   trs, [34, 26, 26, 26, 33, 33])
    pdf.caption("P50 is the best estimate; actual lost energy is expected to fall between P10 and P90 in "
                "8 out of 10 cases, based on the method's historical errors. Farm P10/P90 are summed per turbine "
                "(conservative, since all turbines share the same reference data). Power curve values are "
                "interpreted as kW; energy = expected power × 10 min.")

    # ------------------------------------------------ page 2: charts
    pdf.add_page()
    pdf.h2("3  Reconstructed wind and lost production")
    pdf.image_buf(chart_wind(result))
    pdf.caption("Figure 1 – Measured wind speed at each turbine (navy) one day before and after the outage, and "
                "the reconstructed wind speed during the outage (shaded) with its P10–P90 band.")
    pdf.image_buf(chart_energy(result))
    pdf.caption("Figure 2 – Left: expected power per turbine during the outage. Right: cumulative lost energy for "
                "the wind farm with P10–P90 band.")

    # ------------------------------------------------ valuation + method & accuracy
    if v is not None:
        _valuation_section(pdf, v, ctx)

    pdf.add_page()
    pdf.h2("5  Method, accuracy and limitations")
    pdf.para(result.method_summary)
    pdf.para("Why this method: every number can be checked by hand. Each reference has one weight and one "
             "time shift (Appendix A); non-negative weights mean stronger wind at a reference can never lower "
             "the estimate. Inputs are selected automatically, but only kept if they measurably improve "
             "accuracy on held-out history, so the model stays as simple as the data allows.")
    acc = [[t.turbine, f"{t.metrics['bt_ws_rmse']:.2f}", f"{t.metrics['bt_ws_mae']:.2f}",
            f"{t.metrics['bt_p_rmse']:,.0f}", f"{t.metrics['bt_abs_energy_err_pct']:.1f}",
            _fmt(t.metrics.get('val_ws_rmse'), "{:.2f}"), _fmt(t.metrics.get('val_energy_dev_pct'), "{:+.1f}")]
           for t in result.turbines]
    pdf.grid_table(["Turbine", "Wind RMSE", "Wind MAE", "Power RMSE", "|Energy err.| %",
                    "Val. wind RMSE", "Val. energy %"], acc, [22, 24, 22, 26, 28, 28, 28], size=8)
    n_win = ctx["n_windows"]
    pdf.caption(f"First four columns: cross-validation over {n_win} historical 3-day blocks (model refitted "
                "without each block, then scored on it). 'Val.' columns: the 4–9 Sep validation week, held out "
                "entirely ('–' = not available for this data). Wind in m/s, power in kW, energy error = deviation of summed energy.")
    vb = chart_validation(result)
    if vb:
        pdf.image_buf(vb)
        pdf.caption("Figure 3 – Validation week: reconstructed vs measured 10-minute wind speed. Points on the "
                    "dashed line would be perfect predictions.")
    pdf.set_font("DejaVu", "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 6, "Limitations", new_x="LMARGIN", new_y="NEXT")
    pdf.para("• The figure is an estimate. Individual 10-minute values are uncertain; totals over several days "
             "are considerably more reliable.\n"
             "• It assumes the turbines were available and would have operated normally; curtailment, "
             "maintenance or grid limits during the outage are not included.\n"
             "• Loss of communication does not by itself prove a production stop. The grid operator's metering "
             "data for the period should be used to confirm that no energy was fed in.\n"
             "• Monetary value, waiting period and policy terms depend on contracts provided by the operator. "
             "Market valuation uses day-ahead prices; actual revenue may differ (balancing, marketing fees).",
             size=8.5)

    # ------------------------------------------------ page 4: audit appendix
    pdf.add_page()
    pdf.h2("Appendix A  Reproducibility and audit trail")
    pdf.para("Re-running the tool with the input files below (verified by their SHA-256 fingerprints) and the "
             "same model version reproduces every figure in this report.", size=8.5)
    pc = result.power_curve
    rated = float(pc.power.max())
    rows = [
        ("Model", f"{result.model_name} ({result.model_version})"),
        ("Reference turbines", ", ".join(result.references)),
        ("Outage window", f"{result.outage_start:%Y-%m-%d %H:%M} – {result.outage_end:%Y-%m-%d %H:%M} "
                          f"({result.timezone}, 10-min stamps)"),
        ("Power curve", f"rated {rated:,.0f} kW from {pc.loc[pc.power >= rated, 'wind_speed'].min():g} m/s; "
                        f"first output at {pc.loc[pc.power > 0, 'wind_speed'].min():g} m/s"),
        ("Report generated", generated),
    ]
    pdf.kv_table(rows, w1=42)
    pdf.ln(2)
    refs = list(result.references)
    wrows = []
    for t in result.turbines:
        w, lag = t.method.get("weights", {}), t.method.get("lags_min", {})
        wrows.append([t.turbine] + [f"{w[r]:.2f} ({lag[r]:+d})" if r in w else "–" for r in refs])
    cw = min(26, int(160 / max(len(refs), 1)))
    pdf.grid_table(["Turbine"] + refs, wrows, [18] + [cw] * len(refs), size=8)
    pdf.caption("Weight of each reference turbine's wind speed, with its time shift in minutes in brackets "
                "(negative = the wind reaches the target turbine first). Example: 0.40 means +1 m/s at that "
                "reference adds 0.40 m/s at the target turbine. '–' = reference not used for this turbine.")
    xrows = []
    for t in result.turbines:
        ex = t.method.get("extra_coefficients", {})
        xrows.append([t.turbine, _extras_text(ex),
                      f"{t.method.get('intercept', float('nan')):.2f}", f"{t.metrics.get('r2', float('nan')):.2f}",
                      f"{t.method.get('n_train_rows', '')}"])
    pdf.grid_table(["Turbine", "Extra terms (coefficient)", "Intercept m/s", "CV R²", "Train rows"],
                   xrows, [18, 92, 26, 18, 24], align=["L", "L", "R", "R", "R"], size=7.6)
    frows = [[name, h] for name, h in sorted(result.input_files.items())]
    pdf.grid_table(["Input file", "SHA-256 fingerprint"], frows, [44, 134], align=["L", "L"], mono_cols=(1,), size=7.8)

    pdf.output(str(path))
    return path


# ================================================================ package
def build_claim_package(result: AssessmentResult, info: ClaimInfo, out_dir: str | Path = "outputs",
                        stem: str = "claim") -> dict:
    """Write email (.txt), report (.pdf), data (.csv) and a zip of all three."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_name, data_name = f"{stem}_energy_loss_report.pdf", f"{stem}_outage_timeseries.csv"
    valuation = compute_valuation(result, info)          # one price download per package
    email = build_email(result, info, report_filename=report_name, data_filename=data_name,
                        valuation=valuation)
    files = {
        "email": out / f"{stem}_email.txt",
        "report": build_report_pdf(result, info, out / report_name, valuation=valuation),
        "data": out / data_name,
    }
    files["email"].write_text(email["full"], encoding="utf-8")
    export_timeseries(result).to_csv(files["data"], index=False)
    parts = ["email", "report", "data"]
    avail = {k: x for k, x in valuation.items() if x is not None}
    if avail:
        base = next(iter(avail.values())).series[["timestamp", "lost_mwh", "price_eur_per_mwh", "provisional"]].copy()
        for k, x in avail.items():
            base[f"value_eur_{k}"] = x.series["value_eur"].to_numpy()
        files["valuation"] = out / f"{stem}_valuation.csv"
        base.round(4).to_csv(files["valuation"], index=False)
        parts.append("valuation")
    files["zip"] = out / f"{stem}_package.zip"
    with zipfile.ZipFile(files["zip"], "w", zipfile.ZIP_DEFLATED) as z:
        for k in parts:
            z.write(files[k], files[k].name)
    return {**files, "email_subject": email["subject"], "email_body": email["body"], "valuation_obj": valuation}
