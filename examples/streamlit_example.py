"""Minimal example of how the GUI talks to the model: one call to estimate_loss().

    streamlit run examples/streamlit_example.py

Flow: load files -> pick target & references -> auto-detect outage (adjustable)
      -> estimate_loss() -> show energy, metrics, method, wind & energy charts, CSV download.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import detect_outages, estimate_loss, load_folder, load_power_curve, load_turbine  # noqa: E402
from src.data_io import turbine_name  # noqa: E402

st.set_page_config(page_title="Outage Energy Loss", layout="wide")
st.title("Wind farm outage energy loss")


# ---------------------------------------------------------------- 1. inputs
@st.cache_data
def load_sample():
    return load_folder(ROOT / "data")


@st.cache_data
def run_model(target, references, power_curve, outage, references_to_use, select_extras):
    return estimate_loss(target, references, power_curve, outage,
                         references_to_use=references_to_use, select_extras=select_extras)


with st.sidebar:
    st.header("Data")
    if st.toggle("Use sample data (data/)", value=True):
        turbines, curves = load_sample()
        curve = next(iter(curves.values()))
    else:
        files = st.file_uploader("Turbine time series", type=["xlsx", "csv"], accept_multiple_files=True)
        pc_file = st.file_uploader("Power curve", type=["xlsx", "csv"])
        if not files or not pc_file:
            st.info("Upload turbine files and a power curve to start.")
            st.stop()
        turbines = {}
        for f in files:
            df = load_turbine(f)
            turbines[turbine_name(df, f.name.split("_")[0])] = df
        curve = load_power_curve(pc_file)

    st.header("Setup")
    names = sorted(turbines)
    target_name = st.selectbox("Target turbine", names, index=next(
        (i for i, n in enumerate(names) if n.startswith("T")), 0))
    park = target_name[0]
    candidates = [n for n in names if n[0] != park]          # references must be from another park
    auto = st.checkbox("Pick references automatically", value=True)
    chosen = None if auto else st.multiselect("References", candidates, default=candidates)
    select_extras = st.checkbox("Allow extra features (direction, hour, turbulence, ...)", value=True)

target = turbines[target_name]
references = {n: turbines[n] for n in candidates}

# ---------------------------------------------------------------- 2. outage
outages = detect_outages(target, references)
if not outages:
    st.warning("No outage (>= 1 h of missing wind speed) found in the target.")
    st.stop()
with st.sidebar:
    st.header("Outage")
    labels = [f"{s:%d %b %H:%M} - {e:%d %b %H:%M} ({(e - s).total_seconds() / 3600 + 1 / 6:.1f} h)"
              for s, e in outages]
    s, e = outages[labels.index(st.selectbox("Detected outages", labels))]
    c1, c2 = st.columns(2)
    start = pd.Timestamp.combine(c1.date_input("Start", s.date()), c1.time_input("Start time", s.time()))
    end = pd.Timestamp.combine(c2.date_input("End", e.date()), c2.time_input("End time", e.time()))

if not chosen and not auto:
    st.warning("Select at least one reference.")
    st.stop()

# ---------------------------------------------------------------- 3. the one model call
with st.spinner("Fitting model..."):
    try:
        r = run_model(target, references, curve, (start, end), chosen, select_extras)
    except ValueError as err:
        st.error(str(err))
        st.stop()

# ---------------------------------------------------------------- 4. results
e = r.energy_kwh
m = r.metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lost energy (P50)", f"{e['p50'] / 1000:,.1f} MWh",
          help=f"P10-P90: {e['p10'] / 1000:,.1f} - {e['p90'] / 1000:,.1f} MWh")
c2.metric("P10 - P90", f"{e['p10'] / 1000:,.0f} - {e['p90'] / 1000:,.0f} MWh")
c3.metric("Wind RMSE (CV)", f"{m['ws_rmse']:.2f} m/s", help=f"MAE {m['ws_mae']:.2f} m/s, R² {m['r2']:.2f}")
c4.metric("Energy deviation (CV)", f"{m['energy_dev_pct']:+.1f} %",
          help=f"Power RMSE {m['p_rmse']:.0f} kW")

st.subheader("Method")
st.write(r.method["description"])

h = r.history
pad = pd.Timedelta(days=3)
view = h[(h.index >= start - pad) & (h.index <= end + pad)]
p = r.predictions

st.subheader("Wind speed")
fig = go.Figure()
fig.add_vrect(x0=start, x1=end, fillcolor="grey", opacity=0.15, line_width=0, annotation_text="outage")
fig.add_trace(go.Scatter(x=p["timestamp"], y=p["ws_p90"], line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=p["timestamp"], y=p["ws_p10"], line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(31,119,180,0.2)", name="P10-P90"))
fig.add_trace(go.Scatter(x=view.index, y=view["ws_actual"], name="measured", line=dict(color="black", width=1)))
fig.add_trace(go.Scatter(x=view.index, y=view["ws_fit"], name="model", line=dict(color="#1f77b4", width=1.5)))
fig.update_layout(yaxis_title="m/s", height=380, margin=dict(t=20, b=20))
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Cumulative lost energy")
    cum = p["power_expected"].fillna(0).cumsum() / 6 / 1000
    fig = go.Figure(go.Scatter(x=p["timestamp"], y=cum, fill="tozeroy"))
    fig.update_layout(yaxis_title="MWh", height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")
with c2:
    st.subheader("Target vs. references (training data)")
    sc = h[h["ws_actual"].notna()].sample(3000, random_state=0) if h["ws_actual"].notna().sum() > 3000 else h
    fig = go.Figure(go.Scattergl(x=sc["ref_mean"], y=sc["ws_actual"], mode="markers",
                                 marker=dict(size=3, opacity=0.4)))
    fig.update_layout(xaxis_title="reference mean (m/s)", yaxis_title="target (m/s)",
                      height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

with st.expander("Model details"):
    md = r.method
    st.write("References:", ", ".join(md["references"]), "| extra features:",
             ", ".join(md["extra_features"]) or "none")
    st.dataframe(pd.DataFrame({"lag (min)": md["lags_min"], "correlation": md["correlations"]}))
    st.dataframe(pd.Series(md["coefficients"], name="coefficient").to_frame())
    if md["reference_selection"]:
        st.write("Reference selection (backward elimination)")
        st.dataframe(pd.DataFrame(md["reference_selection"]))
    if md["feature_selection"]:
        st.write("Feature selection (forward)")
        st.dataframe(pd.DataFrame(md["feature_selection"]))

st.download_button("Download predictions (CSV)", p.to_csv(index=False),
                   file_name=f"{target_name}_outage_predictions.csv", mime="text/csv")
