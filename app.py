"""Entry point for the web app (local and Streamlit Community Cloud).

    streamlit run app.py

Pages:
  WindGap        – the team's front end (templates/windgap.html), embedded as-is.
                   NOTE: it currently computes with its own built-in JavaScript model,
                   not src/model.py. Connecting it to the Python backend is the next step.
  Python model   – examples/streamlit_example.py, running src/model.py estimate_loss().
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="WindGap – outage energy loss", page_icon="🌬", layout="wide")


def windgap_page():
    html = (ROOT / "templates" / "windgap.html").read_text(encoding="utf-8")
    st.caption("WindGap front end · computes with its built-in browser model (not yet the Python backend)")
    components.html(html, height=1500, scrolling=True)


pages = [
    st.Page(windgap_page, title="WindGap", icon="🌬", url_path="windgap", default=True),
    st.Page(str(ROOT / "examples" / "streamlit_example.py"), title="Python model", icon="🐍",
            url_path="model"),
]
st.navigation(pages).run()
