"""Netrikan offline demo. The sidebar navigator is the dataset chooser.

Run:  make demo   (needs `make demo-train z24-train zero-shot` once)
"""
import streamlit as st

st.set_page_config(page_title="Netrikan · attack forecasting", layout="wide")

nav = st.navigation({
    "Choose a dataset": [
        st.Page("views/dapt2020.py", title="DAPT2020 · pentest campaign", url_path="dapt2020", default=True),
        st.Page("views/zeekdata24.py", title="UWF-ZeekData24 · scripted campaign", url_path="zeekdata24"),
        st.Page("views/cic17.py", title="CIC-IDS2017 · scan, DDoS, Heartbleed", url_path="cic2017"),
        st.Page("views/ctu13.py", title="CTU-13 s4 · botnet", url_path="ctu13"),
    ],
    "Evidence": [st.Page("views/zero_shot.py", title="Zero-shot transfer across labs", url_path="zero-shot")],
})
nav.run()
