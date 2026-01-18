import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="Silence Zone Early Warning System (SZ‑EWS)",
    layout="wide"
)

DATA_PATH = "data_processed/SZEWS_final.csv"

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("Data file not found")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    df["yyyymm_dt"] = pd.to_datetime(df["yyyymm_dt"], errors="coerce")

    num_cols = [
        "SZI",
        "suppression_ratio",
        "suppression_depth_pct",
        "baseline_total_ma6",
        "total_activity",
        "enrol_activity",
        "demo_activity",
        "bio_activity",
        "alert_flag",
        "silence_duration_months"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    text_cols = ["state", "district", "pin_code", "region_id"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["Month_Label"] = df["yyyymm_dt"].dt.strftime("%b %Y")

    def cat(szi):
        if pd.isna(szi):
            return "Unknown"
        if szi <= 0.30:
            return "Severe Silence"
        if szi <= 0.60:
            return "Moderate Silence"
        return "Normal"

    df["SZI_Category"] = df["SZI"].apply(cat)
    return df.dropna(subset=["SZI", "region_id"])

df = load_data()

# ================= INTELLIGENCE MODULES =================
def compute_trends(d):
    d = d.sort_values(["region_id", "yyyymm_dt"]).copy()
    d["SZI_prev"] = d.groupby("region_id")["SZI"].shift(1)
    d["SZI_delta"] = d["SZI"] - d["SZI_prev"]
    d["Pre_Silence_Warning"] = (d["SZI_delta"] < -0.08) & (d["SZI"] > 0.30)
    return d

def priority_engine(d):
    s = d.copy()

    s["dur_norm"] = s["silence_duration_months"] / s["silence_duration_months"].max()
    s["depth_norm"] = s["suppression_depth_pct"] / 100
    s["impact_norm"] = s["baseline_total_ma6"] / s["baseline_total_ma6"].max()
    s["risk_norm"] = 1 - s["SZI"]

    s["Priority_Score"] = (
        0.35 * s["risk_norm"]
        + 0.25 * s["dur_norm"]
        + 0.25 * s["depth_norm"]
        + 0.15 * s["impact_norm"]
    )

    return s.sort_values("Priority_Score", ascending=False)

def dynamic_recommendation(r):
    if r["silence_duration_months"] >= 4 and r["suppression_depth_pct"] >= 50:
        return "Immediate mobile enrollment + infra audit"
    if r["bio_activity"] < r["demo_activity"]:
        return "Biometric device refresh and operator training"
    if r["enrol_activity"] < r["baseline_total_ma6"] * 0.4:
        return "Awareness drive and outreach camps"
    return "Monitor and reassess next cycle"

# ================= FILTERS =================
st.sidebar.title("Filters")

states = ["All"] + sorted(df["state"].unique())
cats = ["All"] + sorted(df["SZI_Category"].unique())

sel_state = st.sidebar.selectbox("State", states)
sel_cat = st.sidebar.selectbox("SZI Category", cats)
search = st.sidebar.text_input("Search District / PIN")

df_f = df.copy()
if sel_state != "All":
    df_f = df_f[df_f["state"] == sel_state]
if sel_cat != "All":
    df_f = df_f[df_f["SZI_Category"] == sel_cat]
if search.strip():
    df_f = df_f[
        df_f["district"].str.contains(search, case=False, na=False) |
        df_f["pin_code"].str.contains(search, case=False, na=False)
    ]

# ================= KPIs =================
def show_kpis(d):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zones Monitored", len(d))
    c2.metric("Avg SZI", round(d["SZI"].mean(), 3))
    c3.metric("Severe Zones", int((d["SZI_Category"] == "Severe Silence").sum()))
    c4.metric("Active Alerts", int((d["alert_flag"] == 1).sum()))

# ================= NAVIGATION =================
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "National Overview",
        "Pre‑Silence Warning",
        "Priority Intelligence",
        "Trend Explorer",
        "Action Planner",
    ],
)

# ================= HOME =================
if page == "Home":
    st.title("Silence Zone Early Warning System (SZ‑EWS)")

    st.markdown("""
**SZ‑EWS identifies Aadhaar service degradation *before* failure becomes visible.**

Instead of tracking high activity, it focuses on:
- sustained suppression
- silence propagation
- intervention prioritisation
""")

    show_kpis(df_f)

# ================= NATIONAL OVERVIEW =================
elif page == "National Overview":
    st.title("National Silence Landscape")

    show_kpis(df_f)

    fig = px.pie(
        df_f["SZI_Category"].value_counts().reset_index(),
        names="index",
        values="SZI_Category",
        hole=0.55,
        color_discrete_map={
            "Severe Silence": "#c0392b",
            "Moderate Silence": "#f39c12",
            "Normal": "#27ae60",
        }
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Worst Zones")
    st.dataframe(
        df_f.sort_values("SZI").head(20)[
            ["state", "district", "pin_code", "SZI", "silence_duration_months"]
        ],
        width="stretch"
    )

# ================= PRE‑SILENCE =================
elif page == "Pre‑Silence Warning":
    st.title("Pre‑Silence Early Warning")

    t = compute_trends(df)
    warn = t[t["Pre_Silence_Warning"] == True]

    if len(warn) == 0:
        st.info("No imminent silence escalation detected.")
    else:
        st.dataframe(
            warn[
                ["state", "district", "pin_code", "Month_Label", "SZI", "SZI_delta"]
            ],
            width="stretch"
        )

# ================= PRIORITY =================
elif page == "Priority Intelligence":
    st.title("Intervention Priority Engine")

    scored = priority_engine(df_f)

    fig = px.bar(
        scored.head(15),
        x="Priority_Score",
        y="district",
        orientation="h",
        text="Priority_Score"
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        scored.head(25)[
            ["state", "district", "pin_code", "Priority_Score", "SZI"]
        ],
        width="stretch"
    )

# ================= TREND =================
elif page == "Trend Explorer":
    st.title("Silence Trend Explorer")

    region = st.selectbox("Select Region", sorted(df_f["region_id"].unique()))
    d = df[df["region_id"] == region].sort_values("yyyymm_dt")

    fig = px.line(
        d,
        x="yyyymm_dt",
        y=["total_activity", "baseline_total_ma6"],
        labels={"value": "Activity", "variable": "Metric"}
    )
    st.plotly_chart(fig, width="stretch")

# ================= ACTION =================
elif page == "Action Planner":
    st.title("Action Planner")

    scored = priority_engine(df_f).head(25).copy()
    scored["Recommended_Action"] = scored.apply(dynamic_recommendation, axis=1)

    st.dataframe(
        scored[
            ["state", "district", "pin_code", "SZI",
             "silence_duration_months", "Priority_Score", "Recommended_Action"]
        ],
        width="stretch"
    )
