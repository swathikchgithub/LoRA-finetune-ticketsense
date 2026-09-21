"""
dashboard.py -- Stage 5: unified observability dashboard

A single Streamlit view over everything the other four stages produce:
drift (PSI/KL), prediction and confidence distribution, data quality
issue rates, and the prioritized alert digest from alerting.py.
Deliberately lightweight -- this reads the CSV/JSON/SQLite files the
other scripts already write, it doesn't re-run any of them.

Run with: streamlit run observability/dashboard.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

OBS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OBS_DIR))
from alerting import analyze_drift, analyze_skew, analyze_data_quality, score_priority  # noqa: E402

st.set_page_config(page_title="TicketSense Observability", layout="wide")
st.title("TicketSense: Post-Deployment Observability")
st.caption("Monitoring the fine-tuned ITSM ticket classifier after deployment -- "
           "drift, training-serving skew, and data quality, with impact-based alert prioritization.")

# ---- Alert digest (loaded first -- this is the summary an on-call engineer wants immediately) ----
st.header("Prioritized Alert Digest")
incidents = []
incidents += analyze_drift(OBS_DIR / "drift_results_drift.csv")
incidents += analyze_skew(OBS_DIR / "skew_results" / "summary.json")
incidents += analyze_data_quality(OBS_DIR / "data_quality_results.csv")
incidents = [score_priority(i) for i in incidents]
priority_order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4}
incidents.sort(key=lambda i: priority_order.get(i.priority, 5))

if not incidents:
    st.info("No result files found yet. Run simulate_traffic.py (--scenario drift), "
            "offline_vs_serving_eval.py, and simulate_ingestion.py + data_quality_monitor.py "
            "first, then reload this dashboard.")
else:
    priority_colors = {"P1": "🔴", "P2": "🟠", "P3": "🟡", "P4": "🟢", "P5": "⚪"}
    digest_rows = [{
        "Priority": f"{priority_colors.get(i.priority, '')} {i.priority}",
        "Source": i.source_stage,
        "Alert": i.alert_type,
        "Window": f"day {i.start_day}-{i.end_day}",
        "Peak value": f"{i.peak_value:.3f}",
        "Impact confirmed": "Yes" if i.business_impact_confirmed else "No",
    } for i in incidents]
    st.dataframe(pd.DataFrame(digest_rows), use_container_width=True, hide_index=True)

    n_pageworthy = sum(1 for i in incidents if i.priority in ("P1", "P2"))
    st.caption(f"{len(incidents)} correlated incidents from raw threshold crossings -- "
               f"only {n_pageworthy} rated P1/P2 (page-worthy). The rest are logged, not escalated.")

    with st.expander("Why each priority was assigned"):
        for i in incidents:
            st.markdown(f"**[{i.priority}] {i.source_stage} -- {i.alert_type}**")
            st.write(i.impact_description)
            st.caption(i.reasoning)
            st.divider()

st.divider()

# ---- Stage 2: Drift ----
st.header("Data Drift (Stage 2)")
drift_csv = OBS_DIR / "drift_results_drift.csv"
if drift_csv.exists():
    df = pd.read_csv(drift_csv)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("PSI over time")
        st.line_chart(df.set_index("day")[["psi"]])
        st.caption("Dashed reference: 0.25 = significant-shift threshold (not drawn natively "
                   "by st.line_chart, see alert digest above for exact crossing days).")
    with col2:
        acc_col = "rolling_accuracy_as_known_on_this_day" if "rolling_accuracy_as_known_on_this_day" in df.columns else "rolling_accuracy"
        st.subheader("Rolling accuracy (label-delayed)")
        st.line_chart(df.set_index("day")[[acc_col]])
else:
    st.info("Run simulate_traffic.py --scenario drift and drift_detection.py to populate this section.")

st.divider()

# ---- Stage 3: Training-serving skew ----
st.header("Training-Serving Skew (Stage 3)")
skew_json = OBS_DIR / "skew_results" / "summary.json"
if skew_json.exists():
    import json
    with open(skew_json) as f:
        skew = json.load(f)
    c1, c2, c3 = st.columns(3)
    c1.metric("Offline (CI/CD) accuracy", f"{skew['offline']['accuracy']:.1%}")
    c2.metric("Serving accuracy", f"{skew['serving']['accuracy']:.1%}",
              delta=f"-{skew['gap']:.1%}", delta_color="inverse")
    c3.metric("Gap invisible to offline eval", f"{skew['gap']:.1%}")
    st.caption(f"Truncation severity tested: {skew.get('max_chars', '?')} characters.")
else:
    st.info("Run offline_vs_serving_eval.py to populate this section.")

st.divider()

# ---- Stage 4: Data quality ----
st.header("Data Quality (Stage 4)")
dq_csv = OBS_DIR / "data_quality_results.csv"
if dq_csv.exists():
    df = pd.read_csv(dq_csv)
    st.subheader("Issue rates over time")
    st.line_chart(df.set_index("day")[["null_text_rate", "silent_default_rate", "schema_violation_rate"]])
    st.subheader("Blocked rate vs. model accuracy on unblocked traffic")
    st.line_chart(df.set_index("day")[["blocked_rate", "model_accuracy_on_unblocked_traffic"]])
    st.caption("The core Stage 4 finding: blocked rate can rise substantially while model "
               "accuracy on the traffic that DOES reach it stays flat -- proving this is a "
               "data problem, not a model problem.")
else:
    st.info("Run simulate_ingestion.py and data_quality_monitor.py to populate this section.")

st.divider()

# ---- Stage 1: Raw prediction/confidence distribution (if telemetry exists) ----
st.header("Prediction & Confidence Distribution (Stage 1 telemetry)")
db_path = OBS_DIR / "telemetry.db"
if db_path.exists():
    conn = sqlite3.connect(db_path)
    try:
        tel = pd.read_sql("SELECT * FROM predictions", conn)
        if not tel.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Predicted category distribution")
                st.bar_chart(tel["predicted_category"].value_counts())
            with col2:
                st.subheader("Confidence distribution")
                st.bar_chart(pd.cut(tel["confidence"], bins=10).value_counts().sort_index())
        else:
            st.info("predictions table exists but is empty.")
    except Exception:
        st.info("No 'predictions' table found in telemetry.db (only ingestion_log present -- "
                "run simulate_traffic.py to populate prediction-level telemetry).")
    conn.close()
else:
    st.info("Run simulate_traffic.py to populate this section.")
