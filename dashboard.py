import os
import mlflow
import dagshub
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── Connect to DagsHub MLflow ─────────────────────────────────
os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
dagshub.init(
    repo_owner="rishimithan",
    repo_name="ml_agent",
    mlflow=True
)
mlflow.set_tracking_uri(mlflow.get_tracking_uri())

# ── Fetch MLflow Runs ─────────────────────────────────────────
@st.cache_data(ttl=30)    # refresh every 30 seconds
def fetch_runs():
    """Fetch all runs from MLflow experiment"""
    client = mlflow.tracking.MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name("ml_agent")
    if experiment is None:
        return pd.DataFrame()

    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=500,
    )

    if not runs:
        return pd.DataFrame()

    # Convert to DataFrame
    records = []
    for run in runs:
        records.append({
            "run_id"        : run.info.run_id,
            "timestamp"     : datetime.fromtimestamp(run.info.start_time / 1000),
            "question"      : run.data.params.get("question_preview", "N/A"),
            "question_length": int(run.data.params.get("question_length", 0)),
            "tool_used"     : run.data.params.get("tool_used", "unknown"),
            "iterations"    : int(run.data.params.get("iterations", 0)),
            "quality_score" : float(run.data.metrics.get("quality_score", 0)),
            "tool_latency"  : float(run.data.metrics.get("tool_latency", 0)),
            "total_latency" : float(run.data.metrics.get("total_latency", 0)),
            "answer_length" : int(run.data.metrics.get("answer_length", 0)),
        })

    return pd.DataFrame(records)

# ── Dashboard Layout ──────────────────────────────────────────
def main():
    st.set_page_config(
        page_title = "ML Agent Dashboard",
        page_icon  = "🤖",
        layout     = "wide",
    )

    st.title("🤖 ML Systems Agent — Evaluation Dashboard")
    st.caption("Real-time analytics from MLflow experiment tracking via DagsHub")

    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

    # Fetch data
    df = fetch_runs()

    if df.empty:
        st.warning("No runs found. Make some API requests first!")
        st.info("POST https://ml-agent-hr47.onrender.com/ask")
        return

    # ── Row 1 — Overview KPIs ─────────────────────────────────
    st.subheader("📊 Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label = "Total Queries",
            value = len(df),
        )
    with col2:
        avg_quality = round(df["quality_score"].mean(), 2)
        st.metric(
            label = "Avg Quality Score",
            value = avg_quality,
            delta = f"{avg_quality - 0.7:.2f} vs baseline",
        )
    with col3:
        avg_latency = round(df["total_latency"].mean(), 2)
        st.metric(
            label = "Avg Latency (s)",
            value = avg_latency,
        )
    with col4:
        retry_rate = round((df["iterations"] > 0).mean() * 100, 1)
        st.metric(
            label = "Retry Rate",
            value = f"{retry_rate}%",
        )
    with col5:
        success_rate = round((df["quality_score"] >= 0.7).mean() * 100, 1)
        st.metric(
            label = "Success Rate",
            value = f"{success_rate}%",
        )

    st.divider()

    # ── Row 2 — Quality and Latency Over Time ─────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Quality Score Over Time")
        fig = px.line(
            df.sort_values("timestamp"),
            x     = "timestamp",
            y     = "quality_score",
            title = "Answer Quality Trend",
            labels = {"quality_score": "Quality Score", "timestamp": "Time"},
            markers = True,
        )
        fig.add_hline(
            y           = 0.7,
            line_dash   = "dash",
            line_color  = "red",
            annotation_text = "Threshold (0.7)",
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⚡ Response Latency Over Time")
        fig = px.line(
            df.sort_values("timestamp"),
            x       = "timestamp",
            y       = "total_latency",
            title   = "Total Latency Trend",
            labels  = {"total_latency": "Latency (s)", "timestamp": "Time"},
            markers = True,
            color_discrete_sequence = ["orange"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 3 — Tool Usage and Quality Distribution ───────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔧 Tool Usage Distribution")
        tool_counts = df["tool_used"].value_counts().reset_index()
        tool_counts.columns = ["tool", "count"]
        fig = px.pie(
            tool_counts,
            values = "count",
            names  = "tool",
            title  = "Which Tool Gets Used Most",
            color_discrete_sequence = px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Quality Score Distribution")
        fig = px.histogram(
            df,
            x      = "quality_score",
            nbins  = 10,
            title  = "Distribution of Answer Quality",
            labels = {"quality_score": "Quality Score"},
            color_discrete_sequence = ["steelblue"],
        )
        fig.add_vline(
            x                = 0.7,
            line_dash        = "dash",
            line_color       = "red",
            annotation_text  = "Threshold",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 4 — Tool Performance Comparison ──────────────────
    st.subheader("🔍 Tool Performance Comparison")
    col1, col2 = st.columns(2)

    with col1:
        tool_quality = df.groupby("tool_used")["quality_score"].mean().reset_index()
        fig = px.bar(
            tool_quality,
            x     = "tool_used",
            y     = "quality_score",
            title = "Avg Quality Score by Tool",
            color = "tool_used",
            color_discrete_sequence = px.colors.qualitative.Set2,
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tool_latency = df.groupby("tool_used")["total_latency"].mean().reset_index()
        fig = px.bar(
            tool_latency,
            x     = "tool_used",
            y     = "total_latency",
            title = "Avg Latency by Tool",
            color = "tool_used",
            color_discrete_sequence = px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 5 — Latency Breakdown ────────────────────────────
    st.subheader("⏱ Latency Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        # Tool vs generation latency
        df["generation_latency"] = df["total_latency"] - df["tool_latency"]
        fig = go.Figure(data=[
            go.Bar(name="Tool Latency",       x=df["tool_used"], y=df["tool_latency"]),
            go.Bar(name="Generation Latency", x=df["tool_used"], y=df["generation_latency"]),
        ])
        fig.update_layout(
            barmode = "stack",
            title   = "Latency Breakdown: Tool vs Generation",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Latency buckets
        df["latency_bucket"] = pd.cut(
            df["total_latency"],
            bins   = [0, 1, 2, 3, float("inf")],
            labels = ["0-1s", "1-2s", "2-3s", "3s+"],
        )
        bucket_counts = df["latency_bucket"].value_counts().reset_index()
        fig = px.bar(
            bucket_counts,
            x     = "latency_bucket",
            y     = "count",
            title = "Requests by Latency Bucket",
            color = "latency_bucket",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 6 — Failure Analysis ──────────────────────────────
    st.subheader("⚠️ Failure Analysis")

    failures = df[df["quality_score"] < 0.7]
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Failed Queries", len(failures))
        if not failures.empty:
            st.dataframe(
                failures[["timestamp", "question", "tool_used", "quality_score"]],
                use_container_width=True,
            )

    with col2:
        if not failures.empty:
            fail_tools = failures["tool_used"].value_counts().reset_index()
            fig = px.bar(
                fail_tools,
                x     = "tool_used",
                y     = "count",
                title = "Failures by Tool",
                color_discrete_sequence = ["red"],
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Row 7 — Recent Queries Table ──────────────────────────
    st.subheader("📋 Recent Queries")
    recent = df.head(20)[
        ["timestamp", "question", "tool_used",
         "quality_score", "total_latency", "iterations", "answer_length"]
    ].copy()
    recent["quality_score"] = recent["quality_score"].round(2)
    recent["total_latency"] = recent["total_latency"].round(2)
    recent.columns = [
        "Time", "Question", "Tool",
        "Quality", "Latency(s)", "Retries", "Answer Length"
    ]
    st.dataframe(recent, use_container_width=True)


if __name__ == "__main__":
    main()