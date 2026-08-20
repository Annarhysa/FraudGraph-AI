import time

import numpy as np
import plotly.express as px
import streamlit as st
import torch

from ..ui import components as ui


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions."""
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.clip(exp_counts / max(len(expected), 1), 1e-6, None)
    act_pct = np.clip(act_counts / max(len(actual), 1), 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def render(store):
    st.title("Model Monitoring")
    st.caption("Health checks computed live from your active upload — no placeholder numbers below are invented.")

    df = store.active_df()
    if df is None or df.empty:
        st.info("No transactions in the active upload.")
        return

    n_predictions = len(df)
    high_risk_rate = df["risk_level"].isin(["HIGH", "CRITICAL"]).mean()
    missing = df[["amount", "merchant_id", "channel", "mcc"]].isna().mean() * 100

    # A single upload batch has no train/test split (that's a property of
    # the shared sample the model was trained on) — split the batch itself
    # in half by time as a stand-in "earlier vs. later" comparison so this
    # page has something real to show instead of always "Not available".
    ordered = df.sort_values("txn_ts")
    half = len(ordered) // 2
    can_compare = half >= 5  # need enough rows on each side for PSI to mean anything
    if can_compare:
        earlier, later = ordered.iloc[:half], ordered.iloc[half:]
        amount_psi = psi(earlier["amount"].values, later["amount"].values)
        velocity_psi = psi(earlier["velocity_1h"].values, later["velocity_1h"].values)
    else:
        amount_psi = velocity_psi = None

    def psi_status(val):
        if val is None:
            return "UNKNOWN"
        if val < 0.1:
            return "HEALTHY"
        if val < 0.25:
            return "WARNING"
        return "CRITICAL"

    amt_status, vel_status = psi_status(amount_psi), psi_status(velocity_psi)
    if amt_status == "CRITICAL" or vel_status == "CRITICAL":
        overall = "CRITICAL"
    elif amt_status == "WARNING" or vel_status == "WARNING":
        overall = "WARNING"
    elif amt_status == "UNKNOWN" and vel_status == "UNKNOWN":
        overall = "UNKNOWN"
    else:
        overall = "HEALTHY"

    status_color = {"HEALTHY": "#2E9E5B", "WARNING": "#D4A72C", "CRITICAL": "#DC2626", "UNKNOWN": "#888"}
    st.markdown(
        f"### Overall status: <span style='color:{status_color[overall]}'>{overall}</span>",
        unsafe_allow_html=True,
    )
    if can_compare:
        st.caption(
            "Heuristic: this upload's earlier half vs. later half, compared by Population "
            "Stability Index on amount and velocity. HEALTHY below 0.1, WARNING 0.1–0.25, "
            "CRITICAL above 0.25."
        )
    else:
        st.caption("Too few transactions in this upload for a meaningful drift comparison (need 10+).")

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total predictions", f"{n_predictions:,}")
    c2.metric("High-risk prediction rate", f"{high_risk_rate:.2%}")

    data, meta = store.active_graph()
    _, _, model, *_ = store.load_graph_and_model()  # shared pretrained model
    start = time.perf_counter()
    with torch.no_grad():
        model(data.x, data.edge_index)
    latency_ms = (time.perf_counter() - start) * 1000
    c3.metric("Full-batch inference latency", f"{latency_ms:.0f} ms", help=f"Single forward pass over all {data.num_nodes:,} graph nodes on CPU.")

    st.divider()
    ui.section_title("Prediction distribution")
    fig = px.histogram(df, x="gnn_score_pct", nbins=50)
    fig.update_layout(height=300, xaxis_title="Risk score", yaxis_title="Count")
    st.plotly_chart(fig, width="stretch")

    st.divider()
    ui.section_title("Feature drift (earlier vs. later half of this upload, PSI)")
    d1, d2 = st.columns(2)
    with d1:
        st.metric("Amount — PSI", f"{amount_psi:.3f}" if amount_psi is not None else "Not available",
                   help="PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift")
        ui.risk_badge_html(amt_status)
    with d2:
        st.metric("Velocity (1h) — PSI", f"{velocity_psi:.3f}" if velocity_psi is not None else "Not available")
        ui.risk_badge_html(vel_status)

    st.divider()
    ui.section_title("Missing values")
    st.dataframe(missing.round(2).rename("% missing").to_frame(), width="stretch")

    st.divider()
    ui.section_title("Not yet implemented")
    st.info(
        "Concept drift detection, per-feature KL divergence across all model inputs, and "
        "production request-level latency tracking are not implemented — this page reflects "
        "one-off batch scoring, not a live serving pipeline."
    )
