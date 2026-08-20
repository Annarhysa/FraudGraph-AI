import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .. import cases as case_store
from .. import metrics
from ..ui import components as ui


def render(store):
    st.header("Overview")

    df = store.active_df()
    has_labels = bool(df["has_fraud_labels"].iloc[0]) if len(df) else False

    n_txn = len(df)
    n_fraud = int(df["is_fraud"].sum()) if has_labels else 0
    fraud_rate = n_fraud / n_txn if (n_txn and has_labels) else 0
    active_alerts = int((df["status"] == "OPEN").sum())
    high_risk = int(df["risk_level"].isin(["HIGH", "CRITICAL"]).sum())
    confirmed = int((df["status"] == "CONFIRMED FRAUD").sum())
    open_cases = len([
        c for c in case_store.list_cases()
        if c["status"] not in ("DISMISSED", "CONFIRMED FRAUD")
        and c["transaction_id"] in set(df["transaction_id"])
    ])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{n_txn:,}")
    c2.metric("Active Alerts", f"{active_alerts:,}")
    c3.metric("High Risk", f"{high_risk:,}")
    c4.metric("Fraud Rate", f"{fraud_rate:.3%}" if has_labels else "No labels")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Fraud Transactions", f"{n_fraud:,}" if has_labels else "N/A")
    c6.metric("Confirmed Fraud", f"{confirmed:,}")
    c7.metric("Open Cases", f"{open_cases:,}")
    c8.metric("Critical", int((df["risk_level"] == "CRITICAL").sum()))

    if not has_labels:
        st.caption("This upload had no `Is Fraud?` column, so fraud-rate/labeled metrics show as N/A rather than a fabricated number.")

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        ui.section_title("Fraud Risk Over Time")
        daily = df.copy()
        daily["date"] = pd.to_datetime(daily["txn_ts"]).dt.to_period("D").dt.start_time
        agg = daily.groupby("date").agg(
            transactions=("transaction_id", "count"),
            fraud=("is_fraud", "sum"),
            alerts=("risk_level", lambda s: (s.isin(["HIGH", "CRITICAL"])).sum()),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg["date"], y=agg["transactions"], name="Transactions",
                                  mode="lines+markers", line=dict(width=1.5, color=ui.BRAND_ACCENT)))
        fig.add_trace(go.Scatter(x=agg["date"], y=agg["alerts"], name="High-risk alerts",
                                  mode="lines+markers", line=dict(width=1.5, color="#E07B39")))
        if has_labels:
            fig.add_trace(go.Bar(x=agg["date"], y=agg["fraud"], name="Confirmed fraud (label)",
                                  marker_color="#DC2626"))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, width="stretch")

    with right:
        ui.section_title("Risk Distribution")
        dist = df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
        fig = px.bar(
            x=dist.index, y=dist.values,
            color=dist.index,
            color_discrete_map=ui.RISK_COLORS,
        )
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                           xaxis_title=None, yaxis_title="Transactions")
        st.plotly_chart(fig, width="stretch")

    st.divider()

    ui.section_title("Model Comparison — Rule-based vs. GraphSAGE")
    if has_labels:
        comp = metrics.model_comparison(df, use_split=False)
        rows = []
        for name, m in comp.items():
            rows.append({
                "Model": name,
                "ROC-AUC": f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "Not available",
                "PR-AUC": f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "Not available",
                f"Precision@{m['k']}" if m["k"] else "Precision@K": f"{m['precision_at_k']:.3f}" if m["precision_at_k"] is not None else "Not available",
                "Lift": f"{m['lift']:.1f}x" if m["lift"] is not None else "Not available",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Computed on this upload's own fraud labels — none of this data was used to train the model.")
    else:
        st.info("No `Is Fraud?` labels in this upload, so there's no ground truth to score the models against.")

    st.divider()

    ui.section_title("Recent High-Risk Alerts")
    top = df[df["risk_level"].isin(["HIGH", "CRITICAL"])].sort_values("txn_ts", ascending=False).head(15)
    show = top[["transaction_id", "gnn_score_pct", "amount", "card_key", "merchant_id", "txn_ts", "status"]].rename(
        columns={"transaction_id": "Alert / Txn ID", "gnn_score_pct": "Risk Score", "amount": "Amount",
                 "card_key": "Card", "merchant_id": "Merchant", "txn_ts": "Timestamp", "status": "Status"}
    )
    if show.empty:
        st.caption("No HIGH/CRITICAL transactions in this upload.")
    else:
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption("Select a transaction on the Fraud Alerts page to open its investigation.")
