import plotly.graph_objects as go
import streamlit as st

from .. import metrics
from ..ui import components as ui


def render(store):
    st.title("Model Performance")
    st.caption(
        "Metrics below are computed on your active upload's own `Is Fraud?` labels, if present. "
        "None of this data was ever used to train the model — the model was trained once on the "
        "shared IBM TabFormer sample (see Settings/About)."
    )

    df = store.active_df()
    if df is None or df.empty:
        st.info("No transactions in the active upload.")
        return

    has_labels = bool(df["has_fraud_labels"].iloc[0])
    if not has_labels:
        st.info(
            "This upload has no `Is Fraud?` column, so there's no ground truth to evaluate "
            "against. Metrics below will show as \"Not available\" rather than a fabricated number."
        )

    comp = metrics.model_comparison(df, use_split=False)

    ui.section_title("Rule-based vs. GraphSAGE")
    cols = st.columns(2)
    for col, (name, m) in zip(cols, comp.items()):
        with col:
            st.subheader(name)
            k_label = f"Precision@{m['k']}" if m["k"] else "Precision@K"
            st.metric("ROC-AUC", f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "Not available")
            st.metric("PR-AUC", f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "Not available")
            st.metric(k_label, f"{m['precision_at_k']:.3f}" if m["precision_at_k"] is not None else "Not available")
            st.metric("Lift", f"{m['lift']:.1f}x" if m["lift"] is not None else "Not available")

    if has_labels:
        st.divider()
        ui.section_title("ROC Curve")
        fig = go.Figure()
        for name, col in [("Rule-based", "anomaly_score"), ("GraphSAGE", "gnn_score")]:
            c = metrics.curves(df, col, use_split=False)
            if c is None:
                continue
            fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], name=name, mode="lines"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=380)
        st.plotly_chart(fig, width="stretch")

        ui.section_title("Precision-Recall Curve")
        fig2 = go.Figure()
        for name, col in [("Rule-based", "anomaly_score"), ("GraphSAGE", "gnn_score")]:
            c = metrics.curves(df, col, use_split=False)
            if c is None:
                continue
            fig2.add_trace(go.Scatter(x=c["recall"], y=c["precision"], name=name, mode="lines"))
        fig2.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=380)
        st.plotly_chart(fig2, width="stretch")

    st.divider()
    ui.section_title("Model Information")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Model:** GraphSAGE (2-layer)")
        st.write("**Node types:** card, transaction, merchant")
        st.write("**Hidden dim:** 32")
        st.write("**Trained via:** temporal split (50/10/40) on the shared sample")
    with c2:
        st.write("**Version:** v1")
        st.write(f"**Evaluated on:** {comp['GraphSAGE']['n_test']:,} transactions from your upload")
        base = comp["GraphSAGE"].get("base_rate")
        st.write(f"**Fraud rate in this upload:** {base:.4%}" if base is not None else "**Fraud rate in this upload:** Not available")
        st.write("**Explainability:** GNNExplainer (on-demand, per-transaction)")
