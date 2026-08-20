import streamlit as st

from .. import batches as batch_store
from .. import cases as case_store
from .. import explainability
from .. import netviz
from ..ui import components as ui


def render(store):
    st.title("Investigation")

    df = store.active_df()
    if df is None or df.empty:
        st.info("No transactions in the active upload.")
        return

    default_txn = st.session_state.get("investigation_txn_id")
    options = df.sort_values("gnn_score_pct", ascending=False)["transaction_id"].tolist()
    index = options.index(default_txn) if default_txn in options else 0
    txn_id = st.selectbox("Transaction", options, index=index)
    st.session_state["investigation_txn_id"] = txn_id

    row = df[df["transaction_id"] == txn_id].iloc[0]
    row_idx = batch_store.parse_row_index(txn_id)

    st.divider()

    # --- Transaction overview ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", f"{row['gnn_score_pct']:.1f}%")
    with c2:
        st.write("**Risk Level**")
        ui.risk_badge_html(row["risk_level"])
    c3.metric("Amount", f"${row['amount']:.2f}")
    c4.metric("Status", row["status"])

    c5, c6, c7 = st.columns(3)
    c5.write(f"**Card:** {row['card_key']}")
    c6.write(f"**User:** U{row['user_id']}")
    c7.write(f"**Merchant:** M{row['merchant_id']}")
    st.write(f"**Timestamp:** {row['txn_ts']}  ·  **Channel:** {row['channel']}")
    if row["is_fraud"]:
        st.warning("Ground truth: this transaction IS labeled fraud in the source data (for evaluation only — not used as a model feature).")

    st.divider()

    # --- Risk assessment ---
    ui.section_title("Risk Assessment")
    ui.progress_bar_text("GraphSAGE score", row["gnn_score_pct"])
    ui.progress_bar_text("Rule-based score", row["rule_score_pct"])
    st.caption(
        f"GraphSAGE score is a calibrated model probability (raw: {row['gnn_score']:.4f}) "
        "rescaled 0-100 for display — see Settings/About for methodology. Scores are kept "
        "separate; no arbitrary combined score is computed."
    )

    st.divider()

    # --- Why flagged: three explicitly separated evidence categories ---
    ui.section_title("Why was this flagged?")
    tab1, tab2, tab3 = st.tabs(["Rule signals", "Network signals", "Model signals (GNNExplainer)"])

    with tab1:
        for r in explainability.rule_signals(row):
            st.markdown(f"- {r}")

    with tab2:
        for n in explainability.network_signals(row):
            st.markdown(f"- {n}")
        st.caption("Computed within your active upload only.")

    with tab3:
        st.caption("At-a-glance signal summary:")
        for name, level in explainability.top_signal_summary(row):
            badge_color = {"HIGH": "#DC2626", "MEDIUM": "#D4A72C", "LOW": "#64748B"}[level]
            st.markdown(
                f"{name} &nbsp;&nbsp; <span class='fg-badge' style='background:{badge_color}'>{level}</span>",
                unsafe_allow_html=True,
            )
        st.write("")
        if st.button("Run GNNExplainer on this transaction"):
            with st.spinner("Explaining prediction (GNNExplainer on 2-hop subgraph)..."):
                try:
                    explainer = store.active_model_explainer()
                    result = explainer.explain_transaction(row_idx)
                    st.session_state[f"explain_{txn_id}"] = result
                except Exception as e:
                    st.error(f"GNNExplainer failed: {e}")

        result = st.session_state.get(f"explain_{txn_id}")
        if result:
            st.write(f"Explained over a {result['subgraph_nodes']}-node local subgraph.")
            st.write("**Top contributing features**")
            for f in result["top_features"]:
                st.markdown(f"- {f['feature']}: importance {f['importance']:.3f}")
            st.write("**Top contributing edges (suspicious paths)**")
            if result["top_edges"]:
                for e in result["top_edges"]:
                    st.markdown(f"- {e['src']} → {e['dst']} (importance {e['importance']:.3f})")
            else:
                st.caption("No edges with positive importance in this subgraph.")

    st.divider()

    # --- Transaction history ---
    ui.section_title("Related Transaction History (within this upload)")
    hcard, huser, hmerchant = st.tabs(["Same card", "Same user", "Same merchant"])

    def history_table(sub_df):
        sub_df = sub_df.sort_values("txn_ts", ascending=False).head(20)
        return sub_df[["transaction_id", "txn_ts", "amount", "merchant_id", "gnn_score_pct", "status"]].rename(
            columns={"transaction_id": "Txn ID", "txn_ts": "Timestamp", "amount": "Amount",
                     "merchant_id": "Merchant", "gnn_score_pct": "Risk Score", "status": "Status"}
        )

    with hcard:
        st.dataframe(history_table(df[df["card_key"] == row["card_key"]]), width="stretch", hide_index=True)
    with huser:
        st.dataframe(history_table(df[df["user_id"] == row["user_id"]]), width="stretch", hide_index=True)
    with hmerchant:
        st.dataframe(history_table(df[df["merchant_id"] == row["merchant_id"]]), width="stretch", hide_index=True)

    st.divider()

    # --- Network view ---
    ui.section_title("Network View (within this upload)")
    hops = st.radio("Neighborhood depth", [1, 2], horizontal=True, key="inv_hops")
    st.caption("Purple = cards, green = merchants, colored by risk = transactions. Black border = this transaction.")

    data, meta = store.active_graph()
    center_node = meta["txn_offset"] + row_idx
    fig, txn_lookup = netviz.build_figure(center_node, data, meta, df, k_hops=hops)
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- Case actions ---
    ui.section_title("Analyst Decision")
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Confirm Fraud"):
        case_store.set_alert_status(txn_id, "CONFIRMED FRAUD")
        store.invalidate_active_df()
        st.rerun()
    if b2.button("Dismiss"):
        case_store.set_alert_status(txn_id, "DISMISSED")
        store.invalidate_active_df()
        st.rerun()
    if b3.button("Escalate"):
        case_store.set_alert_status(txn_id, "ESCALATED")
        store.invalidate_active_df()
        st.rerun()
    if b4.button("Create Case"):
        case_id = case_store.create_case(txn_id, priority=row["risk_level"])
        st.success(f"Created {case_id}")
