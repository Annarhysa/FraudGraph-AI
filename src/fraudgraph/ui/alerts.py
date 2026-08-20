import pandas as pd
import streamlit as st

from ..ui import components as ui

RISK_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def render(store):
    st.title("Fraud Alerts")
    st.caption("Your work queue — GraphSAGE-scored transactions from your currently active upload")

    df = store.active_df()
    if df is None or df.empty:
        st.info("No transactions in the active upload.")
        return

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        risk_filter = c1.multiselect("Risk level", RISK_ORDER, default=["CRITICAL", "HIGH"])
        status_filter = c2.multiselect(
            "Status", ["OPEN", "UNDER REVIEW", "CONFIRMED FRAUD", "DISMISSED", "ESCALATED"], default=[]
        )
        available_splits = sorted(df["split"].unique().tolist())
        split_filter = c3.multiselect("Model split", available_splits, default=available_splits)

        c4, c5, c6 = st.columns(3)
        min_amt, max_amt = float(df["amount"].min()), float(df["amount"].max())
        amount_range = c4.slider("Amount range", min_value=0.0, max_value=round(max_amt, 2),
                                  value=(0.0, round(max_amt, 2)))
        merchant_search = c5.text_input("Merchant ID contains")
        card_search = c6.text_input("Card ID contains")

        search = st.text_input("Search transaction / card / merchant / user ID")
        sort_by = st.selectbox(
            "Sort by", ["Highest risk", "Newest", "Highest amount", "Most connected"]
        )

    filtered = df.copy()
    if risk_filter:
        filtered = filtered[filtered["risk_level"].isin(risk_filter)]
    if status_filter:
        filtered = filtered[filtered["status"].isin(status_filter)]
    if split_filter:
        filtered = filtered[filtered["split"].isin(split_filter)]
    filtered = filtered[(filtered["amount"] >= amount_range[0]) & (filtered["amount"] <= amount_range[1])]
    if merchant_search:
        filtered = filtered[filtered["merchant_id"].astype(str).str.contains(merchant_search)]
    if card_search:
        filtered = filtered[filtered["card_key"].astype(str).str.contains(card_search)]
    if search:
        mask = (
            filtered["transaction_id"].astype(str).str.contains(search, case=False)
            | filtered["card_key"].astype(str).str.contains(search, case=False)
            | filtered["merchant_id"].astype(str).str.contains(search, case=False)
            | filtered["user_id"].astype(str).str.contains(search, case=False)
        )
        filtered = filtered[mask]

    if sort_by == "Highest risk":
        filtered = filtered.sort_values("gnn_score_pct", ascending=False)
    elif sort_by == "Newest":
        filtered = filtered.sort_values("txn_ts", ascending=False)
    elif sort_by == "Highest amount":
        filtered = filtered.sort_values("amount", ascending=False)
    elif sort_by == "Most connected":
        filtered = filtered.sort_values(
            ["merchant_high_risk_cards", "card_shared_suspicious_accounts"], ascending=False
        )

    st.write(f"**{len(filtered):,}** alerts match your filters")

    page_size = 25
    total_pages = max(1, (len(filtered) - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    page_df = filtered.iloc[(page - 1) * page_size: page * page_size]

    display = page_df[[
        "transaction_id", "risk_level", "gnn_score_pct", "amount", "card_key",
        "merchant_id", "txn_ts", "status", "is_fraud",
    ]].rename(columns={
        "transaction_id": "Transaction ID", "risk_level": "Risk", "gnn_score_pct": "Score",
        "amount": "Amount", "card_key": "Card", "merchant_id": "Merchant",
        "txn_ts": "Timestamp", "status": "Status", "is_fraud": "Fraud Label",
    })
    st.dataframe(display, width="stretch", hide_index=True, height=500)

    st.divider()
    st.caption("Open a transaction to investigate:")
    txn_options = page_df["transaction_id"].tolist()
    if txn_options:
        chosen = st.selectbox("Transaction", txn_options)
        if st.button("Open Investigation →", type="primary"):
            st.session_state["investigation_txn_id"] = chosen
            st.session_state["nav_page"] = "Investigation"
            st.rerun()
