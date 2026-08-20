import streamlit as st

from ..ui import components as ui


def render(store):
    st.title("Transactions")
    st.caption("Searchable explorer over your currently active upload")

    df = store.active_df()
    if df is None or df.empty:
        st.info("No transactions in the active upload.")
        return

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        min_amt, max_amt = float(df["amount"].min()), float(df["amount"].max())
        amount_range = c1.slider("Amount", 0.0, round(max_amt, 2), (0.0, round(max_amt, 2)))
        fraud_only = c2.checkbox("Fraud-labeled only")
        risk_min = c3.slider("Minimum risk score", 0.0, 100.0, 0.0)

        c4, c5, c6 = st.columns(3)
        card_search = c4.text_input("Card ID")
        merchant_search = c5.text_input("Merchant ID")
        user_search = c6.text_input("User ID")

    filtered = df[(df["amount"] >= amount_range[0]) & (df["amount"] <= amount_range[1])
                  & (df["gnn_score_pct"] >= risk_min)]
    if fraud_only:
        filtered = filtered[filtered["is_fraud"]]
    if card_search:
        filtered = filtered[filtered["card_key"].astype(str).str.contains(card_search)]
    if merchant_search:
        filtered = filtered[filtered["merchant_id"].astype(str).str.contains(merchant_search)]
    if user_search:
        filtered = filtered[filtered["user_id"].astype(str).str.contains(user_search)]

    sort_by = st.selectbox("Sort by", ["Timestamp (newest)", "Amount (highest)", "Risk score (highest)"])
    if sort_by == "Timestamp (newest)":
        filtered = filtered.sort_values("txn_ts", ascending=False)
    elif sort_by == "Amount (highest)":
        filtered = filtered.sort_values("amount", ascending=False)
    else:
        filtered = filtered.sort_values("gnn_score_pct", ascending=False)

    st.write(f"**{len(filtered):,}** transactions match")

    page_size = 50
    total_pages = max(1, (len(filtered) - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    page_df = filtered.iloc[(page - 1) * page_size: page * page_size]

    display = page_df[[
        "transaction_id", "txn_ts", "amount", "card_key", "merchant_id", "gnn_score_pct", "is_fraud", "status",
    ]].rename(columns={
        "transaction_id": "Transaction ID", "txn_ts": "Timestamp", "amount": "Amount",
        "card_key": "Card", "merchant_id": "Merchant", "gnn_score_pct": "Risk Score",
        "is_fraud": "Fraud Label", "status": "Status",
    })
    st.dataframe(display, width="stretch", hide_index=True, height=600)

    txn_options = page_df["transaction_id"].tolist()
    if txn_options:
        chosen = st.selectbox("Open transaction", txn_options)
        if st.button("Investigate →", type="primary"):
            st.session_state["investigation_txn_id"] = chosen
            st.session_state["nav_page"] = "Investigation"
            st.rerun()
