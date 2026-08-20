import streamlit as st

from .. import batches as batch_store
from .. import netviz
from ..ui import components as ui


def _find_node(query: str, df, meta):
    query = query.strip()
    if not query:
        return None, None

    match = df[df["transaction_id"].str.upper() == query.upper()]
    if len(match):
        row_idx = batch_store.parse_row_index(match.iloc[0]["transaction_id"])
        return meta["txn_offset"] + row_idx, "transaction"

    if ":" in query:
        match = df[df["card_key"] == query]
        if len(match):
            return _card_node_id(match.iloc[0]["card_key"], df, meta), "card"
    if query.upper().startswith("U") and query[1:].isdigit():
        uid = query[1:]
        match = df[df["user_id"].astype(str) == uid]
        if len(match):
            return _card_node_id(match.iloc[0]["card_key"], df, meta), "card"
    match = df[df["merchant_id"].astype(str) == query]
    if len(match):
        return _merchant_node_id(match.iloc[0]["merchant_id"], df, meta), "merchant"
    return None, None


def _card_node_id(card_key, df, meta):
    keys = list(dict.fromkeys((df["user_id"].astype(str) + ":" + df["card_id"].astype(str)).values))
    return keys.index(card_key)


def _merchant_node_id(merchant_id, df, meta):
    merchants = df["merchant_id"].unique().tolist()
    return meta["txn_offset"] + meta["n_txn"] + merchants.index(merchant_id)


def render(store):
    st.title("Network Explorer")
    st.caption("Search any User / Card / Merchant / Transaction ID within your active upload.")

    df = store.active_df()
    if df is None or df.empty:
        st.info("No transactions in the active upload.")
        return
    data, meta = store.active_graph()

    example_txn = df.iloc[0]["transaction_id"]
    example_card = df.iloc[0]["card_key"]
    c1, c2 = st.columns([3, 1])
    query = c1.text_input(f"Search (e.g. {example_txn}, {example_card}, U{df.iloc[0]['user_id']}, or a merchant ID)")
    depth = c2.selectbox("Depth", [1, 2, 3], index=0)

    if not query:
        st.info(f"Enter a User ID (e.g. `U{df.iloc[0]['user_id']}`), Card ID (e.g. `{example_card}`), "
                f"Merchant ID, or Transaction ID (e.g. `{example_txn}`) to begin.")
        return

    node_id, node_kind = _find_node(query, df, meta)
    if node_id is None:
        st.error("No matching entity found in your active upload.")
        return

    st.success(f"Found {node_kind} node — showing {depth}-hop neighborhood")

    fig, txn_lookup = netviz.build_figure(node_id, data, meta, df, k_hops=depth, max_nodes=netviz.MAX_NODES_DEFAULT)
    st.plotly_chart(fig, width="stretch")

    st.divider()
    ui.section_title("Neighborhood summary")
    sub_nodes, _, _ = netviz.get_ego_subgraph(node_id, data.edge_index, depth, netviz.MAX_NODES_DEFAULT)
    txn_row_idxs = [int(n) - meta["txn_offset"] for n in sub_nodes if meta["txn_offset"] <= n < meta["txn_offset"] + meta["n_txn"]]
    neighborhood_df = df.iloc[txn_row_idxs]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions in view", len(neighborhood_df))
    c2.metric("Connected users", neighborhood_df["user_id"].nunique())
    c3.metric("Connected merchants", neighborhood_df["merchant_id"].nunique())
    c4.metric("High-risk txns", int(neighborhood_df["risk_level"].isin(["HIGH", "CRITICAL"]).sum()))

    risk_filter = st.selectbox("Show", ["All transactions", "Suspicious only (MEDIUM+)", "High-risk only (HIGH+)", "Known fraud only"])
    view = neighborhood_df
    if risk_filter == "Suspicious only (MEDIUM+)":
        view = view[view["risk_level"].isin(["MEDIUM", "HIGH", "CRITICAL"])]
    elif risk_filter == "High-risk only (HIGH+)":
        view = view[view["risk_level"].isin(["HIGH", "CRITICAL"])]
    elif risk_filter == "Known fraud only":
        view = view[view["is_fraud"]]

    st.dataframe(
        view[["transaction_id", "risk_level", "gnn_score_pct", "amount", "card_key", "merchant_id", "txn_ts"]],
        width="stretch", hide_index=True,
    )
