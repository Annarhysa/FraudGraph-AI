import streamlit as st

from .. import netviz
from ..ui import components as ui


def render(store):
    st.title("Fraud Rings")
    st.caption(
        "Suspicious clusters detected via Louvain community detection over a card-level graph "
        "(two cards are linked if they share a merchant where at least one transaction is high-risk). "
        "These are potential fraud rings, not confirmed criminal networks — always investigate before acting."
    )

    clusters_df, members_df = store.active_clusters()

    if clusters_df is None or clusters_df.empty:
        st.info(
            "No suspicious clusters found in your active upload. Ring detection needs multiple "
            "cards sharing a merchant with high-risk activity — small uploads often won't have that."
        )
        return

    st.write(f"**{len(clusters_df)}** suspicious clusters found")
    st.dataframe(
        clusters_df.rename(columns={
            "cluster_id": "Cluster ID", "n_users": "Users", "n_cards": "Cards",
            "n_merchants": "Merchants", "n_transactions": "Transactions",
            "n_fraud_transactions": "Fraud Txns", "avg_risk_score": "Avg Risk",
            "max_risk_score": "Max Risk", "first_activity": "First Activity", "last_activity": "Last Activity",
        }),
        width="stretch", hide_index=True,
    )

    st.divider()
    cluster_id = st.selectbox("Inspect cluster", clusters_df["cluster_id"].tolist())
    row = clusters_df[clusters_df["cluster_id"] == cluster_id].iloc[0]

    st.subheader(f"Potential Fraud Ring — {cluster_id}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", int(row["n_users"]))
    c2.metric("Cards", int(row["n_cards"]))
    c3.metric("Merchants", int(row["n_merchants"]))
    c4.metric("Risk Score", f"{row['avg_risk_score']:.1f}%")

    df = store.active_df()
    card_keys = members_df[members_df["cluster_id"] == cluster_id]["card_key"].tolist()
    cluster_txns = df[df["card_key"].isin(card_keys)]

    ui.section_title("Cluster graph")
    data, meta = store.active_graph()
    key_to_node_id = {k: i for i, k in enumerate(
        list(dict.fromkeys((df["user_id"].astype(str) + ":" + df["card_id"].astype(str)).values))
    )}
    center_node = key_to_node_id[card_keys[0]]
    fig, _ = netviz.build_figure(center_node, data, meta, df, k_hops=2, max_nodes=150)
    st.plotly_chart(fig, width="stretch")

    ui.section_title("Transactions in this cluster")
    st.dataframe(
        cluster_txns[["transaction_id", "risk_level", "gnn_score_pct", "amount", "card_key", "merchant_id", "txn_ts", "is_fraud"]]
        .sort_values("gnn_score_pct", ascending=False).head(100),
        width="stretch", hide_index=True,
    )
