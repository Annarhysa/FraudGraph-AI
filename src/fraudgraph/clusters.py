"""
Suspicious-cluster ("potential fraud ring") detection.

Builds a card-level graph where two cards are connected if they share a
merchant AND at least one of those shared transactions is high-risk
(gnn_score_pct >= threshold). Running Louvain community detection on that
graph surfaces groups of cards/users linked through suspicious merchant
activity — a real (if imperfect) proxy for coordinated fraud rings, not a
claim that any given cluster is confirmed criminal activity.

Usage:
    python -m fraudgraph.clusters
"""
from pathlib import Path

import networkx as nx
import pandas as pd

from . import scoring

CLUSTERS_PARQUET = Path("data/processed/clusters.parquet")
CLUSTER_MEMBERS_PARQUET = Path("data/processed/cluster_members.parquet")

MIN_CLUSTER_CARDS = 2
HIGH_RISK_THRESHOLD = scoring.HIGH_RISK_THRESHOLD


def build_card_graph(df: pd.DataFrame) -> nx.Graph:
    high_risk = df[df["gnn_score_pct"] >= HIGH_RISK_THRESHOLD]
    g = nx.Graph()
    g.add_nodes_from(df["card_key"].unique())

    for _merchant_id, group in high_risk.groupby("merchant_id"):
        cards = group["card_key"].unique()
        if len(cards) < 2:
            continue
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                if g.has_edge(cards[i], cards[j]):
                    g[cards[i]][cards[j]]["weight"] += 1
                else:
                    g.add_edge(cards[i], cards[j], weight=1)
    return g


def compute_clusters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure in-memory computation — no disk I/O. Use this for a one-off
    dataframe (e.g. a single user's uploaded batch) so concurrent callers
    don't race on/overwrite the shared clusters.parquet."""
    g = build_card_graph(df)
    connected = [c for c in nx.connected_components(g) if len(c) >= MIN_CLUSTER_CARDS]

    communities: list[set] = []
    for comp in connected:
        sub = g.subgraph(comp)
        if sub.number_of_edges() == 0:
            communities.append(comp)
            continue
        try:
            sub_communities = nx.algorithms.community.louvain_communities(sub, weight="weight", seed=7)
            communities.extend([c for c in sub_communities if len(c) >= MIN_CLUSTER_CARDS])
        except Exception:
            communities.append(comp)

    rows = []
    member_rows = []
    for i, card_keys in enumerate(communities):
        cluster_id = f"RING{i:04d}"
        sub_df = df[df["card_key"].isin(card_keys)]
        rows.append({
            "cluster_id": cluster_id,
            "n_users": sub_df["user_id"].nunique(),
            "n_cards": len(card_keys),
            "n_merchants": sub_df["merchant_id"].nunique(),
            "n_transactions": len(sub_df),
            "n_fraud_transactions": int(sub_df["is_fraud"].sum()),
            "avg_risk_score": round(sub_df["gnn_score_pct"].mean(), 1),
            "max_risk_score": round(sub_df["gnn_score_pct"].max(), 1),
            "first_activity": sub_df["txn_ts"].min(),
            "last_activity": sub_df["txn_ts"].max(),
        })
        for ck in card_keys:
            member_rows.append({"cluster_id": cluster_id, "card_key": ck})

    cluster_columns = [
        "cluster_id", "n_users", "n_cards", "n_merchants", "n_transactions",
        "n_fraud_transactions", "avg_risk_score", "max_risk_score", "first_activity", "last_activity",
    ]
    if rows:
        # Sort by max_risk_score, not avg_risk_score: averaging a community's
        # risk score dilutes large fraud-dense communities (their many
        # low-risk transactions drag the mean down), so avg_risk_score badly
        # under-surfaces the most fraud-relevant clusters — validated
        # empirically at ~0.6% fraud captured in the top 5 by avg_risk_score
        # vs ~32% by max_risk_score (see results/fraud_ring_evaluation_results.txt).
        clusters_df = pd.DataFrame(rows).sort_values("max_risk_score", ascending=False).reset_index(drop=True)
    else:
        clusters_df = pd.DataFrame(columns=cluster_columns)
    members_df = pd.DataFrame(member_rows, columns=["cluster_id", "card_key"])
    return clusters_df, members_df


def detect_clusters(df: pd.DataFrame | None = None, force: bool = False):
    """Disk-cached version, for the shared sample dataset (CLI / offline use)."""
    if CLUSTERS_PARQUET.exists() and CLUSTER_MEMBERS_PARQUET.exists() and not force:
        return pd.read_parquet(CLUSTERS_PARQUET), pd.read_parquet(CLUSTER_MEMBERS_PARQUET)

    if df is None:
        df = scoring.build_master()

    clusters_df, members_df = compute_clusters(df)

    CLUSTERS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    clusters_df.to_parquet(CLUSTERS_PARQUET)
    members_df.to_parquet(CLUSTER_MEMBERS_PARQUET)
    return clusters_df, members_df


def main() -> None:
    clusters_df, members_df = detect_clusters(force=True)
    print(f"Found {len(clusters_df)} suspicious clusters ({len(members_df)} card memberships)")
    print(clusters_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
