"""
Build a User -> Card -> Merchant graph from the transaction sample.

Node ids:
    user:{user_id}
    card:{user_id}:{card_id}      (card_id is only unique *within* a user
                                    in the source data, so it's namespaced)
    merchant:{merchant_id}

Edges:
    user  -> card      "HAS_CARD"
    card  -> merchant  "TXN"        one edge per transaction, carrying
                                     amount/date/time/channel/mcc/errors/is_fraud

Usage:
    python -m fraudgraph.graph_build
"""
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

SAMPLE_PARQUET = Path("data/processed/transactions_sample.parquet")
GRAPH_PATH = Path("data/processed/graph.pkl")


def load_sample() -> pd.DataFrame:
    df = pd.read_parquet(SAMPLE_PARQUET)
    df["txn_ts"] = pd.to_datetime(
        df["txn_date"].astype(str) + " " + df["txn_time"].astype(str)
    )
    return df


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    users = df["user_id"].unique()
    g.add_nodes_from((f"user:{u}", {"type": "user"}) for u in users)

    cards = df[["user_id", "card_id"]].drop_duplicates()
    for row in cards.itertuples(index=False):
        card_node = f"card:{row.user_id}:{row.card_id}"
        g.add_node(card_node, type="card")
        g.add_edge(f"user:{row.user_id}", card_node, type="HAS_CARD")

    merchants = df["merchant_id"].unique()
    g.add_nodes_from((f"merchant:{m}", {"type": "merchant"}) for m in merchants)

    for row in df.itertuples(index=False):
        card_node = f"card:{row.user_id}:{row.card_id}"
        merchant_node = f"merchant:{row.merchant_id}"
        g.add_edge(
            card_node,
            merchant_node,
            type="TXN",
            amount=row.amount,
            ts=row.txn_ts,
            channel=row.channel,
            mcc=row.mcc,
            errors=row.errors,
            is_fraud=bool(row.is_fraud),
            merchant_city=row.merchant_city,
            merchant_state=row.merchant_state,
        )

    return g


def main() -> None:
    df = load_sample()
    print(f"Loaded {len(df)} transactions")

    g = build_graph(df)
    print(
        f"Graph: {g.number_of_nodes()} nodes "
        f"({sum(1 for _, d in g.nodes(data=True) if d['type'] == 'user')} users, "
        f"{sum(1 for _, d in g.nodes(data=True) if d['type'] == 'card')} cards, "
        f"{sum(1 for _, d in g.nodes(data=True) if d['type'] == 'merchant')} merchants), "
        f"{g.number_of_edges()} edges"
    )

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(g, f)
    print(f"Graph written to {GRAPH_PATH}")


if __name__ == "__main__":
    main()
