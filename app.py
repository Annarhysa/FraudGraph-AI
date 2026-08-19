"""
FraudGraph AI — alert triage dashboard.

Shows the top anomaly-scored transactions from src/fraudgraph/anomaly.py,
each with a human-readable "why" explanation, plus a graph view of the
flagged card's neighborhood (which merchants it touches, and which other
cards/users share those merchants).

Run:
    streamlit run app.py
"""
import ast
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ALERTS_CSV = Path("data/processed/alerts.csv")
GRAPH_PATH = Path("data/processed/graph.pkl")

st.set_page_config(page_title="FraudGraph AI", layout="wide")


@st.cache_data
def load_alerts() -> pd.DataFrame:
    df = pd.read_csv(ALERTS_CSV)
    df["txn_ts"] = pd.to_datetime(df["txn_ts"])
    return df


@st.cache_resource
def load_graph() -> nx.MultiDiGraph:
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def ego_subgraph(g: nx.MultiDiGraph, card_node: str, radius: int = 2) -> nx.Graph:
    und = g.to_undirected()
    nodes = nx.single_source_shortest_path_length(und, card_node, cutoff=radius)
    return und.subgraph(nodes.keys())


def plot_neighborhood(g: nx.MultiDiGraph, card_node: str):
    sub = ego_subgraph(g, card_node, radius=2)
    if sub.number_of_nodes() > 120:
        # keep the view readable: nearest nodes only
        und = g.to_undirected()
        nodes = nx.single_source_shortest_path_length(und, card_node, cutoff=1)
        sub = und.subgraph(nodes.keys())

    pos = nx.spring_layout(sub, seed=7)

    edge_x, edge_y = [], []
    for u, v in sub.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.5, color="#888"), hoverinfo="none")

    color_map = {"user": "#4C78A8", "card": "#F58518", "merchant": "#54A24B"}
    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for n in sub.nodes():
        node_x.append(pos[n][0])
        node_y.append(pos[n][1])
        ntype = sub.nodes[n].get("type", "?")
        node_color.append("#E45756" if n == card_node else color_map.get(ntype, "#999"))
        node_text.append(n)
        node_size.append(18 if n == card_node else 9)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_text,
        marker=dict(color=node_color, size=node_size, line=dict(width=1, color="#fff")),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=420,
    )
    return fig


def main():
    st.title("FraudGraph AI")
    st.caption("Explainable fraud alerts from graph analytics + rule-based scoring (baseline, pre-GNN)")

    if not ALERTS_CSV.exists() or not GRAPH_PATH.exists():
        st.error(
            "Missing processed data. Run:\n\n"
            "```\npython -m fraudgraph.data_prep\npython -m fraudgraph.graph_build\npython -m fraudgraph.anomaly\n```"
        )
        return

    alerts = load_alerts()
    g = load_graph()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Flagged transactions")
        alerts_display = alerts.copy()
        alerts_display["label"] = alerts_display.apply(
            lambda r: f"#{r.name} · ${r['amount']:.2f} · score {r['anomaly_score']:.2f}"
            + (" · FRAUD" if r["is_fraud"] else ""),
            axis=1,
        )
        choice = st.selectbox(
            "Select an alert", alerts_display.index,
            format_func=lambda i: alerts_display.loc[i, "label"],
        )
        st.dataframe(
            alerts[["user_id", "amount", "anomaly_score", "is_fraud"]].head(50),
            use_container_width=True, height=400,
        )

    row = alerts.loc[choice]
    card_node = f"card:{row['user_id']}:{row['card_id']}"

    with col2:
        st.subheader(f"Transaction — ${row['amount']:.2f}")
        st.metric("Anomaly score", f"{row['anomaly_score']:.3f}")
        st.write(f"**User:** {row['user_id']}  |  **Card:** {row['card_id']}  |  **When:** {row['txn_ts']}")

        st.markdown("#### Why?")
        reasons = ast.literal_eval(row["reasons_text"]) if row["reasons_text"].startswith("[") \
            else row["reasons_text"].split(" | ")
        for r in reasons:
            st.markdown(f"- {r}")

        if row["is_fraud"]:
            st.warning("Ground truth: this transaction IS labeled fraud in the source data.")

        st.markdown("#### Card neighborhood")
        st.caption("Blue = user, orange = cards, green = merchants, red = this transaction's card")
        if card_node in g:
            st.plotly_chart(plot_neighborhood(g, card_node), use_container_width=True)
        else:
            st.info("Card node not found in graph sample.")


if __name__ == "__main__":
    main()
