"""
Shared network-visualization helper: renders a bounded k-hop neighborhood
around a node (card, transaction, or merchant) as an interactive Plotly
figure. Used by the Investigation page and the Network Explorer.

Node caps keep the view legible: default 1-hop / 100 nodes, expandable to
more hops up to a hard 300-node cap (spec requirement — no giant hairball
graphs rendered by default).
"""
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import torch
from torch_geometric.utils import k_hop_subgraph

RISK_COLORS = {"LOW": "#64748B", "MEDIUM": "#D4A72C", "HIGH": "#E07B39", "CRITICAL": "#DC2626"}
CARD_COLOR = "#60A5FA"
MERCHANT_COLOR = "#2563EB"
CENTER_BORDER = "#0F172A"
DEFAULT_BORDER = "#E2E8F0"

MAX_NODES_DEFAULT = 100
MAX_NODES_HARD_CAP = 300


def node_type(node_id: int, meta: dict) -> str:
    if node_id < meta["txn_offset"]:
        return "card"
    if node_id < meta["txn_offset"] + meta["n_txn"]:
        return "transaction"
    return "merchant"


def node_label(node_id: int, meta: dict, df) -> str:
    t = node_type(node_id, meta)
    if t == "transaction":
        row_idx = node_id - meta["txn_offset"]
        row = df.iloc[row_idx]
        return f"{row['transaction_id']} · ${row['amount']:.2f}"
    if t == "card":
        return f"card node {node_id}"
    return f"merchant node {node_id - meta['txn_offset'] - meta['n_txn']}"


def get_ego_subgraph(center_node_id: int, edge_index: torch.Tensor, k_hops: int, max_nodes: int = MAX_NODES_DEFAULT):
    max_nodes = min(max_nodes, MAX_NODES_HARD_CAP)
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=center_node_id, num_hops=k_hops, edge_index=edge_index, relabel_nodes=True
    )
    sub_nodes = sub_nodes.numpy()
    center_local = mapping.item()

    if len(sub_nodes) > max_nodes:
        g_full = nx.Graph()
        g_full.add_nodes_from(range(len(sub_nodes)))
        e = sub_edge_index.numpy()
        g_full.add_edges_from(zip(e[0], e[1]))
        degree = dict(g_full.degree())
        keep_local = sorted(degree, key=lambda n: -degree[n])[:max_nodes]
        if center_local not in keep_local:
            keep_local = [center_local] + keep_local[:-1]
        keep_local_set = set(keep_local)
        remap = {old: new for new, old in enumerate(sorted(keep_local_set))}
        sub_nodes = sub_nodes[sorted(keep_local_set)]
        edges = e.T
        edges = edges[[(a in keep_local_set and b in keep_local_set) for a, b in edges]]
        sub_edge_index = torch.tensor(
            [[remap[a] for a, b in edges], [remap[b] for a, b in edges]], dtype=torch.long
        )
        center_local = remap[center_local]

    return sub_nodes, sub_edge_index, center_local


def build_figure(center_node_id: int, data, meta, df, k_hops: int = 1, max_nodes: int = MAX_NODES_DEFAULT):
    sub_nodes, sub_edge_index, center_local = get_ego_subgraph(
        center_node_id, data.edge_index, k_hops, max_nodes
    )

    g = nx.Graph()
    g.add_nodes_from(range(len(sub_nodes)))
    e = sub_edge_index.numpy()
    g.add_edges_from(zip(e[0], e[1]))
    pos = nx.spring_layout(g, seed=7, k=1.2 / max(len(sub_nodes) ** 0.5, 1))

    edge_x, edge_y = [], []
    for u, v in g.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.6, color="#999"), hoverinfo="none")

    node_x, node_y, colors, sizes, texts, borders = [], [], [], [], [], []
    txn_id_by_local = {}
    for local_idx, global_id in enumerate(sub_nodes):
        node_x.append(pos[local_idx][0])
        node_y.append(pos[local_idx][1])
        t = node_type(int(global_id), meta)
        is_center = local_idx == center_local

        if t == "transaction":
            row_idx = int(global_id) - meta["txn_offset"]
            row = df.iloc[row_idx]
            colors.append(RISK_COLORS[row["risk_level"]])
            txn_id_by_local[local_idx] = row["transaction_id"]
        elif t == "card":
            colors.append(CARD_COLOR)
        else:
            colors.append(MERCHANT_COLOR)

        sizes.append(20 if is_center else (10 if t == "transaction" else 13))
        borders.append(CENTER_BORDER if is_center else DEFAULT_BORDER)
        texts.append(f"{t}: {node_label(int(global_id), meta, df)}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", hoverinfo="text", text=texts,
        marker=dict(color=colors, size=sizes, line=dict(width=2, color=borders)),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=460,
    )
    return fig, txn_id_by_local
