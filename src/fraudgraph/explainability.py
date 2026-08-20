"""
Turns a scored transaction into three explicitly-labeled evidence categories:

  RULE signals    — from anomaly.py's hand-written heuristics
  NETWORK signals — graph connectivity computed in scoring.py
  MODEL signals   — GNNExplainer run on the trained GraphSAGE model, restricted
                     to a bounded k-hop subgraph around the transaction so it's
                     fast enough to run on demand from the UI

Nothing here is fabricated: RULE and NETWORK signals are read straight off
columns produced upstream, and MODEL signals come from actually running
GNNExplainer against the saved model weights.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from . import anomaly as rule
from . import gnn

FEATURE_NAMES = [
    "is_card", "is_transaction", "is_merchant",
    "amount (log, z-scored)", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "channel: swipe", "channel: chip", "channel: online", "had_error_flag",
]


def rule_signals(row: pd.Series) -> list[str]:
    return rule.explain(row)


def network_signals(row: pd.Series) -> list[str]:
    out = []
    hr_cards = int(row.get("merchant_high_risk_cards", 0))
    shared = int(row.get("card_shared_suspicious_accounts", 0))
    if hr_cards >= 1:
        out.append(f"Merchant is also used by {hr_cards} other high-risk card(s)")
    if shared >= 1:
        out.append(f"Card shares a merchant with {shared} other suspicious account(s)")
    if not out:
        out.append("No notable connections to other high-risk activity in this sample")
    return out


class ModelExplainer:
    """Wraps GNNExplainer over a bounded k-hop ego-subgraph so it's cheap
    enough to run interactively from the Investigation page."""

    def __init__(self, data, meta, model, k_hops: int = 2, epochs: int = 100):
        self.data = data
        self.meta = meta
        self.model = model
        self.k_hops = k_hops
        self.epochs = epochs

    def explain_transaction(self, row_idx: int) -> dict:
        node_id = self.meta["txn_offset"] + row_idx

        sub_nodes, sub_edge_index, mapping, edge_mask_full = k_hop_subgraph(
            node_idx=node_id,
            num_hops=self.k_hops,
            edge_index=self.data.edge_index,
            relabel_nodes=True,
        )
        sub_x = self.data.x[sub_nodes]
        target_local_idx = mapping.item()

        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=self.epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
        )
        explanation = explainer(sub_x, sub_edge_index, index=target_local_idx)

        feat_importance = explanation.node_mask[target_local_idx].detach().numpy()
        top_feat_idx = np.argsort(-np.abs(feat_importance))[:4]
        top_features = [
            {"feature": FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}",
             "importance": float(feat_importance[i])}
            for i in top_feat_idx
        ]

        edge_importance = explanation.edge_mask.detach().numpy()
        top_edge_idx = np.argsort(-edge_importance)[:5]
        global_sub_nodes = sub_nodes.numpy()
        top_edges = []
        for e in top_edge_idx:
            if edge_importance[e] <= 0:
                continue
            src_local, dst_local = sub_edge_index[0, e].item(), sub_edge_index[1, e].item()
            src_global, dst_global = int(global_sub_nodes[src_local]), int(global_sub_nodes[dst_local])
            top_edges.append({
                "src": describe_node(src_global, self.meta),
                "dst": describe_node(dst_global, self.meta),
                "importance": float(edge_importance[e]),
            })

        return {"top_features": top_features, "top_edges": top_edges, "subgraph_nodes": len(sub_nodes)}


def describe_node(node_id: int, meta: dict) -> str:
    txn_offset = meta["txn_offset"]
    n_txn = meta["n_txn"]
    if node_id < txn_offset:
        return f"card #{node_id}"
    if node_id < txn_offset + n_txn:
        return f"transaction TXN{node_id - txn_offset:07d}"
    return f"merchant #{node_id - txn_offset - n_txn}"


def top_signal_summary(row: pd.Series) -> list[tuple[str, str]]:
    """Coarse HIGH/MEDIUM/LOW labels for the at-a-glance signal summary."""
    out = []
    velocity = "HIGH" if row["velocity_1h"] >= 5 else ("MEDIUM" if row["velocity_1h"] >= 2 else "LOW")
    out.append(("Transaction velocity", velocity))

    connectivity = int(row.get("merchant_high_risk_cards", 0)) + int(row.get("card_shared_suspicious_accounts", 0))
    conn_level = "HIGH" if connectivity >= 3 else ("MEDIUM" if connectivity >= 1 else "LOW")
    out.append(("Merchant/card connectivity", conn_level))

    amt = abs(row["amount_zscore"])
    amt_level = "HIGH" if amt >= 3 else ("MEDIUM" if amt >= 1.5 else "LOW")
    out.append(("Amount deviation", amt_level))

    out.append(("New merchant for this user", "MEDIUM" if row["new_merchant"] else "LOW"))
    return out
