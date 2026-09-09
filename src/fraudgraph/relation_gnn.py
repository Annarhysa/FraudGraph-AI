"""
Phase 3: one controlled architecture experiment to explain why Full Graph
(homogeneous SAGEConv over card+merchant edges mixed together, 0.0623) loses
to Merchant-only (0.2063). Hypothesis: plain SAGEConv mean-aggregates every
neighbor — card or merchant — with the same weights, so a transaction node's
embedding blends its (informative) merchant neighbor with its (uninformative,
per the edge-type ablation) card neighbor, diluting the useful signal.

RelationAwareSAGE gives each edge type its own SAGEConv (own weight matrix),
so card and merchant neighborhoods are aggregated independently and then
summed — the model can learn to weight/ignore the card pathway instead of
being forced to average it in. Not a new-architecture search: this is the one
well-motivated variant that directly tests the dilution hypothesis.

Usage:
    python -m fraudgraph.relation_gnn
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from . import gnn
from . import metrics as metrics_mod
from .baselines import _load_and_split, _fmt

RESULTS_TXT = Path("results/relation_aware_results.txt")
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 100
HIDDEN_DIM = 32
LR = 0.01

REFERENCE = {
    "No Graph": (0.0151, 0.0003),
    "Merchant-only": (0.2063, 0.0244),
    "Card-only": (0.0113, 0.0004),
    "Full Graph (homogeneous SAGEConv)": (0.0623, 0.0117),
}


class RelationAwareSAGE(torch.nn.Module):
    """Separate SAGEConv per edge type (card, merchant), summed after each
    layer, instead of one SAGEConv over both edge types mixed together."""

    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.card_conv1 = SAGEConv(in_dim, hidden_dim)
        self.merch_conv1 = SAGEConv(in_dim, hidden_dim)
        self.card_conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.merch_conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index_card, edge_index_merchant):
        h = F.relu(self.card_conv1(x, edge_index_card)) + F.relu(self.merch_conv1(x, edge_index_merchant))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.card_conv2(h, edge_index_card)) + F.relu(self.merch_conv2(h, edge_index_merchant))
        return self.out(h).squeeze(-1)


def train_relation_aware(x, edge_index_card, edge_index_merchant, y, train_idx, val_idx, seed, epochs=EPOCHS, hidden_dim=HIDDEN_DIM, lr=LR):
    torch.manual_seed(seed)
    model = RelationAwareSAGE(in_dim=x.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    y_train = y[train_idx]
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1))
    print(f"train txns: {len(train_idx)} ({n_pos} fraud, pos_weight={pos_weight:.1f})")

    best_val_ap = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index_card, edge_index_merchant)
        loss = F.binary_cross_entropy_with_logits(logits[train_idx], y[train_idx], pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index_card, edge_index_merchant)[val_idx]
            val_probs = torch.sigmoid(val_logits).numpy()
            val_y = y[val_idx].numpy()
            val_ap = average_precision_score(val_y, val_probs)
            val_auc = roc_auc_score(val_y, val_probs)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  val AP {val_ap:.4f}  val AUC {val_auc:.4f}")

    model.load_state_dict(best_state)
    print(f"Best val AP: {best_val_ap:.4f}")
    return model


def score_all_nodes(model, x, edge_index_card, edge_index_merchant, meta):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x, edge_index_card, edge_index_merchant))
    txn_offset = meta["txn_offset"]
    n_txn = meta["n_txn"]
    return probs[txn_offset:txn_offset + n_txn].numpy()


def run() -> list:
    df, train_idx, val_idx, test_idx = _load_and_split()

    # Same node features/layout for all conditions; only which edges are
    # exposed to which conv differs. Building card-only and merchant-only
    # graphs separately and taking their edge_index guarantees identical x/y
    # (gnn.build_graph keeps node layout fixed regardless of edge flags).
    data_card, meta = gnn.build_graph(df, include_card_edges=True, include_merchant_edges=False)
    data_merch, meta2 = gnn.build_graph(df, include_card_edges=False, include_merchant_edges=True)
    assert meta["n_txn"] == meta2["n_txn"] and meta["txn_offset"] == meta2["txn_offset"]
    assert torch.equal(data_card.x, data_merch.x) and torch.equal(data_card.y, data_merch.y)

    x, y = data_card.x, data_card.y
    edge_index_card = data_card.edge_index
    edge_index_merchant = data_merch.edge_index

    g_train_idx, g_val_idx, g_test_idx = gnn.temporal_split(meta["n_txn"], meta["txn_offset"])

    results = []
    for seed in SEEDS:
        print(f"[RelationAwareSAGE] seed {seed} -> training...")
        model = train_relation_aware(x, edge_index_card, edge_index_merchant, y, g_train_idx, g_val_idx, seed)
        scores = score_all_nodes(model, x, edge_index_card, edge_index_merchant, meta)
        df["score__relation_aware"] = scores
        m = metrics_mod.compute_metrics(df, "score__relation_aware", use_split=True)
        results.append(m)
        print(f"[RelationAwareSAGE] seed {seed} -> PR-AUC={_fmt(m.get('pr_auc'))}  P@K={_fmt(m.get('precision_at_k'),3)}")

    write_report(results)
    return results


def write_report(results: list, path: Path = RESULTS_TXT) -> None:
    pr_vals = [m.get("pr_auc") for m in results if m.get("pr_auc") is not None]
    pk_vals = [m.get("precision_at_k") for m in results if m.get("precision_at_k") is not None]
    lift_vals = [m.get("lift") for m in results if m.get("lift") is not None]
    pr_mean, pr_std = float(np.mean(pr_vals)), float(np.std(pr_vals))
    pk_mean, pk_std = float(np.mean(pk_vals)), float(np.std(pk_vals))
    lift_mean, lift_std = float(np.mean(lift_vals)), float(np.std(lift_vals))

    lines = []
    lines.append("=" * 92)
    lines.append("FraudGraph AI — Relation-Aware GNN (Phase 3): does separating edge types help?")
    lines.append("=" * 92)
    lines.append("")
    lines.append("Hypothesis: homogeneous SAGEConv mean-aggregates card and merchant neighbors")
    lines.append("together with shared weights, diluting the strong merchant signal with the")
    lines.append("uninformative card signal. RelationAwareSAGE gives each edge type its own")
    lines.append("SAGEConv (separate weights), summing the two representations per layer instead")
    lines.append("of aggregating over a single mixed edge set.")
    lines.append("")
    lines.append(f"Config: hidden_dim={HIDDEN_DIM}, epochs={EPOCHS}, lr={LR}, seeds={SEEDS}, Window 0.")
    lines.append("")
    lines.append("-" * 92)
    header = f"{'Condition':<38}{'PR-AUC':>20}{'Precision@K':>18}{'Lift':>16}"
    lines.append(header)
    lines.append("-" * len(header))

    normalized = []
    for name, val in REFERENCE.items():
        normalized.append((name, val[0], val[1], None, None, None, None))
    normalized.append(("Relation-Aware GNN (separate card + merchant convs)", pr_mean, pr_std, pk_mean, pk_std, lift_mean, lift_std))

    normalized.sort(key=lambda r: -(r[1] or -1))
    for name, pr_m, pr_s, pk_m, pk_s, lift_m, lift_s in normalized:
        pr_str = f"{_fmt(pr_m)} +/- {_fmt(pr_s)}"
        pk_str = f"{_fmt(pk_m,3)} +/- {_fmt(pk_s,3)}" if pk_m is not None else "n/a"
        lift_str = f"{_fmt(lift_m,1)}x +/- {_fmt(lift_s,1)}" if lift_m is not None else "n/a"
        lines.append(f"{name:<38}{pr_str:>20}{pk_str:>18}{lift_str:>16}")

    lines.append("")
    lines.append("-" * 92)
    lines.append("Per-seed PR-AUC (Relation-Aware GNN)")
    lines.append("-" * 92)
    for seed, m in zip(SEEDS, results):
        lines.append(f"  seed {seed}: {_fmt(m.get('pr_auc'))}")

    merch_mean, merch_std = REFERENCE["Merchant-only"]
    full_mean, full_std = REFERENCE["Full Graph (homogeneous SAGEConv)"]

    lines.append("")
    lines.append("-" * 92)
    lines.append("Verdict")
    lines.append("-" * 92)
    lines.append(f"  Full Graph (homogeneous):    {full_mean:.4f} +/- {full_std:.4f}")
    lines.append(f"  Relation-Aware GNN:          {pr_mean:.4f} +/- {pr_std:.4f}")
    lines.append(f"  Merchant-only (upper ref):   {merch_mean:.4f} +/- {merch_std:.4f}")
    lines.append("")
    if pr_mean > full_mean + full_std:
        if pr_mean >= merch_mean - merch_std:
            lines.append("  -> CONFIRMED: separating edge types recovers most/all of the merchant-only")
            lines.append("     signal while still including card edges. The dilution hypothesis holds —")
            lines.append("     homogeneous aggregation, not the card relationship itself, was the problem.")
        else:
            lines.append("  -> PARTIALLY CONFIRMED: relation-aware aggregation substantially beats the")
            lines.append("     homogeneous Full Graph, supporting the dilution hypothesis, but still falls")
            lines.append("     short of Merchant-only alone — card edges contribute some residual noise")
            lines.append("     even when aggregated separately, or the model needs more capacity/tuning")
            lines.append("     to fully exploit the separated pathways.")
    else:
        lines.append("  -> NOT CONFIRMED: relation-aware aggregation does not meaningfully beat the")
        lines.append("     homogeneous Full Graph. The dilution hypothesis is not supported by this")
        lines.append("     experiment — the card relationship may be actively unhelpful (adding noise)")
        lines.append("     regardless of how it's aggregated, not merely diluting the merchant signal.")
    lines.append("=" * 92)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    run()
