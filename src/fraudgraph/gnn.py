"""
GraphSAGE fraud classifier over a tripartite Card <-> Transaction <-> Merchant
graph. This is the model layer that should beat the rule-based baseline in
anomaly.py (4.9x lift over random at precision@200).

Graph:
    node types: card, transaction, merchant
    edges:      card <-> transaction <-> merchant   (bidirectional)
    label:      is_fraud, defined only on transaction nodes

Split is temporal (train on earliest transactions, test on latest) so the
model is evaluated the way it would actually be deployed, not with random
leakage across time.

Usage:
    python -m fraudgraph.gnn
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

SAMPLE_PARQUET = Path("data/processed/transactions_sample.parquet")
ALERTS_CSV = Path("data/processed/alerts_gnn.csv")
MODEL_PATH = Path("data/processed/gnn_model.pt")

HIDDEN_DIM = 32
EPOCHS = 30
LR = 0.01
TOP_K = 200

CHANNELS = ["Swipe Transaction", "Chip Transaction", "Online Transaction"]


def load_sample() -> pd.DataFrame:
    df = pd.read_parquet(SAMPLE_PARQUET)
    df["txn_ts"] = pd.to_datetime(
        df["txn_date"].astype(str) + " " + df["txn_time"].astype(str)
    )
    df = df.sort_values("txn_ts").reset_index(drop=True)
    return df


def txn_features(df: pd.DataFrame) -> np.ndarray:
    """Flat per-transaction numeric feature matrix (amount, cyclical time,
    channel one-hot, error flag) — the same features fed into the GraphSAGE
    transaction nodes, minus the graph structure. Shared with baselines.py so
    classical models are compared on identical inputs, not a re-derivation."""
    amount = df["amount"].values.astype(np.float32)
    amount_scaled = np.log1p(np.clip(amount, 0, None))
    amount_scaled = (amount_scaled - amount_scaled.mean()) / (amount_scaled.std() + 1e-6)

    hour = df["txn_ts"].dt.hour.values.astype(np.float32)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    dow = df["txn_ts"].dt.dayofweek.values.astype(np.float32)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    channel_onehot = np.stack(
        [(df["channel"] == c).values.astype(np.float32) for c in CHANNELS], axis=1
    )
    has_error = (df["errors"].notna() & (df["errors"].astype(str).str.strip() != "")).values.astype(np.float32)

    return np.column_stack(
        [amount_scaled, hour_sin, hour_cos, dow_sin, dow_cos, channel_onehot, has_error]
    )  # width = 1+2+2+3+1 = 9


def build_graph(df: pd.DataFrame):
    """Returns (Data, id_maps) where id_maps lets us trace node ids back to
    the original card/merchant/transaction identifiers for the alert output."""
    n_txn = len(df)

    card_keys = (df["user_id"].astype(str) + ":" + df["card_id"].astype(str)).values
    unique_cards = pd.unique(card_keys)
    card_index = {k: i for i, k in enumerate(unique_cards)}

    unique_merchants = df["merchant_id"].unique()
    merchant_index = {m: i for i, m in enumerate(unique_merchants)}

    n_card = len(unique_cards)
    n_merchant = len(unique_merchants)

    # Node layout: [0 .. n_card) cards, [n_card .. n_card+n_txn) transactions,
    # [n_card+n_txn .. n_card+n_txn+n_merchant) merchants
    txn_offset = n_card
    merchant_offset = n_card + n_txn
    n_nodes = n_card + n_txn + n_merchant

    txn_numeric = txn_features(df)
    numeric_width = txn_numeric.shape[1]
    type_width = 3  # card, transaction, merchant one-hot
    feat_width = type_width + numeric_width

    x = np.zeros((n_nodes, feat_width), dtype=np.float32)
    x[:n_card, 0] = 1.0  # card type
    x[txn_offset:merchant_offset, 1] = 1.0  # transaction type
    x[txn_offset:merchant_offset, type_width:] = txn_numeric
    x[merchant_offset:, 2] = 1.0  # merchant type

    # --- edges ---
    card_ids = np.array([card_index[k] for k in card_keys])
    merchant_ids = df["merchant_id"].map(merchant_index).values
    txn_ids = np.arange(n_txn) + txn_offset

    src = np.concatenate([card_ids, txn_ids, merchant_ids, txn_ids])
    dst = np.concatenate([txn_ids, card_ids, txn_ids, merchant_ids])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    y = torch.full((n_nodes,), -1.0, dtype=torch.float)
    y[txn_offset:merchant_offset] = torch.tensor(df["is_fraud"].values.astype(np.float32))

    txn_mask = torch.zeros(n_nodes, dtype=torch.bool)
    txn_mask[txn_offset:merchant_offset] = True

    data = Data(x=torch.tensor(x), edge_index=edge_index, y=y)
    meta = {
        "txn_mask": txn_mask,
        "txn_offset": txn_offset,
        "n_txn": n_txn,
        "df": df,
    }
    return data, meta


def strip_edges(data: Data) -> Data:
    """Ablation: replace the card<->transaction<->merchant edges with pure
    self-loops, so a SAGEConv's neighbor aggregation reduces to the node's
    own feature (no cross-node message passing). Same architecture, same
    node features, same everything except graph structure — isolates what
    the graph topology itself contributes."""
    n_nodes = data.num_nodes
    self_loops = torch.arange(n_nodes, dtype=torch.long)
    edge_index = torch.stack([self_loops, self_loops])
    return Data(x=data.x, edge_index=edge_index, y=data.y)


def temporal_split(n_txn: int, txn_offset: int):
    # NOTE: fraud labels in this sample only occur in the first ~70% of the
    # time-sorted rows (the source data stops injecting fraud before the
    # window ends), so a naive 70/15/15 split leaves the test set with zero
    # positives. Cut earlier so val/test both land inside the fraud-bearing
    # period.
    idx = np.arange(n_txn)
    train_end = int(0.50 * n_txn)
    val_end = int(0.60 * n_txn)
    train_idx = torch.tensor(idx[:train_end] + txn_offset, dtype=torch.long)
    val_idx = torch.tensor(idx[train_end:val_end] + txn_offset, dtype=torch.long)
    test_idx = torch.tensor(idx[val_end:] + txn_offset, dtype=torch.long)
    return train_idx, val_idx, test_idx


class FraudSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.out(h).squeeze(-1)


def train(
    data: Data, train_idx, val_idx, seed: int | None = None,
    epochs: int = EPOCHS, hidden_dim: int = HIDDEN_DIM, lr: float = LR,
) -> FraudSAGE:
    if seed is not None:
        torch.manual_seed(seed)
    model = FraudSAGE(in_dim=data.x.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    y_train = data.y[train_idx]
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1))
    print(f"train txns: {len(train_idx)} ({n_pos} fraud, pos_weight={pos_weight:.1f})")

    best_val_ap = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_idx], data.y[train_idx], pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)[val_idx]
            val_probs = torch.sigmoid(val_logits).numpy()
            val_y = data.y[val_idx].numpy()
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


def evaluate(model: FraudSAGE, data: Data, test_idx, meta: dict) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data.x, data.edge_index))[test_idx].numpy()
    y_test = data.y[test_idx].numpy()

    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    print(f"\nTest ROC-AUC: {auc:.4f}   Test PR-AUC: {ap:.4f}")

    order = np.argsort(-probs)[:TOP_K]
    precision_at_k = y_test[order].mean()
    base_rate = y_test.mean()
    print(f"Precision@{TOP_K}: {precision_at_k:.3f}  (base fraud rate in test split: {base_rate:.4f})")
    if base_rate > 0:
        print(f"Lift over random: {precision_at_k / base_rate:.1f}x")

    df = meta["df"]
    test_row_idx = (test_idx - meta["txn_offset"]).numpy()
    result = df.iloc[test_row_idx][
        ["user_id", "card_id", "merchant_id", "txn_ts", "amount", "is_fraud"]
    ].copy()
    result["fraud_probability"] = probs
    result = result.sort_values("fraud_probability", ascending=False)
    return result


def load_or_train_model(data: Data, meta: dict, train_idx, val_idx) -> FraudSAGE:
    """Load saved weights if they match this graph's feature width, else train fresh."""
    if MODEL_PATH.exists():
        model = FraudSAGE(in_dim=data.x.shape[1])
        try:
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            model.eval()
            print(f"Loaded existing model weights from {MODEL_PATH}")
            return model
        except RuntimeError:
            print(f"Saved weights at {MODEL_PATH} don't match current graph shape — retraining.")
    return train(data, train_idx, val_idx)


def score_all_nodes(model: FraudSAGE, data: Data, meta: dict) -> np.ndarray:
    """Fraud probability for every transaction node, in the same row order as meta['df']."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits)
    txn_offset = meta["txn_offset"]
    n_txn = meta["n_txn"]
    return probs[txn_offset:txn_offset + n_txn].numpy()


def main() -> None:
    df = load_sample()
    print(f"Loaded {len(df)} transactions")

    data, meta = build_graph(df)
    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges, feature dim {data.x.shape[1]}")

    train_idx, val_idx, test_idx = temporal_split(meta["n_txn"], meta["txn_offset"])
    model = train(data, train_idx, val_idx)

    result = evaluate(model, data, test_idx, meta)
    ALERTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.head(TOP_K).to_csv(ALERTS_CSV, index=False)
    print(f"Top {TOP_K} GNN alerts written to {ALERTS_CSV}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model weights saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
