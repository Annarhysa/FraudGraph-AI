"""
Builds the single master alerts table the whole dashboard reads from:
rule-based signals (anomaly.py) + GraphSAGE score (gnn.py) + graph/network
signals, joined on the same transaction ordering used to build the graph.

Every downstream page (Overview, Alerts, Investigation, Network Explorer,
Fraud Rings, Transactions, Model Performance) reads this one table instead
of recomputing scores, so the numbers stay consistent everywhere.

Usage:
    python -m fraudgraph.scoring
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from . import anomaly as rule
from . import gnn

MASTER_PARQUET = Path("data/processed/alerts_master.parquet")
CALIBRATOR_PATH = Path("data/processed/calibrator.pkl")
SCALE_PATH = Path("data/processed/score_scale.json")

HIGH_RISK_THRESHOLD = 75.0  # gnn_score_pct at/above this counts as "high-risk" for network signals

RAW_COLUMNS = [
    "User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
    "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
]


def risk_level(score_pct: float) -> str:
    if score_pct < 25:
        return "LOW"
    if score_pct < 50:
        return "MEDIUM"
    if score_pct < 75:
        return "HIGH"
    return "CRITICAL"


def add_network_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Graph-derived signals: how connected this transaction's card/merchant
    are to *other* high-risk activity in the sample."""
    df = df.copy()
    df["card_key"] = df["user_id"].astype(str) + ":" + df["card_id"].astype(str)

    high_risk = df[df["gnn_score_pct"] >= HIGH_RISK_THRESHOLD]

    merchant_hr_cards = high_risk.groupby("merchant_id")["card_key"].nunique()
    df["merchant_high_risk_cards"] = df["merchant_id"].map(merchant_hr_cards).fillna(0).astype(int)

    # a card's own high-risk transaction shouldn't count itself as "another" card
    pair_key = df["merchant_id"].astype(str) + "||" + df["card_key"]
    hr_pair_key = set(high_risk["merchant_id"].astype(str) + "||" + high_risk["card_key"])
    is_self_hr_pair = pair_key.isin(hr_pair_key)
    df["merchant_high_risk_cards"] = (
        df["merchant_high_risk_cards"] - is_self_hr_pair.astype(int)
    ).clip(lower=0)

    card_merchants = high_risk.groupby("card_key")["merchant_id"].apply(set)
    merchant_users_hr = high_risk.groupby("merchant_id")["user_id"].apply(set)

    shared_count = {}
    for card_key, merchants in card_merchants.items():
        owner_user = int(card_key.split(":")[0])
        others = set()
        for m in merchants:
            others |= merchant_users_hr.get(m, set())
        others.discard(owner_user)
        shared_count[card_key] = len(others)

    df["card_shared_suspicious_accounts"] = (
        df["card_key"].map(shared_count).fillna(0).astype(int)
    )
    return df


def build_master(force: bool = False) -> pd.DataFrame:
    if MASTER_PARQUET.exists() and not force:
        return pd.read_parquet(MASTER_PARQUET)

    print("Building master alerts table (rule signals + GNN scores + network signals)...")

    # Single canonical ordering (global time sort) — matches the node order
    # gnn.build_graph() uses, so row i in df == transaction node i.
    df = gnn.load_sample()
    df = rule.add_velocity(df)
    df = rule.add_amount_zscore(df)
    df = rule.add_new_merchant(df)
    df = rule.add_merchant_rarity(df)
    df = rule.add_error_flag(df)
    df = rule.score(df)
    df = df.reset_index(drop=True)
    df["transaction_id"] = [f"TXN{i:07d}" for i in range(len(df))]

    data, meta = gnn.build_graph(df)
    train_idx, val_idx, test_idx = gnn.temporal_split(meta["n_txn"], meta["txn_offset"])
    model = gnn.load_or_train_model(data, meta, train_idx, val_idx)
    df["gnn_score_raw"] = gnn.score_all_nodes(model, data, meta)

    split = np.full(len(df), "train", dtype=object)
    split[(val_idx - meta["txn_offset"]).numpy()] = "val"
    split[(test_idx - meta["txn_offset"]).numpy()] = "test"
    df["split"] = split
    # only val/test labels were never fed into the training loss — those are
    # the alerts it's fair to present as genuine "predictions" rather than
    # recollections of what the model was directly trained on.
    df["label_used_in_training"] = df["split"] == "train"

    # The raw sigmoid output is poorly calibrated: heavy class weighting
    # during training (pos_weight ~237x) pushes a large share of the
    # population toward high raw scores even though the *ranking* is good
    # (that's what precision@K measures). Platt-scale using the held-out
    # validation split so displayed risk buckets reflect realistic
    # probabilities. This is a monotonic transform, so it does not change
    # ROC-AUC / PR-AUC / precision@K as already reported in gnn.py.
    val_mask = (df["split"] == "val").values
    calibrator = LogisticRegression()
    calibrator.fit(df.loc[val_mask, ["gnn_score_raw"]].values, df.loc[val_mask, "is_fraud"].values)
    df["gnn_score"] = calibrator.predict_proba(df[["gnn_score_raw"]].values)[:, 1]

    df["rule_score_pct"] = (df["anomaly_score"].clip(0, 1) * 100).round(1)
    # gnn_score is a Platt-calibrated P(fraud), which for a ~0.3% base-rate
    # problem tops out around 5-6% even for the riskiest transactions — not
    # usable as a display scale. gnn_score_pct min-max-stretches it onto
    # 0-100 so the shape (few very risky, most not) is preserved for the
    # analyst UI's risk buckets. It is NOT itself a probability — the real
    # calibrated probability is kept in gnn_score for anyone who wants it.
    s = df["gnn_score"]
    score_min, score_max = float(s.min()), float(s.max())
    df["gnn_score_pct"] = (100 * (s - score_min) / (score_max - score_min)).round(1)
    df["risk_level"] = df["gnn_score_pct"].apply(risk_level)

    # Persist the calibrator + display scale so uploaded transactions (see
    # score_uploaded()) land on the exact same 0-100 scale as this table,
    # instead of fitting an unstable calibration on a small ad-hoc upload.
    CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATOR_PATH, "wb") as f:
        pickle.dump(calibrator, f)
    SCALE_PATH.write_text(json.dumps({"score_min": score_min, "score_max": score_max}))

    df = add_network_signals(df)

    df["status"] = "OPEN"

    MASTER_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MASTER_PARQUET)
    print(f"Master alerts table written: {len(df)} rows -> {MASTER_PARQUET}")
    return df


def parse_uploaded_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Converts a user-uploaded file in the raw IBM TabFormer column schema
    (User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name,
    Merchant City, Merchant State, Zip, MCC[, Errors?, Is Fraud?]) into the
    same internal schema gnn.load_sample() produces, so it can go through
    the identical scoring pipeline as the working sample."""
    missing = [c for c in RAW_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    df = pd.DataFrame({
        "user_id": raw_df["User"],
        "card_id": raw_df["Card"],
        "txn_date": pd.to_datetime(
            dict(year=raw_df["Year"], month=raw_df["Month"], day=raw_df["Day"])
        ),
        "txn_time": raw_df["Time"].astype(str),
        "amount": raw_df["Amount"].astype(str).str.replace("$", "", regex=False).astype(float),
        "channel": raw_df["Use Chip"],
        "merchant_id": raw_df["Merchant Name"],
        "merchant_city": raw_df["Merchant City"],
        "merchant_state": raw_df["Merchant State"],
        "zip": raw_df["Zip"],
        "mcc": raw_df["MCC"],
        "errors": raw_df["Errors?"] if "Errors?" in raw_df.columns else None,
    })
    if "Is Fraud?" in raw_df.columns:
        df["is_fraud"] = raw_df["Is Fraud?"].astype(str).str.strip().eq("Yes")
        df["has_fraud_labels"] = True
    else:
        df["is_fraud"] = False
        df["has_fraud_labels"] = False

    df["txn_ts"] = pd.to_datetime(df["txn_date"].astype(str) + " " + df["txn_time"])
    df = df.sort_values("txn_ts").reset_index(drop=True)
    return df


def score_uploaded(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, object, dict]:
    """Scores an uploaded file with the *already-trained* model and the
    *already-fitted* calibration from build_master() — never trains or
    re-calibrates on the upload itself, since a handful of ad-hoc rows is
    too small/unlabeled a sample to fit anything stable on."""
    if not gnn.MODEL_PATH.exists():
        raise RuntimeError(
            "No trained model found. Run the pipeline first: "
            "python -m src.fraudgraph.scoring"
        )
    if not CALIBRATOR_PATH.exists() or not SCALE_PATH.exists():
        raise RuntimeError(
            "No saved score calibration found. Run: python -m src.fraudgraph.scoring"
        )

    df = parse_uploaded_transactions(raw_df)
    if len(df) == 0:
        raise ValueError("Uploaded file has no rows.")

    df = rule.add_velocity(df)
    df = rule.add_amount_zscore(df)
    df = rule.add_new_merchant(df)
    df = rule.add_merchant_rarity(df)
    df = rule.add_error_flag(df)
    df = rule.score(df)
    df["transaction_id"] = [f"UPLOAD{i:06d}" for i in range(len(df))]

    data, meta = gnn.build_graph(df)
    model = gnn.FraudSAGE(in_dim=data.x.shape[1])
    model.load_state_dict(torch.load(gnn.MODEL_PATH, weights_only=True))
    model.eval()
    df["gnn_score_raw"] = gnn.score_all_nodes(model, data, meta)

    with open(CALIBRATOR_PATH, "rb") as f:
        calibrator = pickle.load(f)
    scale = json.loads(SCALE_PATH.read_text())

    df["gnn_score"] = calibrator.predict_proba(df[["gnn_score_raw"]].values)[:, 1]
    score_min, score_max = scale["score_min"], scale["score_max"]
    df["gnn_score_pct"] = (
        100 * (df["gnn_score"] - score_min) / (score_max - score_min)
    ).clip(0, 100).round(1)
    df["rule_score_pct"] = (df["anomaly_score"].clip(0, 1) * 100).round(1)
    df["risk_level"] = df["gnn_score_pct"].apply(risk_level)

    df = add_network_signals(df)
    df["status"] = "OPEN"
    df["split"] = "upload"
    df["label_used_in_training"] = False

    return df, data, meta


def main() -> None:
    df = build_master(force=True)
    print(df["risk_level"].value_counts())
    print(f"\nHigh-risk+ transactions: {(df['gnn_score_pct'] >= 50).sum()}")


if __name__ == "__main__":
    main()
