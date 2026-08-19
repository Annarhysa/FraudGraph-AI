"""
Graph-analytics + rule-based anomaly scoring — no trained model required.

This is the "baseline" milestone: turn the transaction graph into a
per-transaction suspicion score using signals a fraud analyst would
recognize, each with a human-readable reason. It also gives us a sanity
check (precision@K against the real `is_fraud` label) before we invest in
a GNN.

Signals:
    - velocity      : transactions by the same card in the trailing hour
    - amount_zscore : how unusual the amount is vs. that user's own history
    - new_merchant  : first time this user has transacted with this merchant
    - merchant_rarity: merchant seen by very few distinct users (small/obscure)
    - has_error     : transaction had a swipe/chip error flag set

Usage:
    python -m fraudgraph.anomaly
"""
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_PARQUET = Path("data/processed/transactions_sample.parquet")
ALERTS_CSV = Path("data/processed/alerts.csv")

TOP_K = 200


def load_sample() -> pd.DataFrame:
    df = pd.read_parquet(SAMPLE_PARQUET)
    df["txn_ts"] = pd.to_datetime(
        df["txn_date"].astype(str) + " " + df["txn_time"].astype(str)
    )
    df = df.sort_values(["user_id", "card_id", "txn_ts"]).reset_index(drop=True)
    return df


def add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    def _velocity(group: pd.DataFrame) -> pd.Series:
        ts = group["txn_ts"].values.astype("datetime64[s]").astype("int64")
        counts = np.zeros(len(ts), dtype=int)
        start = 0
        for i in range(len(ts)):
            while ts[i] - ts[start] > 3600:
                start += 1
            counts[i] = i - start  # transactions in the preceding hour, excluding itself
        return pd.Series(counts, index=group.index)

    df["velocity_1h"] = df.groupby(["user_id", "card_id"], group_keys=False).apply(_velocity)
    return df


def add_amount_zscore(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby("user_id")["amount"].agg(["mean", "std"]).rename(
        columns={"mean": "user_mean", "std": "user_std"}
    )
    df = df.merge(stats, on="user_id", how="left")
    df["user_std"] = df["user_std"].replace(0, np.nan)
    df["amount_zscore"] = (df["amount"] - df["user_mean"]) / df["user_std"]
    df["amount_zscore"] = df["amount_zscore"].fillna(0)
    return df


def add_new_merchant(df: pd.DataFrame) -> pd.DataFrame:
    seen = set()
    flags = np.zeros(len(df), dtype=bool)
    # df is sorted by user/card/time, not global time, so recompute a global sort
    order = df.sort_values("txn_ts").index
    seen_pairs = set()
    for idx in order:
        key = (df.at[idx, "user_id"], df.at[idx, "merchant_id"])
        flags[df.index.get_loc(idx)] = key not in seen_pairs
        seen_pairs.add(key)
    df["new_merchant"] = flags
    return df


def add_merchant_rarity(df: pd.DataFrame) -> pd.DataFrame:
    merchant_users = df.groupby("merchant_id")["user_id"].nunique()
    df["merchant_user_count"] = df["merchant_id"].map(merchant_users)
    df["merchant_rare"] = df["merchant_user_count"] <= 2
    return df


def add_error_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["has_error"] = df["errors"].notna() & (df["errors"].astype(str).str.strip() != "")
    return df


def score(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize each raw signal to ~[0, 1] and combine into one composite score.
    velocity_score = np.tanh(df["velocity_1h"] / 5.0)
    amount_score = np.tanh(np.abs(df["amount_zscore"]) / 3.0)
    new_merchant_score = df["new_merchant"].astype(float) * 0.5
    rarity_score = df["merchant_rare"].astype(float) * 0.3
    error_score = df["has_error"].astype(float) * 0.4

    df["anomaly_score"] = (
        0.35 * velocity_score
        + 0.30 * amount_score
        + 0.15 * new_merchant_score
        + 0.10 * rarity_score
        + 0.10 * error_score
    )
    return df


def explain(row: pd.Series) -> list[str]:
    reasons = []
    if row["velocity_1h"] >= 3:
        reasons.append(f"{int(row['velocity_1h'])} other transactions on this card in the last hour")
    if abs(row["amount_zscore"]) >= 2:
        reasons.append(f"Amount is {abs(row['amount_zscore']):.1f}x this user's normal spending pattern")
    if row["new_merchant"]:
        reasons.append("First transaction between this user and this merchant")
    if row["merchant_rare"]:
        reasons.append(f"Merchant seen for only {int(row['merchant_user_count'])} distinct user(s) in the sample")
    if row["has_error"]:
        reasons.append(f"Transaction flagged with error: {row['errors']}")
    if not reasons:
        reasons.append("No single dominant factor — flagged on combined weak signals")
    return reasons


def evaluate(df: pd.DataFrame, k: int = TOP_K) -> None:
    top = df.sort_values("anomaly_score", ascending=False).head(k)
    precision = top["is_fraud"].mean()
    base_rate = df["is_fraud"].mean()
    print(f"\nPrecision@{k}: {precision:.3f}  (base fraud rate in sample: {base_rate:.4f})")
    print(f"Lift over random: {precision / base_rate:.1f}x" if base_rate > 0 else "")


def main() -> None:
    df = load_sample()
    print(f"Loaded {len(df)} transactions for scoring")

    df = add_velocity(df)
    df = add_amount_zscore(df)
    df = add_new_merchant(df)
    df = add_merchant_rarity(df)
    df = add_error_flag(df)
    df = score(df)

    evaluate(df)

    top = df.sort_values("anomaly_score", ascending=False).head(TOP_K).copy()
    top["reasons"] = top.apply(explain, axis=1)
    top["reasons_text"] = top["reasons"].apply(lambda r: " | ".join(r))

    out_cols = [
        "user_id", "card_id", "merchant_id", "txn_ts", "amount",
        "anomaly_score", "is_fraud", "reasons_text",
    ]
    ALERTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    top[out_cols].to_csv(ALERTS_CSV, index=False)
    print(f"\nTop {TOP_K} alerts written to {ALERTS_CSV}")


if __name__ == "__main__":
    main()
