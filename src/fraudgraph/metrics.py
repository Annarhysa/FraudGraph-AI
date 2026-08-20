"""
Real evaluation metrics — no hardcoded/fabricated numbers.

Two contexts this is used in:
  - the shared sample (`use_split=True`, the default): metrics computed on
    the temporal test split only, matching gnn.py's train/val/test methodology.
  - a user-uploaded batch (`use_split=False`): there's no train/test split
    for an ad-hoc upload — every row is fresh, unseen-by-training data, so
    metrics are computed over the whole batch if it has fraud labels.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

TOP_K = 200


def _select(df: pd.DataFrame, use_split: bool) -> pd.DataFrame:
    return df[df["split"] == "test"] if use_split else df


def compute_metrics(df: pd.DataFrame, score_col: str, k: int = TOP_K, use_split: bool = True) -> dict:
    subset = _select(df, use_split)
    y = subset["is_fraud"].values.astype(int)
    s = subset[score_col].values

    if len(y) == 0 or y.sum() == 0:
        return {"roc_auc": None, "pr_auc": None, "precision_at_k": None, "lift": None,
                "n_test": len(subset), "k": None}

    k_eff = min(k, len(y))
    roc_auc = roc_auc_score(y, s)
    pr_auc = average_precision_score(y, s)

    order = np.argsort(-s)[:k_eff]
    precision_at_k = y[order].mean()
    base_rate = y.mean()
    lift = precision_at_k / base_rate if base_rate > 0 else None

    return {
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "precision_at_k": precision_at_k, "lift": lift,
        "n_test": len(subset), "base_rate": base_rate, "k": k_eff,
    }


def model_comparison(df: pd.DataFrame, k: int = TOP_K, use_split: bool = True) -> dict:
    # Rank on the full-precision scores, not the rounded *_pct display columns —
    # rounding to 1 decimal creates ties that make precision@K non-deterministic.
    return {
        "Rule-based": compute_metrics(df, "anomaly_score", k, use_split),
        "GraphSAGE": compute_metrics(df, "gnn_score", k, use_split),
    }


def curves(df: pd.DataFrame, score_col: str, use_split: bool = True):
    subset = _select(df, use_split)
    y = subset["is_fraud"].values.astype(int)
    s = subset[score_col].values
    if len(y) == 0 or y.sum() == 0:
        return None
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}
