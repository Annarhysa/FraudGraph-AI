"""
Phase 1 of the falsification plan: before claiming GraphSAGE beats XGBoost,
give XGBoost a reasonable tuning budget on the exact same data. Same temporal
split, same flat feature matrix (gnn.txn_features), same evaluation code
(metrics.compute_metrics) as every other experiment in this project — only
the XGBoost hyperparameters change.

Usage:
    python -m fraudgraph.xgboost_tuning
"""
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from . import gnn
from . import metrics as metrics_mod
from .baselines import _load_and_split, _fmt

RESULTS_TXT = Path("results/xgboost_tuning_results.txt")
SEED = 42  # single seed: this is a hyperparameter search, not a robustness sweep

CONFIGS = [
    dict(name="XGB-1", max_depth=3, learning_rate=0.05, n_estimators=300),
    dict(name="XGB-2", max_depth=4, learning_rate=0.05, n_estimators=300),
    dict(name="XGB-3", max_depth=6, learning_rate=0.05, n_estimators=300),
    dict(name="XGB-4", max_depth=4, learning_rate=0.10, n_estimators=300),
    dict(name="XGB-5", max_depth=6, learning_rate=0.10, n_estimators=300),
    dict(name="XGB-6", max_depth=8, learning_rate=0.05, n_estimators=300),
    dict(name="XGB-7", max_depth=6, learning_rate=0.05, n_estimators=500),
    dict(name="XGB-8", max_depth=8, learning_rate=0.05, n_estimators=500),
    # XGB-9: matches the "untuned" baseline's depth (baselines.py uses
    # max_depth=5), which the original 8-config grid never tested — added so
    # the sweep actually covers the point the untuned config sits at.
    dict(name="XGB-9", max_depth=5, learning_rate=0.05, n_estimators=300),
]

# Original (untuned) baseline, for reference — from robustness_results.txt.
CURRENT_XGBOOST_PR_AUC = 0.0277
GRAPHSAGE_100EP_MEAN = 0.0623
GRAPHSAGE_100EP_STD = 0.0117


def run() -> dict:
    df, train_idx, val_idx, test_idx = _load_and_split()
    X = gnn.txn_features(df)
    y = df["is_fraud"].values.astype(int)
    y_train = y[train_idx]
    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    results = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"[{name}] training (max_depth={cfg['max_depth']}, lr={cfg['learning_rate']}, "
              f"n_estimators={cfg['n_estimators']})...")
        model = XGBClassifier(
            max_depth=cfg["max_depth"], learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=SEED, n_jobs=-1,
        )
        model.fit(X[train_idx], y_train)
        df["score__xgb_tuning"] = model.predict_proba(X)[:, 1]
        m = metrics_mod.compute_metrics(df, "score__xgb_tuning", use_split=True)
        results[name] = {**cfg, **m}
        print(f"[{name}] PR-AUC={_fmt(m.get('pr_auc'))}  P@K={_fmt(m.get('precision_at_k'),3)}  "
              f"Lift={_fmt(m.get('lift'),1)}x")

    write_report(results)
    return results


def write_report(results: dict, path: Path = RESULTS_TXT) -> None:
    lines = []
    lines.append("=" * 90)
    lines.append("FraudGraph AI — XGBoost Tuning Sweep (Phase 1: make the comparison fair)")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Goal: before claiming GraphSAGE beats XGBoost, give XGBoost a reasonable tuning")
    lines.append("budget on the exact same data (same temporal split, same flat feature matrix,")
    lines.append("same evaluation code). Single seed (42) — this is a hyperparameter search, not")
    lines.append("a robustness sweep; the winning config should later be re-checked across seeds")
    lines.append("if it becomes the reported baseline.")
    lines.append("")
    lines.append("-" * 90)
    header = f"{'Config':<10}{'max_depth':>11}{'lr':>8}{'n_est':>8}{'PR-AUC':>12}{'ROC-AUC':>10}{'P@K':>8}{'Lift':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    ranked = sorted(results.items(), key=lambda kv: -(kv[1].get("pr_auc") or -1))
    for name, m in ranked:
        lines.append(
            f"{name:<10}{m['max_depth']:>11}{m['learning_rate']:>8}{m['n_estimators']:>8}"
            f"{_fmt(m.get('pr_auc')):>12}{_fmt(m.get('roc_auc')):>10}"
            f"{_fmt(m.get('precision_at_k'),3):>8}"
            f"{(_fmt(m.get('lift'),1)+'x') if m.get('lift') is not None else 'n/a':>10}"
        )

    best_name, best_m = ranked[0]
    lines.append("")
    lines.append("-" * 90)
    lines.append("Comparison against current (untuned) XGBoost and GraphSAGE(100ep, 5-seed)")
    lines.append("-" * 90)
    lines.append(f"  Current (untuned) XGBoost      PR-AUC = {CURRENT_XGBOOST_PR_AUC:.4f}")
    lines.append(f"  Best tuned XGBoost ({best_name:<6})     PR-AUC = {best_m['pr_auc']:.4f}  "
                  f"({'+' if best_m['pr_auc']>=CURRENT_XGBOOST_PR_AUC else ''}"
                  f"{best_m['pr_auc']-CURRENT_XGBOOST_PR_AUC:.4f} vs untuned)")
    lines.append(f"  GraphSAGE 100ep (5-seed mean)   PR-AUC = {GRAPHSAGE_100EP_MEAN:.4f} +/- {GRAPHSAGE_100EP_STD:.4f}")
    lines.append("")

    gap = GRAPHSAGE_100EP_MEAN - best_m["pr_auc"]
    gnn_lower_bound = GRAPHSAGE_100EP_MEAN - GRAPHSAGE_100EP_STD
    lines.append("-" * 90)
    lines.append("Verdict")
    lines.append("-" * 90)
    if best_m["pr_auc"] >= gnn_lower_bound:
        lines.append(f"  Best tuned XGBoost ({best_m['pr_auc']:.4f}) reaches or exceeds GraphSAGE's")
        lines.append(f"  1-std lower bound ({gnn_lower_bound:.4f}) — the GraphSAGE advantage is NOT")
        lines.append("  robust to fair baseline tuning. Do not report GraphSAGE as a clear winner")
        lines.append("  without re-running the multi-seed comparison against this tuned config.")
    else:
        lines.append(f"  Even the best tuned XGBoost ({best_m['pr_auc']:.4f}) falls short of GraphSAGE's")
        lines.append(f"  1-std lower bound ({gnn_lower_bound:.4f}), a gap of {gnn_lower_bound - best_m['pr_auc']:.4f}.")
        lines.append("  Tuning XGBoost within a reasonable budget did not close the gap — the")
        lines.append("  GraphSAGE advantage survives this fairness check.")
    lines.append("")
    lines.append(f"  (Untuned-vs-tuned XGBoost delta: {best_m['pr_auc']-CURRENT_XGBOOST_PR_AUC:+.4f} PR-AUC —")
    lines.append("  for context on how much tuning budget alone was worth here.)")
    lines.append("=" * 90)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    run()
