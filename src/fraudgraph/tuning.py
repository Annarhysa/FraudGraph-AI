"""
Follow-up to baselines.py's robustness sweep: the default GraphSAGE config
(30 epochs, hidden_dim=32) lost to XGBoost in 5/5 seeds, but val AP was still
climbing sharply at epoch 30 in every run — a sign of under-training rather
than a ceiling. This retrains GraphSAGE with a larger budget (more epochs,
wider hidden layer) across multiple seeds and checks whether it closes the
gap with XGBoost (fixed at PR-AUC 0.0277, see robustness_results.txt).

Usage:
    python -m fraudgraph.tuning
"""
from pathlib import Path

import numpy as np

from . import gnn
from . import metrics as metrics_mod
from .baselines import _load_and_split, _fmt

RESULTS_TXT = Path("results/tuning_results.txt")
SEEDS = [0, 1, 2]

# Reference numbers from results/robustness_results.txt (5-seed sweep,
# default config: 30 epochs, hidden_dim=32).
XGBOOST_PR_AUC = 0.0277  # deterministic across seeds in that sweep
DEFAULT_GNN_MEAN_PR_AUC = 0.0172
DEFAULT_GNN_STD_PR_AUC = 0.0018

CONFIGS = {
    "GraphSAGE (default: 30ep, h=32)": dict(epochs=30, hidden_dim=32, lr=0.01),
    "GraphSAGE (100ep, h=32)": dict(epochs=100, hidden_dim=32, lr=0.01),
    "GraphSAGE (100ep, h=64)": dict(epochs=100, hidden_dim=64, lr=0.01),
    "GraphSAGE (200ep, h=64)": dict(epochs=200, hidden_dim=64, lr=0.005),
}


def run() -> None:
    df, train_idx, val_idx, test_idx = _load_and_split()
    data, meta = gnn.build_graph(df)
    g_train_idx, g_val_idx, g_test_idx = gnn.temporal_split(meta["n_txn"], meta["txn_offset"])

    results = {name: [] for name in CONFIGS}  # name -> list of metric dicts per seed

    for name, cfg in CONFIGS.items():
        for seed in SEEDS:
            print(f"[{name}] seed {seed} -> training ({cfg})...")
            model = gnn.train(data, g_train_idx, g_val_idx, seed=seed, **cfg)
            df[f"score__tuning"] = gnn.score_all_nodes(model, data, meta)
            m = metrics_mod.compute_metrics(df, "score__tuning", use_split=True)
            results[name].append(m)
            print(f"[{name}] seed {seed} -> PR-AUC={_fmt(m.get('pr_auc'))}")

    write_report(results)


def write_report(results: dict, path: Path = RESULTS_TXT) -> None:
    lines = []
    lines.append("=" * 86)
    lines.append("FraudGraph AI — GraphSAGE Tuning Sweep (does more capacity close the XGBoost gap?)")
    lines.append("=" * 86)
    lines.append("")
    lines.append("Motivation: the 5-seed robustness sweep (robustness_results.txt) found XGBoost")
    lines.append(f"beat default GraphSAGE (30 epochs, hidden_dim=32) in 5/5 seeds "
                  f"(PR-AUC {XGBOOST_PR_AUC:.4f} vs "
                  f"{DEFAULT_GNN_MEAN_PR_AUC:.4f} +/- {DEFAULT_GNN_STD_PR_AUC:.4f}), but GraphSAGE's")
    lines.append("validation AP was still rising sharply at the epoch-30 cutoff in every seed —")
    lines.append("suggesting under-training rather than a real ceiling. This sweep retrains")
    lines.append(f"GraphSAGE with larger budgets across {len(SEEDS)} seeds ({SEEDS}) to check.")
    lines.append("")
    lines.append("-" * 86)
    header = f"{'Config':<34}{'PR-AUC (mean +/- std)':>26}{'Precision@K':>16}{'vs XGBoost':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, per_seed in results.items():
        pr_vals = [m["pr_auc"] for m in per_seed if m.get("pr_auc") is not None]
        pk_vals = [m["precision_at_k"] for m in per_seed if m.get("precision_at_k") is not None]
        pr_mean, pr_std = (float(np.mean(pr_vals)), float(np.std(pr_vals))) if pr_vals else (None, None)
        pk_mean = float(np.mean(pk_vals)) if pk_vals else None
        verdict = "n/a"
        if pr_mean is not None:
            verdict = "BEATS" if pr_mean > XGBOOST_PR_AUC else "behind"
        pr_str = f"{_fmt(pr_mean)} +/- {_fmt(pr_std)}" if pr_mean is not None else "n/a"
        pk_str = f"{_fmt(pk_mean, 3)}" if pk_mean is not None else "n/a"
        lines.append(f"{name:<34}{pr_str:>26}{pk_str:>16}{verdict:>10}")

    lines.append("")
    lines.append(f"Reference: XGBoost PR-AUC = {XGBOOST_PR_AUC:.4f} (deterministic, from robustness_results.txt)")
    lines.append("")
    lines.append("-" * 86)
    lines.append("Per-seed PR-AUC")
    lines.append("-" * 86)
    seed_header = f"{'Config':<34}" + "".join(f"{'seed '+str(s):>12}" for s in SEEDS)
    lines.append(seed_header)
    for name, per_seed in results.items():
        row = f"{name:<34}"
        for m in per_seed:
            row += f"{_fmt(m.get('pr_auc')):>12}"
        lines.append(row)

    lines.append("")
    lines.append("-" * 86)
    lines.append("Conclusion")
    lines.append("-" * 86)
    best_name, best_mean = None, -1
    for name, per_seed in results.items():
        pr_vals = [m["pr_auc"] for m in per_seed if m.get("pr_auc") is not None]
        if pr_vals and np.mean(pr_vals) > best_mean:
            best_mean = np.mean(pr_vals)
            best_name = name
    if best_mean > XGBOOST_PR_AUC:
        lines.append(f"  Best config ('{best_name}', mean PR-AUC {best_mean:.4f}) SURPASSES XGBoost")
        lines.append(f"  ({XGBOOST_PR_AUC:.4f}) — under-training was the real cause of the earlier gap;")
        lines.append("  with enough capacity/budget, the graph signal wins out.")
    else:
        lines.append(f"  Best config ('{best_name}', mean PR-AUC {best_mean:.4f}) still falls short of")
        lines.append(f"  XGBoost ({XGBOOST_PR_AUC:.4f}). More epochs/width alone doesn't close the gap —")
        lines.append("  the deficit is more likely architectural (SAGEConv's mean/max aggregation vs.")
        lines.append("  XGBoost's ability to carve sharp feature-interaction splits) or needs different")
        lines.append("  levers: better negative sampling, focal loss, attention-based aggregation")
        lines.append("  (GAT), or more graph context (multi-hop, more of the temporal history).")
    lines.append("=" * 86)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTuning sweep written to {path}")


if __name__ == "__main__":
    run()
