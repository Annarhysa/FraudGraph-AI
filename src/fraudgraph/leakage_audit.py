"""
Phase 2 (hard gate): does the merchant-only result depend on test-period
transactions leaking information to each other through their shared merchant
hub node?

Mechanism: in the standard (transductive) graph, a merchant hub's embedding
is a message-passing aggregate over ALL of that merchant's transactions,
train/val/test alike. A test transaction being scored therefore receives an
embedding that was partly computed FROM other test-split transactions at the
same merchant — including transactions that, chronologically, may not have
happened yet relative to the one being scored. That's not label leakage
(only features flow, not is_fraud), but it is a transductive-batch-scoring
assumption that would not hold for pure real-time-only deployment, and it
could inflate the merchant-only PR-AUC if the model leans on it.

This test rebuilds the merchant-only graph with gnn.build_graph's
source_mask: transactions can only feed their features INTO a hub if they're
in TRAIN or VAL (source_mask = split != 'test'); every transaction, including
test ones, still RECEIVES its hub's embedding. So a test transaction's score
can only be informed by what was already known before the test period, never
by other test-split transactions. Comparing this "causal" PR-AUC to the
original transductive 0.2063 +/- 0.0244 tells us how much of the merchant
signal is legitimate (learned from train/val, generalizes forward) vs
dependent on test-test leakage.

Usage:
    python -m fraudgraph.leakage_audit
"""
from pathlib import Path

import numpy as np

from . import gnn
from . import metrics as metrics_mod
from .baselines import _load_and_split, _fmt

RESULTS_TXT = Path("results/leakage_audit_results.txt")
SEEDS = [0, 1, 2, 3, 4]

TRANSDUCTIVE_REFERENCE = (0.0858, 0.0098)  # merchant-only, Window 0, CORRECTED (edge_ablation_results.txt, post merchant_offset bugfix)


def run() -> list:
    df, train_idx, val_idx, test_idx = _load_and_split()
    n_txn = len(df)

    # source_mask: True for train+val rows only (test rows may not feed their
    # hub's embedding; they may still receive it).
    source_mask = np.ones(n_txn, dtype=bool)
    source_mask[test_idx] = False
    print(f"source_mask: {source_mask.sum()} allowed sources (train+val), "
          f"{(~source_mask).sum()} test rows excluded from hub aggregation")

    data, meta = gnn.build_graph(df, include_card_edges=False, include_merchant_edges=True, source_mask=source_mask)
    g_train_idx, g_val_idx, g_test_idx = gnn.temporal_split(meta["n_txn"], meta["txn_offset"])

    results = []
    for seed in SEEDS:
        print(f"[causal merchant-only] seed {seed} -> training...")
        model = gnn.train(data, g_train_idx, g_val_idx, seed=seed, epochs=100, hidden_dim=32, lr=0.01)
        scores = gnn.score_all_nodes(model, data, meta)
        df["score__causal_merchant"] = scores
        m = metrics_mod.compute_metrics(df, "score__causal_merchant", use_split=True)
        results.append(m)
        print(f"[causal merchant-only] seed {seed} -> PR-AUC={_fmt(m.get('pr_auc'))}  "
              f"P@K={_fmt(m.get('precision_at_k'),3)}")

    write_report(results)
    return results


def write_report(results: list, path: Path = RESULTS_TXT) -> None:
    pr_vals = [m.get("pr_auc") for m in results if m.get("pr_auc") is not None]
    pk_vals = [m.get("precision_at_k") for m in results if m.get("precision_at_k") is not None]
    pr_mean, pr_std = float(np.mean(pr_vals)), float(np.std(pr_vals))
    pk_mean, pk_std = float(np.mean(pk_vals)), float(np.std(pk_vals))

    trans_mean, trans_std = TRANSDUCTIVE_REFERENCE

    lines = []
    lines.append("=" * 92)
    lines.append("FraudGraph AI — Leakage Audit (Phase 2 hard gate): causal vs transductive merchant graph")
    lines.append("=" * 92)
    lines.append("")
    lines.append("Setup: merchant-only graph, Window 0, hidden_dim=32, epochs=100, lr=0.01, seeds=0-4.")
    lines.append("CAUSAL variant: test-split transactions cannot feed their features into their")
    lines.append("merchant hub's embedding (source_mask excludes test rows); they still receive")
    lines.append("their hub's embedding, computed only from train+val history. This removes any")
    lines.append("test-test (or future-within-test) information flow through the merchant hub.")
    lines.append("")
    lines.append("-" * 92)
    lines.append(f"{'Condition':<40}{'PR-AUC':>20}{'Precision@K':>20}")
    lines.append("-" * 80)
    lines.append(f"{'Transductive (original, all rows can be hub sources)':<40}{_fmt(trans_mean)+' +/- '+_fmt(trans_std):>20}{'0.069 +/- 0.035':>20}")
    lines.append(f"{'Causal (test rows excluded as hub sources)':<40}{_fmt(pr_mean)+' +/- '+_fmt(pr_std):>20}{_fmt(pk_mean,3)+' +/- '+_fmt(pk_std,3):>20}")
    lines.append("")
    lines.append("Per-seed PR-AUC (causal):")
    for seed, m in zip(SEEDS, results):
        lines.append(f"  seed {seed}: {_fmt(m.get('pr_auc'))}")

    lines.append("")
    lines.append("-" * 92)
    lines.append("Verdict")
    lines.append("-" * 92)
    retained_pct = 100 * pr_mean / trans_mean if trans_mean else None
    drop = trans_mean - pr_mean
    lines.append(f"  Transductive: {trans_mean:.4f} +/- {trans_std:.4f}")
    lines.append(f"  Causal:       {pr_mean:.4f} +/- {pr_std:.4f}")
    lines.append(f"  Drop: {drop:+.4f} PR-AUC ({retained_pct:.1f}% of the transductive result retained)")
    lines.append("")
    # 1-std overlap check
    causal_upper = pr_mean + pr_std
    trans_lower = trans_mean - trans_std
    if causal_upper >= trans_lower:
        lines.append("  -> The causal result's 1-std range overlaps the transductive result's 1-std")
        lines.append("     lower bound: the merchant signal is NOT meaningfully dependent on test-test")
        lines.append("     information flow. The advantage is legitimate — it comes from merchant")
        lines.append("     reputation learned in train/val and correctly generalized forward to test.")
    elif retained_pct is not None and retained_pct >= 70:
        lines.append("  -> The causal result is lower but still retains the large majority of the")
        lines.append("     transductive signal. Most of the merchant advantage is legitimate")
        lines.append("     (train/val-learned reputation), with a smaller leakage-attributable component")
        lines.append("     worth disclosing as a limitation rather than treated as zero.")
    else:
        lines.append("  -> The causal result drops substantially from the transductive one: a material")
        lines.append("     share of the reported merchant-only advantage depends on test-split")
        lines.append("     transactions informing each other through the shared merchant hub, which is")
        lines.append("     NOT available in a strict real-time deployment. Report the CAUSAL number as")
        lines.append(f"     the defensible one going forward, not the transductive {trans_mean:.4f}.")
    lines.append("")
    lines.append("Context from the non-training analysis (see conversation): 48/49 test-period fraud")
    lines.append("merchants already had fraud in train, and 44/49 also in val — i.e. most test fraud")
    lines.append("sits at merchants whose risk was already learnable before the test period began.")
    lines.append("Several merchants show extreme fraud concentration (e.g. 79/87, 108/128, 159/204")
    lines.append("transactions fraudulent) — a very strong, legitimately-learnable reputation signal")
    lines.append("independent of any test-test leakage pathway.")
    lines.append("=" * 92)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    run()
