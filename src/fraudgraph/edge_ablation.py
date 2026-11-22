"""
Phase 4: edge-type ablation. The graph is Card <-> Transaction <-> Merchant.
The full-graph vs no-graph ablation (results/graphsage_100ep_5seed_results.txt)
already established that SOME graph signal matters. This isolates WHICH
relationship type it comes from: card-transaction edges only, merchant-
transaction edges only, both (full graph, the control), or neither.

Node features are identical across all four conditions (gnn.build_graph keeps
x/y/node-count/order fixed regardless of include_card_edges/
include_merchant_edges) — only which edges exist changes. That's what makes
the resulting PR-AUC deltas attributable to the edge type, not confounded by
different features.

Frozen config (per instruction, no re-tuning here): hidden_dim=32, epochs=100,
lr=0.01, seeds=[0,1,2,3,4], Window 0 (the original default 50/10/40 split).

Usage:
    python -m fraudgraph.edge_ablation
"""
from pathlib import Path

import numpy as np

from . import gnn
from . import metrics as metrics_mod
from .baselines import _load_and_split, _fmt

RESULTS_TXT = Path("results/edge_ablation_results.txt")
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 100
HIDDEN_DIM = 32

CONDITIONS = {
    "Full Graph (Card + Merchant)": dict(include_card_edges=True, include_merchant_edges=True),
    "Card-only (Card)":             dict(include_card_edges=True, include_merchant_edges=False),
    "Merchant-only (Merchant)":     dict(include_card_edges=False, include_merchant_edges=True),
    "No Graph (None)":              dict(include_card_edges=False, include_merchant_edges=False, strip=True),
}

# Card-only and No Graph don't touch merchant_ids at all, so they are
# unaffected by the merchant_offset bugfix in gnn.build_graph — reuse the
# already-measured values instead of re-training them.
UNCHANGED_REFERENCE = {
    "Card-only (Card)": [
        {"pr_auc": v} for v in []  # per-seed values not needed; mean/std below suffice
    ],
}
UNCHANGED_MEAN_STD = {
    "Card-only (Card)": dict(pr_auc=(0.0113, 0.0004)),
    "No Graph (None)": dict(pr_auc=(0.0151, 0.0003)),
}


def run(condition_names: list[str] | None = None) -> dict:
    df, train_idx, val_idx, test_idx = _load_and_split()

    names = condition_names or list(CONDITIONS.keys())
    results = {name: [] for name in names}
    for name in names:
        cfg = CONDITIONS[name]
        strip = cfg.get("strip", False)
        data, meta = gnn.build_graph(
            df,
            include_card_edges=cfg["include_card_edges"],
            include_merchant_edges=cfg["include_merchant_edges"],
        )
        if strip:
            # "No Graph": reuse the established self-loops-only ablation
            # method (same as graphsage_100ep_5seed_results.txt) rather than
            # an empty edge_index, for continuity with that prior result.
            data = gnn.strip_edges(data)
        g_train_idx, g_val_idx, g_test_idx = gnn.temporal_split(meta["n_txn"], meta["txn_offset"])

        for seed in SEEDS:
            print(f"[{name}] seed {seed} -> training (epochs={EPOCHS}, hidden_dim={HIDDEN_DIM})...")
            model = gnn.train(data, g_train_idx, g_val_idx, seed=seed, epochs=EPOCHS, hidden_dim=HIDDEN_DIM, lr=0.01)
            scores = gnn.score_all_nodes(model, data, meta)
            df[f"score__{name}"] = scores
            m = metrics_mod.compute_metrics(df, f"score__{name}", use_split=True)
            results[name].append(m)
            print(f"[{name}] seed {seed} -> PR-AUC={_fmt(m.get('pr_auc'))}  P@K={_fmt(m.get('precision_at_k'),3)}")

    write_report(results)
    return results


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def write_report(results: dict, path: Path = RESULTS_TXT) -> None:
    lines = []
    lines.append("=" * 92)
    lines.append("FraudGraph AI — Edge-Type Ablation (Phase 4): which relationship drives the signal?")
    lines.append("=" * 92)
    lines.append("")
    lines.append(f"Fixed config: hidden_dim={HIDDEN_DIM}, epochs={EPOCHS}, lr=0.01, seeds={SEEDS}")
    lines.append("Window 0 (original default 50/10/40 temporal split). Node features identical")
    lines.append("across all four conditions — only which edge types are wired into the graph")
    lines.append("changes (gnn.build_graph include_card_edges / include_merchant_edges flags).")
    lines.append("")
    lines.append("*** CORRECTED RESULTS: gnn.build_graph had a bug where merchant node ids were")
    lines.append("never shifted by merchant_offset, so 'merchant edges' never reached real merchant")
    lines.append("nodes (verified: 0 edges touched the merchant range before the fix). Full Graph and")
    lines.append("Merchant-only below are RE-RUN with the fix. Card-only and No Graph are unaffected")
    lines.append("by this bug (cards sit at node-id offset 0; No Graph uses self-loops only) and are")
    lines.append("reused from the original run rather than re-trained.")
    lines.append("")
    lines.append("-" * 92)
    header = f"{'Condition':<32}{'PR-AUC':>20}{'Precision@K':>20}{'Lift':>18}"
    lines.append(header)
    lines.append("-" * len(header))

    summary = {}
    for name, per_seed in results.items():
        pr_m, pr_s = _mean_std([m.get("pr_auc") for m in per_seed])
        pk_m, pk_s = _mean_std([m.get("precision_at_k") for m in per_seed])
        lift_m, lift_s = _mean_std([m.get("lift") for m in per_seed])
        summary[name] = (pr_m, pr_s, pk_m, pk_s, lift_m, lift_s)
    # Fill in any conditions that weren't re-run (unaffected by the merchant
    # offset bugfix) from the already-measured reference values.
    for name, cfg in CONDITIONS.items():
        if name not in summary and name in UNCHANGED_MEAN_STD:
            pr_m, pr_s = UNCHANGED_MEAN_STD[name]["pr_auc"]
            summary[name] = (pr_m, pr_s, None, None, None, None)

    ranked = sorted(summary.items(), key=lambda kv: -(kv[1][0] or -1))
    for name, (pr_m, pr_s, pk_m, pk_s, lift_m, lift_s) in ranked:
        pr_str = f"{_fmt(pr_m)} +/- {_fmt(pr_s)}" if pr_m is not None else "n/a"
        pk_str = f"{_fmt(pk_m,3)} +/- {_fmt(pk_s,3)}" if pk_m is not None else "n/a"
        lift_str = f"{_fmt(lift_m,1)}x +/- {_fmt(lift_s,1)}" if lift_m is not None else "n/a"
        lines.append(f"{name:<32}{pr_str:>20}{pk_str:>20}{lift_str:>18}")

    lines.append("")
    lines.append("-" * 92)
    lines.append("Per-seed PR-AUC")
    lines.append("-" * 92)
    seed_header = f"{'Condition':<32}" + "".join(f"{'seed '+str(s):>12}" for s in SEEDS)
    lines.append(seed_header)
    for name, per_seed in results.items():
        row = f"{name:<32}"
        for m in per_seed:
            row += f"{_fmt(m.get('pr_auc')):>12}"
        lines.append(row)

    # --- decomposition ---
    lines.append("")
    lines.append("-" * 92)
    lines.append("Decomposition")
    lines.append("-" * 92)
    full = summary.get("Full Graph (Card + Merchant)", (None,))[0]
    card = summary.get("Card-only (Card)", (None,))[0]
    merch = summary.get("Merchant-only (Merchant)", (None,))[0]
    none_ = summary.get("No Graph (None)", (None,))[0]
    if all(v is not None for v in [full, card, merch, none_]):
        card_gain = card - none_
        merch_gain = merch - none_
        full_gain = full - none_
        lines.append(f"  No graph baseline:               {none_:.4f}")
        lines.append(f"  Card-only gain over no-graph:     {card_gain:+.4f}  ({card:.4f} total)")
        lines.append(f"  Merchant-only gain over no-graph: {merch_gain:+.4f}  ({merch:.4f} total)")
        lines.append(f"  Full graph gain over no-graph:    {full_gain:+.4f}  ({full:.4f} total)")
        lines.append(f"  Sum of individual gains:          {card_gain + merch_gain:+.4f}")
        lines.append(f"  Full graph vs sum of individual:  {full_gain - (card_gain + merch_gain):+.4f}")
        lines.append("")
        dominant = "card" if card_gain > merch_gain else "merchant"
        lines.append(f"  -> The larger individual contributor is: {dominant}-transaction edges "
                      f"({max(card_gain, merch_gain):.4f} vs {min(card_gain, merch_gain):.4f}).")
        if full_gain > (card_gain + merch_gain) * 1.05:
            lines.append("  -> Full graph exceeds the sum of individual gains: card and merchant")
            lines.append("     relationships appear to provide COMPLEMENTARY information beyond either alone.")
        elif full_gain < (card_gain + merch_gain) * 0.95:
            lines.append("  -> Full graph falls short of the sum of individual gains: there is overlap/")
            lines.append("     redundancy between what card and merchant relationships provide.")
        else:
            lines.append("  -> Full graph gain is roughly additive: card and merchant relationships")
            lines.append("     appear to contribute largely independent, non-overlapping signal.")
    else:
        lines.append("  (one or more conditions missing PR-AUC — check per-seed table above)")
    lines.append("=" * 92)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    # Only re-run the two conditions affected by the merchant_offset bugfix;
    # Card-only and No Graph are reused via UNCHANGED_MEAN_STD.
    run(condition_names=["Full Graph (Card + Merchant)", "Merchant-only (Merchant)"])
