"""
Phase 1 (validate the merchant-only finding): the edge-type ablation on
Window 0 found Merchant-only (0.2063 +/- 0.0244) massively beats Full Graph
(0.0623 +/- 0.0117), while Card-only (0.0113) barely beats No Graph (0.0151).
Before trusting that, check whether it holds across the 3 additional
chronological windows from temporal_windows.py (same train/val/test cuts,
same frozen config, same 3-seed methodology used there).

Usage:
    python -m fraudgraph.temporal_windows_merchant
"""
from pathlib import Path

import numpy as np

from . import gnn
from . import metrics as metrics_mod
from .baselines import _load_and_split, _fmt
from .temporal_windows import WINDOWS, SEEDS, _make_split_col, torch_arange_offset

RESULTS_TXT = Path("results/temporal_robustness_merchant_results.txt")

# Reference numbers already established (Window 0 from edge_ablation_results.txt,
# Windows 1-3 XGBoost/Full/No-graph from temporal_robustness_results.txt).
REFERENCE = {
    "Window 0 (original default split)": {
        "XGBoost (depth=5)": (0.0277, None),
        "GraphSAGE 100ep (full graph)": (0.0623, 0.0117),
        "GraphSAGE 100ep (no graph)": (0.0151, 0.0003),
        "GraphSAGE 100ep (merchant-only)": (0.2063, 0.0244),  # already have this one
    },
    "Window 1 (train Jan-Apr, val May, test Jun-Oct)": {
        "XGBoost (depth=5)": (0.0445, None),
        "GraphSAGE 100ep (full graph)": (0.1237, 0.0157),
        "GraphSAGE 100ep (no graph)": (0.0368, 0.0004),
    },
    "Window 2 (train Jan-Jun, val Jul, test Aug-Oct)": {
        "XGBoost (depth=5)": (0.0562, None),
        "GraphSAGE 100ep (full graph)": (0.1770, 0.0099),
        "GraphSAGE 100ep (no graph)": (0.0445, 0.0013),
    },
    "Window 3 (train Jan-Aug, val Sep, test Oct)": {
        "XGBoost (depth=5)": (0.0663, None),
        "GraphSAGE 100ep (full graph)": (0.1999, 0.0175),
        "GraphSAGE 100ep (no graph)": (0.0611, 0.0002),
    },
}


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def run() -> dict:
    df = gnn.load_sample()
    df["is_fraud"] = df["is_fraud"].astype(bool)

    results = {}
    for name, (train_end, val_end, test_end) in WINDOWS.items():
        window_df = df.copy()
        window_df["split"] = _make_split_col(len(window_df), train_end, val_end, test_end)
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)

        print(f"\n=== {name}: Merchant-only ===")
        data, meta = gnn.build_graph(window_df, include_card_edges=False, include_merchant_edges=True)
        g_train_idx = torch_arange_offset(train_idx, meta["txn_offset"])
        g_val_idx = torch_arange_offset(val_idx, meta["txn_offset"])

        per_seed = []
        for seed in SEEDS:
            model = gnn.train(data, g_train_idx, g_val_idx, seed=seed, epochs=100, hidden_dim=32, lr=0.01)
            scores = gnn.score_all_nodes(model, data, meta)
            window_df["score__merchant_only"] = scores
            m = metrics_mod.compute_metrics(window_df, "score__merchant_only", use_split=True)
            per_seed.append(m)
            print(f"  seed {seed} -> PR-AUC={_fmt(m.get('pr_auc'))}  P@K={_fmt(m.get('precision_at_k'),3)}")

        results[name] = per_seed

    write_report(results)
    return results


def write_report(results: dict, path: Path = RESULTS_TXT) -> None:
    lines = []
    lines.append("=" * 96)
    lines.append("FraudGraph AI — Merchant-only Temporal Robustness (Phase 1 validation)")
    lines.append("=" * 96)
    lines.append("")
    lines.append("Does Window 0's striking Merchant-only result (0.2063 +/- 0.0244, ~3.3x Full")
    lines.append("Graph) hold across the 3 additional chronological windows? Same frozen config")
    lines.append("(hidden_dim=32, epochs=100, lr=0.01) and same 3-seed methodology as those windows.")
    lines.append("")
    lines.append("-" * 96)
    header = f"{'Window':<45}{'XGBoost':>12}{'No Graph':>16}{'Merchant-only':>18}{'Full Graph':>16}"
    lines.append(header)
    lines.append("-" * len(header))

    all_beats_full = True
    for wname in REFERENCE:
        ref = REFERENCE[wname]
        xgb = f"{ref['XGBoost (depth=5)'][0]:.4f}"
        no_graph_m, no_graph_s = ref["GraphSAGE 100ep (no graph)"]
        no_graph = f"{no_graph_m:.4f}+/-{no_graph_s:.4f}"
        full_m, full_s = ref["GraphSAGE 100ep (full graph)"]
        full = f"{full_m:.4f}+/-{full_s:.4f}"

        if "merchant-only" in str(ref.get("GraphSAGE 100ep (merchant-only)", "")) or "GraphSAGE 100ep (merchant-only)" in ref:
            m_m, m_s = ref["GraphSAGE 100ep (merchant-only)"]
        else:
            per_seed = results.get(wname, [])
            m_m, m_s = _mean_std([m.get("pr_auc") for m in per_seed])
        merch = f"{_fmt(m_m)}+/-{_fmt(m_s)}" if m_m is not None else "n/a"

        lines.append(f"{wname:<45}{xgb:>12}{no_graph:>16}{merch:>18}{full:>16}")
        if m_m is not None and m_m <= full_m:
            all_beats_full = False

    lines.append("")
    lines.append("-" * 96)
    lines.append("Per-seed PR-AUC (new merchant-only runs, Windows 1-3; Window 0 already established)")
    lines.append("-" * 96)
    seed_header = f"{'Window':<45}" + "".join(f"{'seed '+str(s):>12}" for s in SEEDS)
    lines.append(seed_header)
    for wname, per_seed in results.items():
        row = f"{wname:<45}"
        for m in per_seed:
            row += f"{_fmt(m.get('pr_auc')):>12}"
        lines.append(row)

    lines.append("")
    lines.append("-" * 96)
    lines.append("Verdict")
    lines.append("-" * 96)
    if all_beats_full:
        lines.append("  CONFIRMED across all 4 windows: Merchant-only consistently and substantially")
        lines.append("  outperforms Full Graph. This is not a Window-0-specific artifact — the merchant")
        lines.append("  relationship dominates the graph signal, and mixing in card edges consistently")
        lines.append("  hurts rather than helps, in every chronological evaluation period tested.")
    else:
        lines.append("  NOT confirmed in every window — merchant-only's advantage over full graph is")
        lines.append("  not universal. Check the per-window table above for which window(s) differ,")
        lines.append("  and treat the Window-0 magnitude as an upper bound rather than a general rule.")
    lines.append("=" * 96)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    run()
