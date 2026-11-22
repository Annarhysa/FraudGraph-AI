"""
Re-run of the Full Graph condition in temporal_windows.py's 3 additional
windows, after the gnn.build_graph merchant_offset bugfix (merchant edges
previously never reached real merchant nodes). XGBoost and No Graph results
from the original temporal_robustness_results.txt are UNAFFECTED by that bug
(XGBoost never touches the graph; No Graph uses self-loops only) and are
reused here rather than re-trained.

Usage:
    python -m fraudgraph.temporal_windows_fullgraph_fix
"""
from pathlib import Path

import numpy as np

from . import gnn
from . import metrics as metrics_mod
from .baselines import _fmt
from .temporal_windows import WINDOWS, SEEDS, _make_split_col, torch_arange_offset

RESULTS_TXT = Path("results/temporal_robustness_fullgraph_fix_results.txt")

# Unaffected-by-bug reference values, carried over from
# temporal_robustness_results.txt and edge_ablation_results.txt.
UNCHANGED_REFERENCE = {
    "Window 0 (original default split)": {"XGBoost (depth=5)": 0.0277, "No Graph": (0.0151, 0.0003)},
    "Window 1 (train Jan-Apr, val May, test Jun-Oct)": {"XGBoost (depth=5)": 0.0445, "No Graph": (0.0368, 0.0004)},
    "Window 2 (train Jan-Jun, val Jul, test Aug-Oct)": {"XGBoost (depth=5)": 0.0562, "No Graph": (0.0445, 0.0013)},
    "Window 3 (train Jan-Aug, val Sep, test Oct)":     {"XGBoost (depth=5)": 0.0663, "No Graph": (0.0611, 0.0002)},
}

# OLD (buggy-merchant-edge) Full Graph numbers, for an explicit before/after
# comparison in the report.
OLD_BUGGY_FULL_GRAPH = {
    "Window 0 (original default split)": (0.0623, 0.0117),
    "Window 1 (train Jan-Apr, val May, test Jun-Oct)": (0.1237, 0.0157),
    "Window 2 (train Jan-Jun, val Jul, test Aug-Oct)": (0.1770, 0.0099),
    "Window 3 (train Jan-Aug, val Sep, test Oct)":     (0.1999, 0.0175),
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

        print(f"\n=== {name}: Full Graph (fixed merchant edges) ===")
        data, meta = gnn.build_graph(window_df, include_card_edges=True, include_merchant_edges=True)
        g_train_idx = torch_arange_offset(train_idx, meta["txn_offset"])
        g_val_idx = torch_arange_offset(val_idx, meta["txn_offset"])

        per_seed = []
        for seed in SEEDS:
            model = gnn.train(data, g_train_idx, g_val_idx, seed=seed, epochs=100, hidden_dim=32, lr=0.01)
            scores = gnn.score_all_nodes(model, data, meta)
            window_df["score__full_graph_fixed"] = scores
            m = metrics_mod.compute_metrics(window_df, "score__full_graph_fixed", use_split=True)
            per_seed.append(m)
            print(f"  seed {seed} -> PR-AUC={_fmt(m.get('pr_auc'))}  P@K={_fmt(m.get('precision_at_k'),3)}")

        results[name] = per_seed

    # Also re-run Window 0's Full Graph with the SAME 5-seed count as the
    # original edge_ablation for direct comparability there (handled by
    # edge_ablation.py separately) — here we only need Windows 1-3 fresh;
    # Window 0 is taken from edge_ablation_results.txt once that finishes.
    write_report(results)
    return results


def write_report(results: dict, path: Path = RESULTS_TXT) -> None:
    lines = []
    lines.append("=" * 100)
    lines.append("FraudGraph AI — Full Graph Temporal Robustness, CORRECTED (merchant_offset bugfix)")
    lines.append("=" * 100)
    lines.append("")
    lines.append("gnn.build_graph had a bug: merchant edges never reached real merchant nodes")
    lines.append("(missing +merchant_offset shift). This re-runs the Full Graph condition for")
    lines.append("Windows 1-3 with the fix. XGBoost and No Graph are unaffected by the bug and")
    lines.append("reused from temporal_robustness_results.txt. Window 0's corrected Full Graph")
    lines.append("number comes from edge_ablation_results.txt (run separately, same fix).")
    lines.append("")
    lines.append("-" * 100)
    header = f"{'Window':<45}{'XGBoost':>12}{'No Graph':>16}{'Full Graph (OLD, buggy)':>26}{'Full Graph (FIXED)':>22}"
    lines.append(header)
    lines.append("-" * len(header))

    for wname in UNCHANGED_REFERENCE:
        xgb = UNCHANGED_REFERENCE[wname]["XGBoost (depth=5)"]
        ng_m, ng_s = UNCHANGED_REFERENCE[wname]["No Graph"]
        old_m, old_s = OLD_BUGGY_FULL_GRAPH[wname]
        if wname in results:
            new_vals = [m.get("pr_auc") for m in results[wname]]
            new_m, new_s = _mean_std(new_vals)
        else:
            new_m, new_s = None, None  # Window 0, filled from edge_ablation separately
        new_str = f"{_fmt(new_m)}+/-{_fmt(new_s)}" if new_m is not None else "(see edge_ablation_results.txt)"
        lines.append(f"{wname:<45}{xgb:>12.4f}{ng_m:>10.4f}+/-{ng_s:<5.4f}"
                      f"{old_m:>18.4f}+/-{old_s:<6.4f}{new_str:>22}")

    lines.append("")
    lines.append("-" * 100)
    lines.append("Per-seed PR-AUC (corrected Full Graph, Windows 1-3)")
    lines.append("-" * 100)
    seed_header = f"{'Window':<45}" + "".join(f"{'seed '+str(s):>12}" for s in SEEDS)
    lines.append(seed_header)
    for wname, per_seed in results.items():
        row = f"{wname:<45}"
        for m in per_seed:
            row += f"{_fmt(m.get('pr_auc')):>12}"
        lines.append(row)
    lines.append("=" * 100)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    run()
