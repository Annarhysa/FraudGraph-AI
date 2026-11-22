"""
Phase: temporal robustness. Does the GraphSAGE advantage over tuned XGBoost
survive when the chronological evaluation period changes?

IMPORTANT dataset constraint (see Part A of the report / gnn.py's existing
note): fraud labels only occur Jan-Oct 2019 in this sample; Nov 2019-Feb 2020
(~192K rows, ~28% of the sample) has zero fraud. Every window below keeps its
TEST period inside the fraud-bearing region so PR-AUC stays meaningful — see
write_fraud_distribution() for the actual per-month counts that drove this.

Frozen configs (per instruction — do not re-tune here):
    XGBoost:   max_depth=5, learning_rate=0.05, n_estimators=300, seed=42
    GraphSAGE: hidden_dim=32, epochs=100, lr=0.01, seeds=[0,1,2]
               (3 seeds here, not 5, to keep this multi-window sweep's CPU-only
               runtime bounded — see report notes for the reasoning)

Usage:
    python -m fraudgraph.temporal_windows
"""
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from . import gnn
from . import metrics as metrics_mod
from .baselines import _fmt

RESULTS_TXT = Path("results/temporal_robustness_results.txt")
SEEDS = [0, 1, 2]
XGB_PARAMS = dict(max_depth=5, learning_rate=0.05, n_estimators=300, random_state=42)

# Row-index cuts (train_end, val_end, test_end) computed from the sample's
# actual month boundaries (see Part A). test_end is exclusive.
WINDOWS = {
    "Window 1 (train Jan-Apr, val May, test Jun-Oct)": (191285, 240899, 485556),
    "Window 2 (train Jan-Jun, val Jul, test Aug-Oct)": (289082, 338787, 485556),
    "Window 3 (train Jan-Aug, val Sep, test Oct)":     (388509, 436293, 485556),
}

# Reference numbers for "Window 0" — the ORIGINAL default 50/10/40 split used
# in every earlier experiment. Its test period runs Sep 2019 -> Feb 2020, i.e.
# it includes the zero-fraud Nov-Feb tail as padding. Not re-run here.
WINDOW_0_REFERENCE = {
    "XGBoost (depth=5)": 0.0277,
    "GraphSAGE 100ep (full graph)": (0.0623, 0.0117),
    "GraphSAGE 100ep (no graph)": (0.0151, 0.0003),
}


def write_fraud_distribution(df: pd.DataFrame, path: Path) -> str:
    period = df["txn_ts"].dt.to_period("M")
    g = df.groupby(period)["is_fraud"].agg(n_txn="size", n_fraud="sum")
    g["fraud_pct"] = 100 * g["n_fraud"] / g["n_txn"]

    lines = []
    lines.append("-" * 60)
    lines.append("Part A: chronological fraud distribution (monthly)")
    lines.append("-" * 60)
    lines.append(f"{'Month':<10}{'Transactions':>14}{'Fraud':>10}{'Fraud %':>10}")
    for period_val, row in g.iterrows():
        lines.append(f"{str(period_val):<10}{int(row['n_txn']):>14}{int(row['n_fraud']):>10}{row['fraud_pct']:>10.4f}")
    lines.append("")
    lines.append("Fraud is present Jan-Oct 2019 only; Nov 2019-Feb 2020 (~192K rows, ~28% of")
    lines.append("the sample) has ZERO fraud labels. This is a dataset limitation (the source")
    lines.append("data stops injecting fraud before the sample window ends), not an artifact of")
    lines.append("our sampling — every temporal window below keeps its TEST period within the")
    lines.append("fraud-bearing Jan-Oct range for this reason.")
    return "\n".join(lines)


def _make_split_col(n_txn: int, train_end: int, val_end: int, test_end: int) -> np.ndarray:
    split = np.full(n_txn, "unused", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    split[val_end:test_end] = "test"
    return split


def run_window(df: pd.DataFrame, name: str, train_end: int, val_end: int, test_end: int) -> dict:
    n_txn = len(df)
    df = df.copy()
    df["split"] = _make_split_col(n_txn, train_end, val_end, test_end)
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)

    n_test = test_end - val_end
    n_fraud_test = int(df.iloc[val_end:test_end]["is_fraud"].sum())
    print(f"\n=== {name} ===")
    print(f"train={train_end}  val={val_end - train_end}  test={n_test} ({n_fraud_test} fraud)")

    results = {}

    # --- XGBoost (frozen config) ---
    X = gnn.txn_features(df)
    y = df["is_fraud"].values.astype(int)
    y_train = y[train_idx]
    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    xgb = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight, eval_metric="aucpr", n_jobs=-1)
    xgb.fit(X[train_idx], y_train)
    df["score__xgb"] = xgb.predict_proba(X)[:, 1]
    m = metrics_mod.compute_metrics(df, "score__xgb", use_split=True)
    results["XGBoost (depth=5)"] = [m]
    print(f"  XGBoost           PR-AUC={_fmt(m.get('pr_auc'))}  P@K={_fmt(m.get('precision_at_k'),3)}")

    # --- GraphSAGE full graph + no-graph ablation, 3 seeds ---
    data, meta = gnn.build_graph(df)
    g_train_idx = torch_arange_offset(train_idx, meta["txn_offset"])
    g_val_idx = torch_arange_offset(val_idx, meta["txn_offset"])

    for label, strip in [("GraphSAGE 100ep (full graph)", False), ("GraphSAGE 100ep (no graph)", True)]:
        results[label] = []
        d = gnn.strip_edges(data) if strip else data
        for seed in SEEDS:
            model = gnn.train(d, g_train_idx, g_val_idx, seed=seed, epochs=100, hidden_dim=32, lr=0.01)
            scores = gnn.score_all_nodes(model, d, meta)
            df[f"score__{label}"] = scores
            m = metrics_mod.compute_metrics(df, f"score__{label}", use_split=True)
            results[label].append(m)
            print(f"  {label} seed={seed}  PR-AUC={_fmt(m.get('pr_auc'))}  P@K={_fmt(m.get('precision_at_k'),3)}")

    return results


def torch_arange_offset(idx: np.ndarray, offset: int):
    import torch
    return torch.tensor(idx + offset, dtype=torch.long)


def main() -> None:
    df = gnn.load_sample()
    df["is_fraud"] = df["is_fraud"].astype(bool)

    dist_text = write_fraud_distribution(df, RESULTS_TXT)
    print(dist_text)

    all_window_results = {}
    for name, (train_end, val_end, test_end) in WINDOWS.items():
        all_window_results[name] = run_window(df, name, train_end, val_end, test_end)

    write_report(dist_text, all_window_results)


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def write_report(dist_text: str, all_window_results: dict, path: Path = RESULTS_TXT) -> None:
    lines = [dist_text, ""]
    lines.append("=" * 92)
    lines.append("Part B/C: Temporal Robustness — GraphSAGE vs Tuned XGBoost across evaluation windows")
    lines.append("=" * 92)
    lines.append("")
    lines.append("Frozen configs (no re-tuning in this experiment):")
    lines.append("  XGBoost:   max_depth=5, learning_rate=0.05, n_estimators=300, seed=42")
    lines.append(f"  GraphSAGE: hidden_dim=32, epochs=100, lr=0.01, seeds={SEEDS}")
    lines.append("")
    lines.append("NOTE: 3 seeds (not the original 5) are used per window here to keep this")
    lines.append("multi-window sweep's CPU-only runtime bounded (this is 3x the model-training")
    lines.append("volume of the single-window 5-seed sweep). Mean+/-std is still reported, but")
    lines.append("treat the per-window std as a slightly less precise estimate than Window 0's.")
    lines.append("")
    lines.append("Window 0 (reference, NOT re-run here) = the ORIGINAL default 50/10/40 split")
    lines.append("used in every earlier experiment. Its test period runs Sep 2019 -> Feb 2020,")
    lines.append("i.e. ~71% of its test rows fall in the zero-fraud Nov-Feb tail (see Part A).")
    lines.append("Windows 1-3 below keep test entirely inside the fraud-bearing Jan-Oct region.")
    lines.append("")

    def fmt_row(name, pr_m, pr_s):
        pr_str = f"{_fmt(pr_m)} +/- {_fmt(pr_s)}" if pr_s is not None else (_fmt(pr_m) if pr_m is not None else "n/a")
        return f"  {name:<32}{pr_str}"

    lines.append("-" * 92)
    lines.append("Window 0 (reference / original split, includes zero-fraud padding in test)")
    lines.append("-" * 92)
    for name, val in WINDOW_0_REFERENCE.items():
        if isinstance(val, tuple):
            lines.append(fmt_row(name, val[0], val[1]))
        else:
            lines.append(fmt_row(name, val, None))

    per_window_verdicts = []
    for wname, results in all_window_results.items():
        lines.append("")
        lines.append("-" * 92)
        lines.append(wname)
        lines.append("-" * 92)
        header = f"{'Model':<32}{'PR-AUC':>22}{'Precision@K':>18}{'Lift':>18}"
        lines.append(header)
        lines.append("-" * len(header))

        summary = {}
        for name, per_seed in results.items():
            pr_m, pr_s = _mean_std([m.get("pr_auc") for m in per_seed])
            pk_m, pk_s = _mean_std([m.get("precision_at_k") for m in per_seed])
            lift_m, lift_s = _mean_std([m.get("lift") for m in per_seed])
            summary[name] = (pr_m, pr_s, pk_m, pk_s, lift_m, lift_s)

        for name, (pr_m, pr_s, pk_m, pk_s, lift_m, lift_s) in sorted(summary.items(), key=lambda kv: -(kv[1][0] or -1)):
            pr_str = f"{_fmt(pr_m)} +/- {_fmt(pr_s)}" if pr_s else _fmt(pr_m)
            pk_str = f"{_fmt(pk_m,3)} +/- {_fmt(pk_s,3)}" if pk_s else _fmt(pk_m,3)
            lift_str = f"{_fmt(lift_m,1)}x +/- {_fmt(lift_s,1)}" if lift_s else f"{_fmt(lift_m,1)}x"
            lines.append(f"{name:<32}{pr_str:>22}{pk_str:>18}{lift_str:>18}")

        xgb_pr = summary.get("XGBoost (depth=5)", (None,))[0]
        full_pr = summary.get("GraphSAGE 100ep (full graph)", (None, None))
        no_graph_pr = summary.get("GraphSAGE 100ep (no graph)", (None, None))
        beats_xgb = full_pr[0] is not None and xgb_pr is not None and (full_pr[0] - (full_pr[1] or 0)) > xgb_pr
        graph_helps = full_pr[0] is not None and no_graph_pr[0] is not None and full_pr[0] > no_graph_pr[0]
        per_window_verdicts.append((wname, beats_xgb, graph_helps))
        lines.append("")
        lines.append(f"  -> GraphSAGE(100ep) beats XGBoost here (1-std lower bound > XGBoost): {beats_xgb}")
        lines.append(f"  -> Graph structure helps here (full > no-graph): {graph_helps}")

    lines.append("")
    lines.append("=" * 92)
    lines.append("Overall temporal robustness verdict")
    lines.append("=" * 92)
    n_beats = sum(1 for _, b, _ in per_window_verdicts if b)
    n_graph = sum(1 for _, _, g in per_window_verdicts if g)
    n_windows = len(per_window_verdicts)
    for wname, beats, graph in per_window_verdicts:
        lines.append(f"  {wname}: GraphSAGE>XGBoost={beats}  graph_helps={graph}")
    lines.append("")
    lines.append(f"GraphSAGE beat tuned XGBoost in {n_beats}/{n_windows} new windows (+ Window 0 reference).")
    lines.append(f"Graph structure helped in {n_graph}/{n_windows} new windows (+ Window 0 reference).")
    lines.append("")
    if n_beats == n_windows and n_graph == n_windows:
        lines.append("-> The GraphSAGE advantage and the graph's contribution both survive across")
        lines.append("   every chronological evaluation window tested, not just the original split.")
        lines.append("   This is the strongest form of the claim your research question can currently")
        lines.append("   support: relational structure provides a temporally robust advantage.")
    else:
        lines.append("-> The advantage does NOT hold uniformly across all evaluation windows.")
        lines.append("   Report this honestly: temporal robustness is partial, and the specific")
        lines.append("   window(s) where it breaks down are worth investigating (fraud volume in")
        lines.append("   that window's test set, distribution shift, etc.) rather than glossed over.")
    lines.append("=" * 92)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    main()
