"""
Classical ML baselines trained on the exact same flat transaction features
and temporal train/val/test split as GraphSAGE (gnn.py), so the resulting
comparison isolates the effect of graph structure rather than differences in
feature engineering or evaluation methodology.

Also runs the graph-ablation (GraphSAGE vs GraphSAGE with edges stripped to
self-loops) and a multi-seed robustness sweep, since a single run isn't
enough to trust a ranking when models are close (see robustness_results.txt).

Models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - MLP (single hidden layer)
    - GraphSAGE (full graph) and GraphSAGE (no graph, ablation)

Usage:
    python -m fraudgraph.baselines              # multi-seed robustness sweep
    python -m fraudgraph.baselines --single      # one seeded run only
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import gnn
from . import metrics as metrics_mod

RESULTS_TXT = Path("results/baseline_comparison.txt")
ROBUSTNESS_TXT = Path("results/robustness_results.txt")
SEEDS = [0, 1, 2, 3, 4]


def _fit_predict_all(model, X_train, y_train, X_all) -> np.ndarray:
    model.fit(X_train, y_train)
    return model.predict_proba(X_all)[:, 1]


def train_all(df: pd.DataFrame, train_idx: np.ndarray, seed: int) -> dict:
    """Returns {model_name: fraud_probability array, aligned to df row order}."""
    X = gnn.txn_features(df)
    y = df["is_fraud"].values.astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_idx])
    X_all_scaled = scaler.transform(X)

    y_train = y[train_idx]
    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    scores = {}

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    scores["Logistic Regression"] = _fit_predict_all(logreg, X_train_scaled, y_train, X_all_scaled)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    scores["Random Forest"] = _fit_predict_all(rf, X[train_idx], y_train, X)

    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=seed, n_jobs=-1,
    )
    scores["XGBoost"] = _fit_predict_all(xgb, X[train_idx], y_train, X)

    mlp = MLPClassifier(
        hidden_layer_sizes=(32,), max_iter=500, early_stopping=True,
        random_state=seed,
    )
    scores["MLP"] = _fit_predict_all(mlp, X_train_scaled, y_train, X_all_scaled)

    return scores


def _fmt(v, digits=4):
    return f"{v:.{digits}f}" if v is not None else "n/a"


def _load_and_split():
    df = gnn.load_sample()
    n_txn = len(df)
    df["is_fraud"] = df["is_fraud"].astype(bool)

    train_idx, val_idx, test_idx = gnn.temporal_split(n_txn, txn_offset=0)
    train_idx, val_idx, test_idx = train_idx.numpy(), val_idx.numpy(), test_idx.numpy()

    split = np.full(n_txn, "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    df["split"] = split
    return df, train_idx, val_idx, test_idx


def run_single(seed: int, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """One fully-seeded pass: classical baselines + rule-based + GraphSAGE
    (full graph) + GraphSAGE (no-graph ablation). Returns (df, comparison)."""
    df, train_idx, val_idx, test_idx = _load_and_split()
    if verbose:
        print(f"[seed {seed}] {len(df)} txns | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    if verbose:
        print(f"[seed {seed}] training classical baselines...")
    baseline_scores = train_all(df, train_idx, seed)
    for name, s in baseline_scores.items():
        df[f"score__{name}"] = s

    if verbose:
        print(f"[seed {seed}] scoring rule-based baseline...")
    from . import anomaly as rule
    df = rule.add_velocity(df)
    df = rule.add_amount_zscore(df)
    df = rule.add_new_merchant(df)
    df = rule.add_merchant_rarity(df)
    df = rule.add_error_flag(df)
    df = rule.score(df)

    if verbose:
        print(f"[seed {seed}] training GraphSAGE (full graph)...")
    data, meta = gnn.build_graph(df)
    g_train_idx, g_val_idx, g_test_idx = gnn.temporal_split(meta["n_txn"], meta["txn_offset"])
    model = gnn.train(data, g_train_idx, g_val_idx, seed=seed)
    df["gnn_score"] = gnn.score_all_nodes(model, data, meta)

    if verbose:
        print(f"[seed {seed}] training GraphSAGE (no graph, ablation)...")
    data_no_graph = gnn.strip_edges(data)
    model_no_graph = gnn.train(data_no_graph, g_train_idx, g_val_idx, seed=seed)
    df["score__GraphSAGE (no graph)"] = gnn.score_all_nodes(model_no_graph, data_no_graph, meta)

    extra_scores = {name: f"score__{name}" for name in baseline_scores}
    extra_scores["GraphSAGE (no graph)"] = "score__GraphSAGE (no graph)"
    comparison = metrics_mod.model_comparison(df, use_split=True, extra_scores=extra_scores)

    if verbose:
        print(f"[seed {seed}] " + "  ".join(
            f"{name}={_fmt(m.get('pr_auc'))}" for name, m in comparison.items()
        ))
    return df, comparison


def SAMPLE_USERS_NOTE(df: pd.DataFrame) -> str:
    return f"{df['user_id'].nunique()} users, {df['merchant_id'].nunique()} merchants"


def write_report(df: pd.DataFrame, comparison: dict, path: Path = RESULTS_TXT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_train = (df["split"] == "train").sum()
    n_val = (df["split"] == "val").sum()
    n_test = (df["split"] == "test").sum()
    n_fraud_test = int(df.loc[df["split"] == "test", "is_fraud"].sum())

    lines = []
    lines.append("=" * 78)
    lines.append("FraudGraph AI — Baseline vs GraphSAGE Model Comparison (single seed)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Dataset       : {len(df)} transactions ({SAMPLE_USERS_NOTE(df)})")
    lines.append(f"Split (temporal, earliest -> latest):")
    lines.append(f"  train : {n_train}")
    lines.append(f"  val   : {n_val}")
    lines.append(f"  test  : {n_test}  ({n_fraud_test} fraud, base rate {n_fraud_test / max(n_test,1):.4%})")
    lines.append("")
    lines.append("Features (identical across every model, incl. GraphSAGE's node features):")
    lines.append("  amount (log1p, standardized), hour-of-day (sin/cos), day-of-week (sin/cos),")
    lines.append("  channel one-hot [Swipe/Chip/Online], has_error flag")
    lines.append("  -> GraphSAGE additionally sees the card<->transaction<->merchant graph structure.")
    lines.append("")
    lines.append("NOTE: this is a single seeded run. See robustness_results.txt for a")
    lines.append("multi-seed sweep — single-run rankings between close models are not reliable")
    lines.append("(see that file's notes on GraphSAGE vs XGBoost variance).")
    lines.append("")
    lines.append("-" * 78)
    k = None
    for m in comparison.values():
        if m.get("k") is not None:
            k = m["k"]
            break
    lines.append(f"Results on held-out TEST split (k = {k})")
    lines.append("-" * 78)
    header = f"{'Model':<22}{'PR-AUC':>10}{'ROC-AUC':>10}{'Precision@K':>14}{'Lift over random':>20}"
    lines.append(header)
    lines.append("-" * len(header))

    def sort_key(item):
        v = item[1].get("pr_auc")
        return (-1 if v is None else 1, -(v or 0))

    for name, m in sorted(comparison.items(), key=sort_key):
        lines.append(
            f"{name:<22}{_fmt(m.get('pr_auc')):>10}{_fmt(m.get('roc_auc')):>10}"
            f"{_fmt(m.get('precision_at_k'), 3):>14}"
            f"{(_fmt(m.get('lift'), 1) + 'x') if m.get('lift') is not None else 'n/a':>20}"
        )
    lines.append("")
    lines.append("-" * 78)
    lines.append("Ablation: what does the graph structure itself contribute?")
    lines.append("-" * 78)
    full = comparison.get("GraphSAGE")
    no_graph = comparison.get("GraphSAGE (no graph)")
    if full and no_graph and full.get("pr_auc") is not None and no_graph.get("pr_auc") is not None:
        d_prauc = full["pr_auc"] - no_graph["pr_auc"]
        d_p_at_k = (full.get("precision_at_k") or 0) - (no_graph.get("precision_at_k") or 0)
        lines.append(f"  GraphSAGE (full graph)     PR-AUC={_fmt(full['pr_auc'])}  P@K={_fmt(full.get('precision_at_k'),3)}")
        lines.append(f"  GraphSAGE (no graph)       PR-AUC={_fmt(no_graph['pr_auc'])}  P@K={_fmt(no_graph.get('precision_at_k'),3)}")
        lines.append(f"  Delta from graph structure: PR-AUC {'+' if d_prauc >= 0 else ''}{d_prauc:.4f}   "
                      f"P@K {'+' if d_p_at_k >= 0 else ''}{d_p_at_k:.3f}")
    else:
        lines.append("  (ablation scores unavailable)")
    lines.append("")
    lines.append("-" * 78)
    lines.append("Notes")
    lines.append("-" * 78)
    lines.append("- PR-AUC (average precision) is the primary metric given the heavy class")
    lines.append("  imbalance in this dataset; ROC-AUC is included for reference.")
    lines.append("- Rule-based = handcrafted anomaly score (anomaly.py), no training.")
    lines.append("- Classical baselines and GraphSAGE are trained on the SAME temporal split")
    lines.append("  and SAME flat feature matrix (gnn.txn_features); GraphSAGE additionally")
    lines.append("  sees the card/merchant graph topology.")
    lines.append("=" * 78)

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSingle-seed results written to {path}")


def run_robustness(seeds: list[int] = SEEDS, path: Path = ROBUSTNESS_TXT) -> None:
    per_seed = []  # list of (seed, comparison dict)
    last_df = None
    for seed in seeds:
        df, comparison = run_single(seed)
        per_seed.append((seed, comparison))
        last_df = df

    # Also refresh the single-seed report (last seed run) for continuity with
    # anyone looking at baseline_comparison.txt directly.
    write_report(last_df, per_seed[-1][1])

    model_names = list(per_seed[0][1].keys())
    metrics_keys = ["pr_auc", "roc_auc", "precision_at_k", "lift"]

    agg = {}  # model -> metric -> list of values across seeds
    for name in model_names:
        agg[name] = {mk: [] for mk in metrics_keys}
        for _, comparison in per_seed:
            m = comparison.get(name, {})
            for mk in metrics_keys:
                v = m.get(mk)
                if v is not None:
                    agg[name][mk].append(v)

    lines = []
    lines.append("=" * 86)
    lines.append(f"FraudGraph AI — Multi-seed Robustness Sweep ({len(seeds)} seeds: {seeds})")
    lines.append("=" * 86)
    lines.append("")
    lines.append("Every model (classical baselines, GraphSAGE full-graph, GraphSAGE no-graph")
    lines.append("ablation) is retrained from scratch per seed, on the same temporal train/")
    lines.append("val/test split, and evaluated on the held-out test split. Mean +/- std")
    lines.append("across seeds below — a ranking is only meaningful if the gap between two")
    lines.append("models exceeds their overlapping std range.")
    lines.append("")
    lines.append("-" * 86)
    lines.append("Mean +/- std PR-AUC / Precision@K / Lift, ranked by mean PR-AUC")
    lines.append("-" * 86)
    header = f"{'Model':<24}{'PR-AUC':>20}{'Precision@K':>20}{'Lift':>20}"
    lines.append(header)
    lines.append("-" * len(header))

    def mean_std(vals):
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    ranked = sorted(model_names, key=lambda n: -(mean_std(agg[n]["pr_auc"])[0] or -1))
    summary_rows = {}
    for name in ranked:
        pr_m, pr_s = mean_std(agg[name]["pr_auc"])
        pk_m, pk_s = mean_std(agg[name]["precision_at_k"])
        lift_m, lift_s = mean_std(agg[name]["lift"])
        summary_rows[name] = (pr_m, pr_s, pk_m, pk_s, lift_m, lift_s)
        pr_str = f"{_fmt(pr_m)} +/- {_fmt(pr_s)}" if pr_m is not None else "n/a"
        pk_str = f"{_fmt(pk_m,3)} +/- {_fmt(pk_s,3)}" if pk_m is not None else "n/a"
        lift_str = f"{_fmt(lift_m,1)}x +/- {_fmt(lift_s,1)}" if lift_m is not None else "n/a"
        lines.append(f"{name:<24}{pr_str:>20}{pk_str:>20}{lift_str:>20}")

    lines.append("")
    lines.append("-" * 86)
    lines.append("Per-seed PR-AUC (raw values, for inspecting run-to-run variance directly)")
    lines.append("-" * 86)
    seed_header = f"{'Model':<24}" + "".join(f"{'seed '+str(s):>12}" for s, _ in per_seed)
    lines.append(seed_header)
    for name in ranked:
        row = f"{name:<24}"
        for _, comparison in per_seed:
            v = comparison.get(name, {}).get("pr_auc")
            row += f"{_fmt(v):>12}"
        lines.append(row)

    lines.append("")
    lines.append("-" * 86)
    lines.append("Graph ablation stability: does GraphSAGE(full) beat GraphSAGE(no graph) every seed?")
    lines.append("-" * 86)
    n_full_wins = 0
    for seed, comparison in per_seed:
        full = comparison.get("GraphSAGE", {}).get("pr_auc")
        no_graph = comparison.get("GraphSAGE (no graph)", {}).get("pr_auc")
        if full is not None and no_graph is not None:
            won = full > no_graph
            n_full_wins += int(won)
            lines.append(f"  seed {seed}: full={_fmt(full)}  no-graph={_fmt(no_graph)}  "
                          f"-> {'graph helped' if won else 'graph did NOT help'}")
    lines.append(f"  Graph structure improved PR-AUC in {n_full_wins}/{len(per_seed)} seeds.")
    lines.append("")
    if n_full_wins == len(per_seed):
        lines.append("  -> Consistent: graph topology reliably contributes positive signal on")
        lines.append("     this sample, independent of GraphSAGE's random initialization.")
    elif n_full_wins == 0:
        lines.append("  -> Consistent: graph topology did NOT help on this sample in any seed.")
    else:
        lines.append("  -> Inconsistent across seeds — the graph's contribution is not reliably")
        lines.append("     positive at the current training budget (30 epochs, hidden_dim=32).")
        lines.append("     Treat any single-seed 'GraphSAGE beats X' or 'X beats GraphSAGE' claim")
        lines.append("     as unsupported until this is resolved (more epochs, tuning, or more")
        lines.append("     seeds to narrow the confidence interval).")

    lines.append("")
    lines.append("-" * 86)
    lines.append("Ranking stability: how often is each model's PR-AUC rank #1 across seeds?")
    lines.append("-" * 86)
    rank1_counts = {name: 0 for name in model_names}
    for _, comparison in per_seed:
        pr_vals = {name: comparison.get(name, {}).get("pr_auc") for name in model_names}
        pr_vals = {k: v for k, v in pr_vals.items() if v is not None}
        if pr_vals:
            best = max(pr_vals, key=pr_vals.get)
            rank1_counts[best] += 1
    for name in ranked:
        lines.append(f"  {name:<24} #1 in {rank1_counts[name]}/{len(per_seed)} seeds")

    lines.append("")
    lines.append("=" * 86)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRobustness sweep written to {path}")


def main() -> None:
    if "--single" in sys.argv:
        df, comparison = run_single(seed=0)
        write_report(df, comparison)
    else:
        run_robustness(SEEDS)


if __name__ == "__main__":
    main()
