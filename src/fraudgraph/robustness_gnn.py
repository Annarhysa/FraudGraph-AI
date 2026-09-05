"""
Confirms (or refutes) the single-seed tuning.py finding that GraphSAGE at
100 epochs (PR-AUC 0.0501) beats XGBoost (0.0277) — across 5 seeds, and with
the graph-ablation (with vs without edges) repeated at the same 100-epoch
budget, since the earlier ablation was only ever run at 30 epochs.

Also persists every run's per-epoch validation curve (train_loss, val_loss,
val_pr_auc, val_roc_auc) to CSV so training dynamics can be inspected/plotted
later, not just the final-epoch number.

Fixed by design (per user instruction — no new features, no hyperparameter
search yet): hidden_dim=32, epochs=100, lr=gnn.LR (0.01), same features/graph/
temporal split as every other experiment in this project.

Usage:
    python -m fraudgraph.robustness_gnn
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import gnn
from . import metrics as metrics_mod
from .baselines import _load_and_split, _fmt

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 100
HIDDEN_DIM = 32

CURVES_DIR = Path("results/curves")
RESULTS_TXT = Path("results/graphsage_100ep_5seed_results.txt")

# Reference numbers already established (deterministic / previously measured
# in robustness_results.txt and tuning_results.txt).
REFERENCE = {
    "Rule-based": (0.0034, None),
    "Logistic Regression": (0.0040, 0.0000),
    "Random Forest": (0.0160, 0.0003),
    "XGBoost": (0.0277, 0.0000),
    "GraphSAGE 30ep (full graph)": (0.0172, 0.0018),
    "GraphSAGE 30ep (no graph)": (0.0110, 0.0005),
}


def run_variant(data, meta, g_train_idx, g_val_idx, seed: int, label: str, strip: bool):
    d = gnn.strip_edges(data) if strip else data
    model, history = gnn.train(
        d, g_train_idx, g_val_idx, seed=seed,
        epochs=EPOCHS, hidden_dim=HIDDEN_DIM, lr=gnn.LR, return_history=True,
    )
    hist_df = pd.DataFrame(history)
    hist_df["seed"] = seed
    hist_df["variant"] = label
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
    hist_df.to_csv(CURVES_DIR / f"{safe_label}_seed{seed}.csv", index=False)

    scores = gnn.score_all_nodes(model, d, meta)
    return scores, hist_df


def main() -> None:
    df, train_idx, val_idx, test_idx = _load_and_split()
    data, meta = gnn.build_graph(df)
    g_train_idx, g_val_idx, g_test_idx = gnn.temporal_split(meta["n_txn"], meta["txn_offset"])

    variants = {
        "GraphSAGE 100ep (full graph)": False,
        "GraphSAGE 100ep (no graph)": True,
    }

    per_variant_metrics = {name: [] for name in variants}
    all_histories = []

    for seed in SEEDS:
        for label, strip in variants.items():
            print(f"[seed {seed}] training {label} ({EPOCHS} epochs, hidden_dim={HIDDEN_DIM})...")
            scores, hist_df = run_variant(data, meta, g_train_idx, g_val_idx, seed, label, strip)
            all_histories.append(hist_df)
            df[f"score__{label}"] = scores
            m = metrics_mod.compute_metrics(df, f"score__{label}", use_split=True)
            per_variant_metrics[label].append(m)
            print(f"[seed {seed}] {label} -> PR-AUC={_fmt(m.get('pr_auc'))}  "
                  f"P@K={_fmt(m.get('precision_at_k'),3)}")

    combined_history = pd.concat(all_histories, ignore_index=True)
    combined_history.to_csv(CURVES_DIR / "all_runs_combined.csv", index=False)

    write_report(per_variant_metrics, combined_history)


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def write_report(per_variant_metrics: dict, combined_history: pd.DataFrame, path: Path = RESULTS_TXT) -> None:
    lines = []
    lines.append("=" * 90)
    lines.append("FraudGraph AI — GraphSAGE @ 100 epochs, 5-seed confirmation + graph ablation")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"Fixed config: hidden_dim={HIDDEN_DIM}, epochs={EPOCHS}, lr={gnn.LR}, seeds={SEEDS}")
    lines.append("Same temporal split, same node/txn features as every other experiment in this repo.")
    lines.append("Purpose: the single-seed tuning.py run found GraphSAGE(100ep)=0.0501 PR-AUC,")
    lines.append("beating XGBoost's 0.0277 — but that was one seed and explicitly flagged as")
    lines.append("directional only. This repeats it across 5 seeds, and re-runs the graph-vs-")
    lines.append("no-graph ablation at 100 epochs (the earlier ablation was only measured at 30).")
    lines.append("")
    lines.append("-" * 90)
    lines.append("Core results table (mean +/- std where multi-seed; single numbers are prior")
    lines.append("deterministic/known results carried over from robustness_results.txt)")
    lines.append("-" * 90)
    header = f"{'Model':<32}{'PR-AUC':>22}{'Precision@K':>18}{'Lift':>18}"
    lines.append(header)
    lines.append("-" * len(header))

    rows = []
    for name, (m, s) in REFERENCE.items():
        rows.append((name, m, s, None, None, None, None))
    for name, per_seed in per_variant_metrics.items():
        pr_m, pr_s = _mean_std([m.get("pr_auc") for m in per_seed])
        pk_m, pk_s = _mean_std([m.get("precision_at_k") for m in per_seed])
        lift_m, lift_s = _mean_std([m.get("lift") for m in per_seed])
        rows.append((name, pr_m, pr_s, pk_m, pk_s, lift_m, lift_s))

    rows.sort(key=lambda r: -(r[1] if r[1] is not None else -1))
    for name, pr_m, pr_s, pk_m, pk_s, lift_m, lift_s in rows:
        pr_str = f"{_fmt(pr_m)} +/- {_fmt(pr_s)}" if pr_s is not None else (f"{_fmt(pr_m)}" if pr_m is not None else "n/a")
        pk_str = f"{_fmt(pk_m,3)} +/- {_fmt(pk_s,3)}" if pk_s is not None else "n/a"
        lift_str = f"{_fmt(lift_m,1)}x +/- {_fmt(lift_s,1)}" if lift_s is not None else "n/a"
        lines.append(f"{name:<32}{pr_str:>22}{pk_str:>18}{lift_str:>18}")

    lines.append("")
    lines.append("-" * 90)
    lines.append("Per-seed PR-AUC for the two new 100-epoch variants")
    lines.append("-" * 90)
    seed_header = f"{'Model':<32}" + "".join(f"{'seed '+str(s):>12}" for s in SEEDS)
    lines.append(seed_header)
    for name, per_seed in per_variant_metrics.items():
        row = f"{name:<32}"
        for m in per_seed:
            row += f"{_fmt(m.get('pr_auc')):>12}"
        lines.append(row)

    # --- verdict: does GraphSAGE 100ep consistently beat XGBoost? ---
    lines.append("")
    lines.append("-" * 90)
    lines.append("Verdict 1: does GraphSAGE(100ep, full graph) consistently beat XGBoost (0.0277)?")
    lines.append("-" * 90)
    full_vals = [m.get("pr_auc") for m in per_variant_metrics["GraphSAGE 100ep (full graph)"]]
    n_beats = sum(1 for v in full_vals if v is not None and v > 0.0277)
    for seed, v in zip(SEEDS, full_vals):
        lines.append(f"  seed {seed}: PR-AUC={_fmt(v)}  -> {'beats XGBoost' if v and v > 0.0277 else 'does NOT beat XGBoost'}")
    lines.append(f"  GraphSAGE(100ep) beat XGBoost in {n_beats}/{len(SEEDS)} seeds.")
    lines.append("")
    if n_beats == len(SEEDS):
        lines.append("  -> CONFIRMED: increasing training budget to 100 epochs consistently lifts")
        lines.append("     GraphSAGE above the strongest tabular baseline. This is a defensible result.")
    elif n_beats == 0:
        lines.append("  -> NOT CONFIRMED: the single-seed 0.0501 result was an optimistic outlier;")
        lines.append("     across 5 seeds GraphSAGE(100ep) does not reliably beat XGBoost.")
    else:
        lines.append(f"  -> MIXED ({n_beats}/{len(SEEDS)}): GraphSAGE(100ep) can outperform XGBoost under")
        lines.append("     some initializations but is not consistently better — report this as reduced")
        lines.append("     stability rather than a clean win, per the pre-registered interpretation.")

    # --- verdict: graph ablation at 100 epochs ---
    lines.append("")
    lines.append("-" * 90)
    lines.append("Verdict 2: graph ablation at 100 epochs — does the graph still help at this budget?")
    lines.append("-" * 90)
    no_graph_vals = [m.get("pr_auc") for m in per_variant_metrics["GraphSAGE 100ep (no graph)"]]
    n_graph_wins = sum(
        1 for f, n in zip(full_vals, no_graph_vals)
        if f is not None and n is not None and f > n
    )
    for seed, f, n in zip(SEEDS, full_vals, no_graph_vals):
        lines.append(f"  seed {seed}: full={_fmt(f)}  no-graph={_fmt(n)}  "
                      f"-> {'graph helped' if (f is not None and n is not None and f > n) else 'graph did NOT help'}")
    lines.append(f"  Graph structure improved PR-AUC in {n_graph_wins}/{len(SEEDS)} seeds (100-epoch budget).")
    lines.append("")
    if n_graph_wins == len(SEEDS):
        lines.append("  -> CONFIRMED at 100 epochs too: graph topology contributes consistent, positive")
        lines.append("     signal beyond node features alone, at both training budgets tested (30, 100 ep).")
    else:
        lines.append("  -> The graph's contribution is less consistent at 100 epochs than at 30 —")
        lines.append("     investigate whether more training lets the no-graph variant memorize enough")
        lines.append("     node-feature signal to close the gap (possible overfitting risk at longer budgets).")

    # --- epoch milestone table from combined_history, full-graph variant only ---
    lines.append("")
    lines.append("-" * 90)
    lines.append("Validation PR-AUC trajectory (mean across 5 seeds) — full-graph variant")
    lines.append("Text substitute for 'Figure 1: Validation PR-AUC vs Epoch' (raw per-epoch")
    lines.append(f"curves saved to {CURVES_DIR}/*.csv for actual plotting)")
    lines.append("-" * 90)
    full_hist = combined_history[combined_history["variant"] == "GraphSAGE 100ep (full graph)"]
    milestones = [1, 5, 10, 20, 30, 50, 75, 100]
    lines.append(f"{'epoch':>8}{'val_pr_auc (mean)':>22}{'val_pr_auc (std)':>20}{'val_loss (mean)':>20}")
    for ep in milestones:
        sub = full_hist[full_hist["epoch"] == ep]
        if len(sub) == 0:
            continue
        pr_m, pr_s = sub["val_pr_auc"].mean(), sub["val_pr_auc"].std()
        vl_m = sub["val_loss"].mean()
        lines.append(f"{ep:>8}{_fmt(pr_m):>22}{_fmt(pr_s if pr_s == pr_s else 0):>20}{_fmt(vl_m):>20}")
    lines.append("")
    lines.append("If val_pr_auc is still rising at epoch 100, that's evidence there's more")
    lines.append("headroom left (i.e. don't yet conclude 100 epochs is the ceiling either).")

    lines.append("")
    lines.append("-" * 90)
    lines.append("Notes")
    lines.append("-" * 90)
    lines.append("- Per-epoch curves (train_loss, val_loss, val_pr_auc, val_roc_auc) for every")
    lines.append(f"  seed x variant run are saved as CSV under {CURVES_DIR}/.")
    lines.append("- 'Reference' rows (Rule-based / LogReg / RF / XGBoost / GraphSAGE 30ep) are")
    lines.append("  carried over from the prior 5-seed sweep, not re-run here.")
    lines.append("- No new features, no hyperparameter search performed in this run by design —")
    lines.append("  only the training-budget question (30ep vs 100ep) and the graph ablation")
    lines.append("  repeated at the new budget.")
    lines.append("=" * 90)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    main()
