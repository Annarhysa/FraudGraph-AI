"""
Phase 6, step 4: generate the final set of figures for the thesis/paper from
the already-frozen results (no training, pure plotting). Deliberately limited
to 5 figures per the roadmap: model comparison, temporal robustness, edge
ablation, training curves, fraud-community enrichment.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def fig1_model_comparison():
    models = ["Rule-based", "Logistic\nRegression", "MLP", "Card-only\n(GraphSAGE)",
              "No Graph\n(GraphSAGE)", "Random\nForest", "Full Graph\n(GraphSAGE)",
              "Relation-\nAware", "XGBoost\n(tuned)", "Merchant-only\n(GraphSAGE)"]
    pr_auc = [0.0034, 0.0040, 0.0040, 0.0113, 0.0151, 0.0160, 0.0549, 0.0575, 0.0277, 0.0858]
    err = [0, 0, 0.0003, 0.0004, 0.0003, 0.0003, 0.0067, 0.0053, 0, 0.0098]
    colors = ["#9e9e9e"] * 3 + ["#e08283"] * 1 + ["#7fb3d5"] * 1 + ["#9e9e9e"] + \
              ["#7fb3d5"] * 2 + ["#f4b942"] + ["#4a90a4"]

    order = sorted(range(len(pr_auc)), key=lambda i: pr_auc[i])
    models = [models[i] for i in order]
    pr_auc = [pr_auc[i] for i in order]
    err = [err[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(models, pr_auc, xerr=err, color=colors, capsize=3)
    ax.set_xlabel("PR-AUC (Window 0 test split)")
    ax.set_title("Model comparison — PR-AUC, Window 0\n(error bars = std across 5 seeds where applicable)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_model_comparison.png")
    plt.close(fig)


def fig2_temporal_robustness():
    windows = ["Window 0\n(orig. split)", "Window 1\n(Jun-Oct)", "Window 2\n(Aug-Oct)", "Window 3\n(Oct)"]
    xgb = [0.0277, 0.0445, 0.0562, 0.0663]
    no_graph = [0.0151, 0.0368, 0.0445, 0.0611]
    no_graph_err = [0.0003, 0.0004, 0.0013, 0.0002]
    full = [0.0549, 0.1670, 0.1824, 0.2139]
    full_err = [0.0067, 0.0047, 0.0059, 0.0093]
    merch = [0.0858, 0.2293, 0.2491, 0.3065]
    merch_err = [0.0098, 0.0206, 0.0273, 0.0355]

    x = range(len(windows))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, xgb, marker="o", label="XGBoost (tuned)", color="#f4b942")
    ax.errorbar(x, no_graph, yerr=no_graph_err, marker="o", label="No Graph", color="#9e9e9e", capsize=3)
    ax.errorbar(x, full, yerr=full_err, marker="o", label="Full Graph", color="#7fb3d5", capsize=3)
    ax.errorbar(x, merch, yerr=merch_err, marker="o", label="Merchant-only", color="#4a90a4", capsize=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(windows)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Temporal robustness across 4 chronological windows")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_temporal_robustness.png")
    plt.close(fig)


def fig3_edge_ablation():
    conditions = ["No Graph", "Card-only", "Full Graph\n(homogeneous)",
                  "Relation-\nAware", "Merchant-only"]
    pr_auc = [0.0151, 0.0113, 0.0549, 0.0575, 0.0858]
    err = [0.0003, 0.0004, 0.0067, 0.0053, 0.0098]
    colors = ["#9e9e9e", "#e08283", "#7fb3d5", "#7fb3d5", "#4a90a4"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(conditions, pr_auc, yerr=err, color=colors, capsize=4)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Edge-type ablation, Window 0\n(100 epochs, 5 seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_edge_ablation.png")
    plt.close(fig)


def fig4_training_curves():
    curves_dir = Path("results/curves")
    combined_path = curves_dir / "all_runs_combined.csv"
    if not combined_path.exists():
        print(f"[skip fig4] {combined_path} not found")
        return
    df = pd.read_csv(combined_path)

    fig, ax = plt.subplots(figsize=(8, 5.8))
    for variant, color in [("GraphSAGE 100ep (full graph)", "#7fb3d5"),
                            ("GraphSAGE 100ep (no graph)", "#9e9e9e")]:
        sub = df[df["variant"] == variant]
        if len(sub) == 0:
            continue
        grouped = sub.groupby("epoch")["val_pr_auc"].agg(["mean", "std"]).reset_index()
        ax.plot(grouped["epoch"], grouped["mean"], label=variant.replace("GraphSAGE 100ep ", ""), color=color)
        ax.fill_between(grouped["epoch"], grouped["mean"] - grouped["std"], grouped["mean"] + grouped["std"],
                         color=color, alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation PR-AUC (AP)")
    ax.set_title(
        "Training dynamics: full graph vs no graph (Window 0, mean +/- std over 5 seeds)\n"
        "NOTE: 'full graph' curve predates the merchant_offset bugfix — shown only to illustrate\n"
        "that val AP was still rising at epoch 100; final PR-AUC values are superseded (see Table 1)",
        fontsize=9,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_training_curves.png")
    plt.close(fig)


def fig5_fraud_ring_capture():
    top_n = [5, 10, 20]
    # From fraud_ring_evaluation_results.txt (post-retrain, corrected production model).
    by_fraud_count = [59.32, 66.22, 67.56]
    by_avg_risk = [0.57, 13.90, 16.29]

    x = range(len(top_n))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar([i - width / 2 for i in x], by_fraud_count, width, label="Ranked by fraud count", color="#4a90a4")
    ax.bar([i + width / 2 for i in x], by_avg_risk, width, label="Ranked by avg_risk_score\n(app's current default)", color="#e08283")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Top {n}" for n in top_n])
    ax.set_ylabel("% of all fraud transactions captured")
    ax.set_title("Fraud-ring capture: ranking method matters")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_fraud_ring_capture.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_model_comparison()
    fig2_temporal_robustness()
    fig3_edge_ablation()
    fig4_training_curves()
    fig5_fraud_ring_capture()
    print(f"Figures written to {OUT}/")
