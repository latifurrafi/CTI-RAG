# ─────────────────────────────────────────────────────────────────
# figures.py  —  Generate all paper figures
# Produces:
#   Fig 1: Calibration curve (reliability diagram)
#   Fig 2: Confidence score distribution
#   Fig 3: Recall@5 comparison bar chart
#   Fig 4: ROUGE-L comparison bar chart
# ─────────────────────────────────────────────────────────────────

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
from config import FIGURES_DIR, RESULTS_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path
from config import FIGURES_DIR, RESULTS_DIR
from confidence import calibration_curve_data, compute_ece


COLORS = {
    "naive_rag":   "#94A3B8",
    "fixed_chunks":"#F97316",
    "semantic_rag":"#0EA5C9",
    "perfect":     "#16A669",
}


# ══════════════════════════════════════════════════════════════════
# FIG 1 — CALIBRATION CURVE (Reliability Diagram)
# ══════════════════════════════════════════════════════════════════

def plot_calibration_curve(results_csv: Path, label: str = "Proposed System"):
    """
    Reliability diagram: predicted confidence vs actual accuracy per bin.
    Perfect calibration = diagonal line.
    """
    df          = pd.read_csv(results_csv)
    confs       = df["confidence"].tolist()
    corrects    = df["correct"].astype(bool).tolist()
    cal_data    = calibration_curve_data(confs, corrects, n_bins=10)
    ece         = compute_ece(confs, corrects)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color=COLORS["perfect"],
            linewidth=1.5, label="Perfect calibration", alpha=0.8)

    # Actual calibration bars
    ax.bar(cal_data["bin_centers"],
           cal_data["accuracy"],
           width=0.08,
           color=COLORS["semantic_rag"],
           alpha=0.75,
           label=f"{label} (ECE={ece:.4f})",
           edgecolor="white",
           linewidth=0.5)

    # Gap fill (overconfidence indicator)
    for bc, acc, conf in zip(cal_data["bin_centers"],
                              cal_data["accuracy"],
                              cal_data["confidence"]):
        if conf > acc:
            ax.bar(bc, conf - acc, bottom=acc,
                   width=0.08, color=COLORS["naive_rag"],
                   alpha=0.3, edgecolor="none")

    ax.set_xlabel("Confidence Score", fontsize=11)
    ax.set_ylabel("Accuracy (Fraction Correct)", fontsize=11)
    ax.set_title("Reliability Diagram — Confidence Calibration", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    out = FIGURES_DIR / f"fig_calibration_{label.replace(' ', '_')}.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved → {out}")


# ══════════════════════════════════════════════════════════════════
# FIG 2 — CONFIDENCE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════

def plot_confidence_distribution(results_csvs: dict):
    """
    Histogram of confidence scores for each system.
    results_csvs: {"System Name": Path}
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    color_list = list(COLORS.values())

    for i, (label, csv_path) in enumerate(results_csvs.items()):
        if not Path(csv_path).exists():
            continue
        df   = pd.read_csv(csv_path)
        confs = df["confidence"]
        ax.hist(confs, bins=20, alpha=0.6,
                color=color_list[i % len(color_list)],
                label=label, edgecolor="white", linewidth=0.5)

    ax.axvline(0.15, color="red", linestyle="--", linewidth=1.2,
               label="Low-conf threshold (0.15)")
    ax.set_xlabel("Confidence Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Confidence Score Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    out = FIGURES_DIR / "fig_confidence_distribution.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved → {out}")


# ══════════════════════════════════════════════════════════════════
# FIG 3 + 4 — BAR CHARTS (Recall@5, ROUGE-L)
# ══════════════════════════════════════════════════════════════════

def plot_comparison_bars(all_metrics: list[dict]):
    """
    Side-by-side bar charts: Recall@5 and ROUGE-L across systems.
    """
    labels    = [m["label"] for m in all_metrics]
    recall    = [m["recall_at_k"]  for m in all_metrics]
    rouge     = [m["rouge_l_mean"] for m in all_metrics]
    ece       = [m["ece"]          for m in all_metrics]

    x         = np.arange(len(labels))
    bar_colors= [COLORS.get(l.lower().replace(" ", "_"),
                  COLORS["semantic_rag"]) for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    metrics_list = [
        ("Recall@5",  recall, "Recall@5 — Retrieval Quality"),
        ("ROUGE-L",   rouge,  "ROUGE-L — Generation Quality"),
        ("ECE",       ece,    "ECE — Calibration (lower = better)"),
    ]

    for ax, (ylabel, vals, title) in zip(axes, metrics_list):
        bars = ax.bar(x, vals, color=bar_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.8, width=0.55)

        # value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.2 + 0.01)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "System Comparison: Naive RAG vs Fixed Chunks vs Semantic RAG",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    out = FIGURES_DIR / "fig_comparison_bars.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved → {out}")

# ══════════════════════════════════════════════════════════════════
# FIG 5 — ABLATION CHART
# ══════════════════════════════════════════════════════════════════

def plot_ablation(ablation_metrics: list[dict]):
    """
    Ablation study: show contribution of each component.
    ablation_metrics: list of {"label", "recall_at_k", "rouge_l_mean", "ece"}
    """
    labels = [m["label"] for m in ablation_metrics]
    rouge  = [m["rouge_l_mean"] for m in ablation_metrics]
    recall = [m["recall_at_k"]  for m in ablation_metrics]

    x  = np.arange(len(labels))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w/2, recall, w, label="Recall@5",
                color=COLORS["semantic_rag"], alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, rouge,  w, label="ROUGE-L",
                color=COLORS["fixed_chunks"], alpha=0.85, edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Ablation Study — Component Contribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_ablation.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved → {out}")


if __name__ == "__main__":
    # Test with mock metrics
    mock = [
        {"label": "Naive RAG",    "recall_at_k": 0.42, "rouge_l_mean": 0.28, "ece": 0.22},
        {"label": "Fixed Chunks", "recall_at_k": 0.55, "rouge_l_mean": 0.34, "ece": 0.18},
        {"label": "Semantic RAG", "recall_at_k": 0.68, "rouge_l_mean": 0.41, "ece": 0.12},
    ]
    plot_comparison_bars(mock)
    ablation = [
        {"label": "No Chunking",          "recall_at_k": 0.42, "rouge_l_mean": 0.28, "ece": 0.22},
        {"label": "+ Fixed Chunking",     "recall_at_k": 0.55, "rouge_l_mean": 0.34, "ece": 0.18},
        {"label": "+ Semantic Chunking",  "recall_at_k": 0.63, "rouge_l_mean": 0.38, "ece": 0.15},
        {"label": "+ Confidence Score",   "recall_at_k": 0.68, "rouge_l_mean": 0.41, "ece": 0.12},
    ]
    plot_ablation(ablation)
    print("Mock figures generated.")
