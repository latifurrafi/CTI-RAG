# plot_risk_coverage.py — Fig. 2 of the camera-ready: precision vs. coverage
# over the full 200-query run. The gate does not change accepted
# answers, so each threshold is evaluated by gating System B's saved answers.
# Run: python plot_risk_coverage.py
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = pd.read_csv("results/results_B. Hybrid RAG (no abstain).csv")
A = pd.read_csv("results/results_A. Naive RAG.csv")
self_abs = B.rouge_l == 0          # the LLM's own "Insufficient context" replies
TH = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
cov, prec = [], []
for t in TH:
    ans = (B.confidence >= t) & ~self_abs
    cov.append(100 * ans.mean()); prec.append(100 * B.correct[ans].mean())

plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42,  # TrueType, for IEEE PDF eXpress
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.45, 2.1))
ax.axhline(100 * A.correct.mean(), color="#8a8984", lw=1, ls=(0, (4, 3)))
ax.text(100, 100 * A.correct.mean() - 1.0, "Naive RAG (always answers), 70.0%",
        ha="left", va="top", color="#52514e", fontsize=7)
ax.plot(cov, prec, color="#2a78d6", lw=2, marker="o", ms=5,
        markeredgecolor="white", markeredgewidth=1, zorder=3)
for t, x, y in zip(TH, cov, prec):
    if t in (0.20, 0.30, 0.40, 0.50):
        ax.annotate(rf"$\theta$={t:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color="#0b0b0b")
ax.set_xlim(102, 12); ax.set_ylim(64, 90)
ax.set_xlabel("Coverage (% of queries answered)")
ax.set_ylabel("Precision@Answered (%)")
ax.grid(axis="y", color="#e5e4e0", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
fig.savefig("figures/fig_risk_coverage.pdf"); fig.savefig("figures/fig_risk_coverage.png", dpi=300)
print("saved figures/fig_risk_coverage.{pdf,png}")
