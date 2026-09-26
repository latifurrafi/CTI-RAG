import matplotlib.pyplot as plt
import numpy as np

# Set font to Times New Roman for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Data
methods = ['Naive RAG', 'Hybrid RAG', 'Conf-Aware RAG']
recall = [0.8100, 0.7950, 0.7950]
rouge = [0.3299, 0.2983, 0.2803]
ece = [0.0384, 0.3094, 0.2594]

# Colors: distinct across charts
colors = ['#808080', '#4472C4', '#ED7D31']  # Gray, Blue, Orange

# Metrics and titles
metrics = [recall, rouge, ece]
titles = ['(a) Recall@5 — Retrieval Quality', '(b) ROUGE-L — Generation Quality', '(c) ECE — Calibration\n(Lower is Better)']
ylabels = ['Recall@5 Score', 'ROUGE-L Score', 'ECE Score']

# Create subplots with larger size
fig, axs = plt.subplots(1, 3, figsize=(26, 10), constrained_layout=True)

for i, ax in enumerate(axs):
    bars = ax.bar(methods, metrics[i], color=colors, width=0.55, edgecolor='black', linewidth=1.5)
    ax.set_title(titles[i], fontsize=18, fontweight='bold', pad=18)
    ax.set_ylabel(ylabels[i], fontsize=15, fontweight='bold')
    ax.set_xlabel('System', fontsize=15, fontweight='bold')
    ax.set_ylim(0, max(metrics[i]) * 1.35)  # Adjust y-limit for labels
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add value labels on top of bars
    for bar, value in zip(bars, metrics[i]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(metrics[i]) * 0.025,
                f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Add legend box below each subplot, beneath the x-axis label
    handles = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', linewidth=1.2) for color in colors]
    ax.legend(handles, methods, loc='upper center', bbox_to_anchor=(0.5, -0.35), fontsize=11, frameon=True,
              fancybox=True, shadow=True, title='Systems', title_fontsize=12,
              framealpha=0.95, edgecolor='black', ncol=1)
# Overall title
fig.suptitle('System Comparison: Naive RAG vs Hybrid RAG vs Confidence-Aware RAG', 
             fontsize=18, fontweight='bold', y=0.98)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/Users/md.latifurrahmanrafi/Desktop/CTI-RAG/cti_rag/figures/fig_comparison_bars.pdf', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/fig_comparison_bars.pdf")

# Also save as PNG
plt.savefig('/Users/md.latifurrahmanrafi/Desktop/CTI-RAG/cti_rag/figures/fig_comparison_bars.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/fig_comparison_bars.png")

plt.show()