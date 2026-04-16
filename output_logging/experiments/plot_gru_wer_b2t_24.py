"""
Bar plot comparing WER for Original GRU vs GRU Shared Linear Layer for b2t_24.
Overlays individual seed points and marks paired t-test significance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# DATA  (seeds 0–9 for each condition)
# =============================================================================

wer_original = np.array([
    0.141557,  # seed 0
    0.138282,  # seed 1
    0.142831,  # seed 2
    0.145924,  # seed 3
    0.140284,  # seed 4
    0.141557,  # seed 5
    0.142649,  # seed 6
    0.141557,  # seed 7
    0.141739,  # seed 8
    0.139010,  # seed 9
])

wer_shared = np.array([
    0.1743,  # seed 0
    0.1801,  # seed 1
    0.1772,  # seed 2
    0.1788,  # seed 3
    0.1743,  # seed 4
    0.1764,  # seed 5
    0.1786,  # seed 6
    0.1728,  # seed 7
    0.1710,  # seed 8
    0.1719,  # seed 9
])

# =============================================================================
# PAIRED T-TEST
# =============================================================================

def paired_ttest(a, b):
    mask = ~(np.isnan(a) | np.isnan(b))
    _, p = stats.ttest_rel(a[mask], b[mask])
    return p

p_shared = paired_ttest(wer_original, wer_shared)

print(f"Original vs GRU Shared Linear Layer: p = {p_shared:.4f}  {'*' if p_shared < 0.05 else 'ns'}")

# =============================================================================
# PLOT
# =============================================================================

conditions  = ["Original GRU", "GRU Shared\nLinear Layer"]
all_wers    = [wer_original, wer_shared]
means       = [np.nanmean(w) for w in all_wers]
p_values    = [None, p_shared]

colors = ["#55A868", "#4C72B0"]
x = np.arange(len(conditions))
bar_width = 0.5

fig, ax = plt.subplots(figsize=(4.5, 5))

bars = ax.bar(x, means, width=bar_width, color=colors, alpha=0.8, zorder=2)

# Overlay individual seed points
rng = np.random.default_rng(42)
for i, wers in enumerate(all_wers):
    valid = wers[~np.isnan(wers)]
    jitter = rng.uniform(-0.08, 0.08, size=len(valid))
    ax.scatter(
        x[i] + jitter, valid,
        color="black", s=30, zorder=3, alpha=0.7, linewidths=0,
    )

ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=11)
ax.set_ylabel("Val WER (↓ better)", fontsize=12)
ax.set_title("Impact of Day-Specific Linear Layer\non GRU (B2T '24)", fontsize=11, pad=12)
ax.set_ylim(0, max(means) * 1.35)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=1)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# Significance brackets connecting Original (x=0) to each worse condition
bracket_gap = 0.004
tick_height = 0.002
bracket_y_base = max(means) * 1.12

for i, p in enumerate(p_values):
    if p is None or p >= 0.05:
        continue
    worse = means[i] > means[0]
    if not worse:
        continue

    y = bracket_y_base + (i - 1) * 0.025
    x0, x1 = x[0], x[i]

    # Horizontal line
    ax.plot([x0, x1], [y, y], color="black", lw=1.2)
    # Tick marks
    ax.plot([x0, x0], [y - tick_height, y], color="black", lw=1.2)
    ax.plot([x1, x1], [y - tick_height, y], color="black", lw=1.2)
    # Star
    mid = (x0 + x1) / 2
    ax.text(mid, y + bracket_gap, "*", ha="center", va="bottom", fontsize=12)

plt.tight_layout()
out_path = "gru_wer_b2t_24.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
plt.show()
