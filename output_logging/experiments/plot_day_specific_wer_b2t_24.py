"""
Bar plot comparing WER across three conditions for b2t_24 day-specific transformer.
Overlays individual seed points and marks paired t-test significance vs Original.

Fill in seeds 8 and 9 for day_specific and softsign once those runs complete.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# DATA  (seeds 0–9 for each condition)
# =============================================================================

wer_original = np.array([
    0.114629,  # seed 0
    0.116266,  # seed 1
    0.110444,  # seed 2
    0.118996,  # seed 3
    0.113719,  # seed 4
    0.119360,  # seed 5
    0.125546,  # seed 6
    0.118814,  # seed 7
    0.123726,  # seed 8
    0.116448,  # seed 9
])

wer_day_specific = np.array([
    0.126274,  # seed 0
    0.134643,  # seed 1
    0.134461,  # seed 2
    0.134279,  # seed 3
    0.133188,  # seed 4
    0.125546,  # seed 5
    0.132096,  # seed 6
    0.135007,  # seed 7
    0.130640,  # seed 8
    0.122453,  # seed 9
])

wer_softsign = np.array([
    0.140284,  # seed 0
    0.147198,  # seed 1
    0.140830,  # seed 2
    0.135189,  # seed 3
    0.136827,  # seed 4
    0.138464,  # seed 5
    0.136463,  # seed 6
    0.137373,  # seed 7
    0.143923,  # seed 8
    0.133006,  # seed 9
])

# =============================================================================
# PAIRED T-TESTS  (drop seeds where either condition has NaN)
# =============================================================================

def paired_ttest(a, b):
    mask = ~(np.isnan(a) | np.isnan(b))
    _, p = stats.ttest_rel(a[mask], b[mask])
    return p

p_day    = paired_ttest(wer_original, wer_day_specific)
p_soft   = paired_ttest(wer_original, wer_softsign)

print(f"Original vs Day Specific:           p = {p_day:.4f}  {'*' if p_day < 0.05 else 'ns'}")
print(f"Original vs Day Specific + Softsign: p = {p_soft:.4f}  {'*' if p_soft < 0.05 else 'ns'}")

# =============================================================================
# PLOT
# =============================================================================

conditions  = ["Original", "Day Specific", "Day Specific\n+ Softsign"]
all_wers    = [wer_original, wer_day_specific, wer_softsign]
means       = [np.nanmean(w) for w in all_wers]
p_values    = [None, p_day, p_soft]

colors = ["#4C72B0", "#DD8452", "#55A868"]
x = np.arange(len(conditions))
bar_width = 0.5

fig, ax = plt.subplots(figsize=(7, 5))

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
ax.set_title("Impact of Day-Specific Linear Layer on Transformer (B2T '24)", fontsize=13, pad=18)
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
out_path = "day_specific_wer_b2t_24.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
plt.show()
