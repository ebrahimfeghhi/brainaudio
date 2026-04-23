"""
2x2 grid: Impact of Day-Specific Linear Layer
  Top row:    GRU B2T '24  |  GRU B2T '25
  Bottom row: Transformer B2T '24  |  Transformer B2T '25
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# DATA
# =============================================================================

# --- GRU B2T '24 ---
gru_24_original = np.array([0.140102, 0.141921, 0.140102, 0.137009, 0.139556,
                             0.142831, 0.148836, 0.134098, 0.143195, 0.139010])
gru_24_shared   = np.array([0.186317, 0.185044, 0.180131, 0.179585, 0.186681,
                             0.184680, 0.186317, 0.187773, 0.190684, 0.187409])

# --- GRU B2T '25 ---
gru_25_original = np.array([0.059185, 0.065749, 0.061196, 0.064690, 0.059926])
gru_25_shared   = np.array([0.065537, 0.067761, 0.068925, 0.068396, 0.066702])

# --- Transformer B2T '24 ---
tr_24_original     = np.array([0.114629, 0.116266, 0.110444, 0.118996, 0.113719,
                                0.119360, 0.125546, 0.118814, 0.123726, 0.116448])
tr_24_day_specific = np.array([0.126274, 0.134643, 0.134461, 0.134279, 0.133188,
                                0.125546, 0.132096, 0.135007, 0.130640, 0.122453])
tr_24_softsign     = np.array([0.140284, 0.147198, 0.140830, 0.135189, 0.136827,
                                0.138464, 0.136463, 0.137373, 0.143923, 0.133006])

# --- Transformer B2T '25 ---
tr_25_original     = np.array([0.045315, 0.048809, 0.049444, 0.055479, 0.051562,
                                0.047750, 0.049127, 0.059820, 0.052409, 0.049232])
tr_25_day_specific = np.array([0.081630, 0.079089, 0.083007, 0.084913, 0.084701,
                                0.080360, 0.082583, 0.078666, 0.083113, 0.072843])
tr_25_softsign     = np.array([0.094018, 0.094759, 0.112652, 0.097512, 0.099100,
                                0.094018, 0.090630, 0.088724, 0.096241, 0.093277])

# =============================================================================
# HELPERS
# =============================================================================

def paired_ttest(a, b):
    mask = ~(np.isnan(a) | np.isnan(b))
    _, p = stats.ttest_rel(a[mask], b[mask])
    return p

def draw_panel(ax, all_wers, conditions, p_values, colors, title):
    means = [np.nanmean(w) for w in all_wers]
    x = np.arange(len(conditions))

    ax.bar(x, means, width=0.5, color=colors, alpha=0.8, zorder=2)

    rng = np.random.default_rng(42)
    for i, wers in enumerate(all_wers):
        valid = wers[~np.isnan(wers)]
        jitter = rng.uniform(-0.08, 0.08, size=len(valid))
        ax.scatter(x[i] + jitter, valid, color="black", s=20, zorder=3,
                   alpha=0.7, linewidths=0)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Val WER (↓ better)", fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_ylim(0, max(means) * 1.40)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    bracket_gap   = max(means) * 0.02
    tick_height   = max(means) * 0.01
    bracket_y_base = max(means) * 1.15

    for i, p in enumerate(p_values):
        if p is None or p >= 0.05:
            continue
        if means[i] <= means[0]:
            continue
        y = bracket_y_base + (i - 1) * max(means) * 0.08
        x0, x1 = x[0], x[i]
        ax.plot([x0, x1], [y, y], color="black", lw=1.2)
        ax.plot([x0, x0], [y - tick_height, y], color="black", lw=1.2)
        ax.plot([x1, x1], [y - tick_height, y], color="black", lw=1.2)
        ax.text((x0 + x1) / 2, y + bracket_gap, "*",
                ha="center", va="bottom", fontsize=12)

# =============================================================================
# BUILD FIGURE
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top-left: GRU B2T '24
p = paired_ttest(gru_24_original, gru_24_shared)
draw_panel(axes[0, 0],
           [gru_24_original, gru_24_shared],
           ["Original GRU", "GRU Shared\nLinear Layer"],
           [None, p],
           ["#888888", "#4C72B0"],
           "GRU (B2T '24)")

# Top-right: GRU B2T '25
p = paired_ttest(gru_25_original, gru_25_shared)
draw_panel(axes[0, 1],
           [gru_25_original, gru_25_shared],
           ["Original GRU", "GRU Shared\nLinear Layer"],
           [None, p],
           ["#888888", "#4C72B0"],
           "GRU (B2T '25)")

# Bottom-left: Transformer B2T '24
p_ds = paired_ttest(tr_24_original, tr_24_day_specific)
draw_panel(axes[1, 0],
           [tr_24_original, tr_24_day_specific],
           ["Original", "Day Specific"],
           [None, p_ds],
           ["#4C72B0", "#888888"],
           "Transformer (B2T '24)")

# Bottom-right: Transformer B2T '25
p_ds = paired_ttest(tr_25_original, tr_25_day_specific)
draw_panel(axes[1, 1],
           [tr_25_original, tr_25_day_specific],
           ["Original", "Day Specific"],
           [None, p_ds],
           ["#4C72B0", "#888888"],
           "Transformer (B2T '25)")

fig.suptitle("Impact of Day-Specific Linear Layer on Neural Speech Decoders",
             fontsize=13, y=1.01)

plt.tight_layout()
out_path = "day_specific_wer_grid.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
