import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOSSES_DIR = os.path.join(os.path.dirname(__file__), "losses")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "experiments")

MODELS_24 = {
    "neurips_b2t_24_chunked_unidirectional_day_specific_transformer": {
        "label": "Day-Specific",
        "color": "#888888",
    },
    "neurips_b2t_24_chunked_unidirectional_transformer_5to20_sec": {
        "label": "Original",
        "color": "#4C72B0",
    },
}

MODELS_25 = {
    "neurips_b2t_25_causal_transformer_day_specific": {
        "label": "Day-Specific",
        "color": "#888888",
    },
    "neurips_b2t_25_causal_transformer_v4_prob_1": {
        "label": "Original",
        "color": "#4C72B0",
    },
}

MODELS_25_GRU = {
    "baseline_rnn_ucd_npl": {
        "label": "Original (day-specific)",
        "color": "#888888",
    },
    "gru_b2t25_shared_input": {
        "label": "Shared input",
        "color": "#4C72B0",
    },
}

MODELS_24_GRU = {
    "gru_b2t_24_baseline_brainaudio": {
        "label": "Original (day-specific)",
        "color": "#888888",
    },
    "gru_b2t_24_shared_input_brainaudio": {
        "label": "Shared input",
        "color": "#4C72B0",
    },
}

METRICS_TRANSFORMER = {
    "train_ctc_Loss": "Training CTC Loss",
    "ctc_loss_0":     "Validation CTC Loss",
    "per_0":          "Validation PER",
}

METRICS_GRU = {
    "train_loss": "Training Loss",
    "val_loss":   "Validation Loss",
    "val_per":    "Validation PER",
}

def plot_model(ax, df, color, label):
    col = ax._metric
    df = df.dropna(subset=[col])
    df = df.groupby(["seed", "epoch"])[col].mean().reset_index()
    pivot = df.pivot(index="epoch", columns="seed", values=col)
    mean  = pivot.mean(axis=1)
    sem   = pivot.sem(axis=1)
    for seed in pivot.columns:
        ax.plot(pivot.index, pivot[seed], color=color, linewidth=0.6, alpha=0.2)
    ax.plot(mean.index, mean.values, color=color, linewidth=2.0, label=label)
    ax.fill_between(mean.index, mean - sem, mean + sem, alpha=0.4, color=color)

def make_grid(models_dict, title, out_filename, metrics=METRICS_TRANSFORMER, ylims=None):
    data = {name: pd.read_csv(os.path.join(LOSSES_DIR, f"{name}.csv"))
            for name in models_dict}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (col, metric_title) in zip(axes, metrics.items()):
        ax._metric = col
        for name, cfg in models_dict.items():
            plot_model(ax, data[name], cfg["color"], cfg["label"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_title)
        ax.set_title(metric_title)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        if ylims and col in ylims:
            ax.set_ylim(ylims[col])

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, out_filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

make_grid(MODELS_24_GRU, "B2T '24 GRU — mean ± SEM per model", "gru_b2t_24_loss_curves.png",
          metrics=METRICS_GRU)
make_grid(MODELS_25_GRU, "B2T '25 GRU — mean ± SEM per model", "gru_b2t_25_loss_curves.png",
          metrics=METRICS_GRU)
make_grid(MODELS_24, "B2T '24 Transformer — mean ± SEM per model", "neurips_b2t_24_loss_curves.png")
make_grid(MODELS_25, "B2T '25 Transformer — mean ± SEM per model", "neurips_b2t_25_loss_curves.png",
          ylims={"train_ctc_Loss": (0, 3.5), "ctc_loss_0": (0, 5.0), "per_0": (0, 1.0)})
