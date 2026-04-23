import os
import pandas as pd
import wandb

ENTITY = "lionelhu926-ucla"
PROJECT = "nejm-brain-to-text"
SEEDS = list(range(5))

# Standard metric names used by the Transformer/B2T-24 GRU trainer
METRICS_DEFAULT = {
    "train_ctc_Loss": "train_loss",
    "ctc_loss_0":     "val_loss",
    "per_0":          "val_per",
}



# seed_overrides: {seed: run_id} — fetched directly by ID instead of by name
MODEL_NAMES = [
    ("gru_b2t_25_baseline_brainaudio", METRICS_DEFAULT, {}),
    ("gru_b2t_25_shared_input_brainaudio", METRICS_DEFAULT, {}),
]

out_dir = os.path.join(os.path.dirname(__file__), "losses")
os.makedirs(out_dir, exist_ok=True)

api = wandb.Api()

for entry in MODEL_NAMES:
    model_name, metrics = entry[0], entry[1]
    seed_overrides = entry[2] if len(entry) > 2 else {}

    wandb_keys = list(metrics.keys())
    per_key = next(k for k, v in metrics.items() if v == "val_per")

    # fetch name-based runs for seeds not covered by overrides
    name_seeds = [s for s in SEEDS if s not in seed_overrides]
    run_names = {f"{model_name}_seed_{s}": s for s in name_seeds}
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"display_name": {"$in": list(run_names.keys())}},
    )

    # deduplicate: keep the run with the lowest final val_per (converged run)
    best = {}
    for run in runs:
        if run.name not in run_names:
            continue
        if run.name not in best:
            best[run.name] = run
        else:
            curr_per = run.summary.get(per_key, float("inf"))
            prev_per = best[run.name].summary.get(per_key, float("inf"))
            if curr_per < prev_per:
                best[run.name] = run

    def extract(run, seed):
        df = run.history(pandas=True)
        df["seed"] = seed
        df = df.rename(columns={"_step": "epoch"})
        df = df.rename(columns=metrics)
        available = [c for c in ["seed", "epoch"] + list(metrics.values()) if c in df.columns]
        return df[available]

    frames = []
    for run_name, run in best.items():
        frames.append(extract(run, run_names[run_name]))

    # fetch seed overrides directly by run ID
    for seed, run_id in seed_overrides.items():
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        frames.append(extract(run, seed))

    if not frames:
        print(f"No runs found for {model_name}")
        continue

    combined = pd.concat(frames, ignore_index=True).sort_values(["seed", "epoch"])
    out_path = os.path.join(out_dir, f"{model_name}.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(frames)} seeds → {out_path}")
