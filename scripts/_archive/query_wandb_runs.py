"""Query wandb runs by config parameters and date range."""

import wandb
from datetime import datetime

# Configure these
PROJECT = "brainaudio-neural-lm-fusion"
ENTITY = None  # Set to your wandb username/team if needed

# Filters
ACOUSTIC_SCALE = 0.3
START_DATE = datetime(2025, 1, 30)
END_DATE = datetime(2025, 1, 31, 23, 59, 59)


def query_runs():
    api = wandb.Api()

    # Get all runs from the project
    runs = api.runs(
        f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT,
        filters={
            "config.acoustic_scale": ACOUSTIC_SCALE,
            "created_at": {
                "$gte": START_DATE.isoformat(),
                "$lte": END_DATE.isoformat(),
            }
        }
    )

    print(f"Found {len(runs)} runs with acoustic_scale={ACOUSTIC_SCALE} from {START_DATE.date()} to {END_DATE.date()}\n")

    for run in runs:
        print(f"Run: {run.name}")
        print(f"  ID: {run.id}")
        print(f"  Created: {run.created_at}")
        print(f"  State: {run.state}")
        print(f"  Config:")
        for key in ['acoustic_scale', 'temperature', 'llm_weight', 'beam_size', 'encoder_model_name']:
            if key in run.config:
                print(f"    {key}: {run.config[key]}")
        if 'WER' in run.summary:
            print(f"  WER: {run.summary['WER']:.4f}")
        print()

    return runs


if __name__ == "__main__":
    runs = query_runs()
