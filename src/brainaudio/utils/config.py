"""
config.py

Loads environment variables from the project's .env file and exposes
them as expanded constants. Does NOT create or modify directories.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the repo root
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)


# --- Brain2Text 2024 ---
B2T24_RAW_DIR = os.getenv("B2T24_RAW_DIR")
B2T24_SAVE_PATH = os.getenv("B2T24_SAVE_PATH")
B2T24_OUT_DIR = os.getenv("B2T24_OUT_DIR")
B2T24_PKL = os.getenv("B2T24_PKL")
B2T24_ALIGN_TRAIN = os.getenv("B2T24_ALIGN_TRAIN")
B2T24_ALIGN_VAL = os.getenv("B2T24_ALIGN_VAL")
B2T24_WITH_FA = os.getenv("B2T24_WITH_FA")

# --- Brain2Text 2025 ---
B2T25_DRYAD_DOI = os.getenv("B2T25_DRYAD_DOI")
B2T25_DRYAD_ROOT = os.getenv("B2T25_DRYAD_ROOT")
B2T25_DATA_DIR = os.getenv("B2T25_DATA_DIR")
B2T25_OUT_DIR = os.getenv("B2T25_OUT_DIR")
B2T25_PKL = os.getenv("B2T25_PKL")
B2T25_ALIGN_TRAIN = os.getenv("B2T25_ALIGN_TRAIN")
B2T25_ALIGN_VAL = os.getenv("B2T25_ALIGN_VAL")
B2T25_WITH_FA = os.getenv("B2T25_WITH_FA")

# --- Combined ---
B2T_COMBINED_OUT = os.getenv("B2T_COMBINED_OUT")
B2T_LOAD_MODEL_FOLDER = os.getenv("B2T_LOAD_MODEL_FOLDER")
B2T_DATASET_PATHS = os.getenv("B2T_DATASET_PATHS").split(",")
B2T_SAVE_PATHS = os.getenv("B2T_SAVE_PATHS").split(",")

# --- Misc ---
B2T_DATA3_24 = os.getenv("B2T_DATA3_24")
B2T_DATA3_25 = os.getenv("B2T_DATA3_25")
B2T_PTDECODER = os.getenv("B2T_PTDECODER")
B2T_CONFIG_DIR = os.getenv("B2T_CONFIG_DIR")
B2T_CUSTOM_CONFIG_YAML = os.getenv("B2T_CUSTOM_CONFIG_YAML")
B2T_E2E_CONFIG_YAML = os.getenv("B2T_E2E_CONFIG_YAML")

# --- WandB / Runtime ---
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
B2T_DEVICE = os.getenv("B2T_DEVICE")


def print_config() -> None:
    """Print loaded configuration values (for debugging only)."""
    print(f"Loaded .env from: {ENV_PATH}\n")
    env_vars = {
        "B2T24_RAW_DIR": B2T24_RAW_DIR,
        "B2T24_OUT_DIR": B2T24_OUT_DIR,
        "B2T24_PKL": B2T24_PKL,
        "B2T25_DATA_DIR": B2T25_DATA_DIR,
        "B2T25_OUT_DIR": B2T25_OUT_DIR,
        "B2T25_PKL": B2T25_PKL,
        "B2T_COMBINED_OUT": B2T_COMBINED_OUT,
        "B2T_LOAD_MODEL_FOLDER": B2T_LOAD_MODEL_FOLDER,
        "B2T_DATASET_PATHS": ", ".join(B2T_DATASET_PATHS),
        "B2T_SAVE_PATHS": ", ".join(B2T_SAVE_PATHS),
        "WANDB_PROJECT": WANDB_PROJECT,
        "WANDB_ENTITY": WANDB_ENTITY,
        "B2T_DEVICE": B2T_DEVICE,
    }
    for key, val in env_vars.items():
        print(f"{key:>25}: {val}")


if __name__ == "__main__":
    print_config()
