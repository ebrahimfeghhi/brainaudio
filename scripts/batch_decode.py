#!/usr/bin/env python3
"""
Simple batch runner for run_decoder.py.

Run a specific family on a specific device:
    python scripts/batch_decode.py --family softsign --device cuda:0
    python scripts/batch_decode.py --family linear   --device cuda:1

Any additional arguments are passed through to run_decoder.py.
Example:
    python scripts/batch_decode.py --family softsign --device cuda:0 --beam-size 1200
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

LOGITS_BASE = "/home/ebrahim/data2/brain2text/b2t_25/logits"
LOGITS_VAL_FILENAME = "logits_val_chunk:1_context:20.npz"

MODEL_TEMPLATES = {
    "softsign": "neurips_b2t_25_causal_transformer_day_specific_softsign_seed_{seed}",
    "linear":   "neurips_b2t_25_causal_transformer_day_specific_seed_{seed}",
}
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
WEIGHTS_SUFFIX = "PER_25"

# =============================================================================


def build_runs(family_filter=None):
    val_paths, test_paths, train_paths, save_names = [], [], [], []
    for family, template in MODEL_TEMPLATES.items():
        if family_filter and family != family_filter:
            continue
        for seed in SEEDS:
            model_name = template.format(seed=seed)
            logits_dir = f"{LOGITS_BASE}/{model_name}_{WEIGHTS_SUFFIX}"
            val_paths.append(f"{logits_dir}/{LOGITS_VAL_FILENAME}")
            test_paths.append(None)
            train_paths.append(None)
            save_names.append(f"{model_name}_{WEIGHTS_SUFFIX}")
    return val_paths, test_paths, train_paths, save_names


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--family", choices=["softsign", "linear"], default=None,
                        help="Run only this model family. Runs all if omitted.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override decoder device (e.g. cuda:0, cuda:1).")
    known, extra_args = parser.parse_known_args()

    VAL_PATHS, TEST_PATHS, TRAIN_PATHS, SAVE_NAMES = build_runs(known.family)

    script_dir = Path(__file__).parent
    decoder_script = script_dir / "run_decoder.py"

    if not decoder_script.exists():
        print(f"Error: {decoder_script} not found")
        sys.exit(1)

    if known.device:
        extra_args = ["--device", known.device] + list(extra_args)

    timestamp = datetime.now().strftime("%m_%d_%H%M")
    label = f" (family={known.family})" if known.family else ""
    print(f"Total runs: {len(SAVE_NAMES)}{label}")

    for val_path, test_path, train_path, save_name in zip(VAL_PATHS, TEST_PATHS, TRAIN_PATHS, SAVE_NAMES):
        if val_path is None and test_path is None and train_path is None:
            print(f"Skipping {save_name}: no val, test, or train path")
            continue

        results_filename = f"{save_name}_{timestamp}"

        print(f"\n{'='*60}")
        print(f"Running: {save_name}")
        if val_path:
            print(f"Val: {val_path}")
        if test_path:
            print(f"Test: {test_path}")
        if train_path:
            print(f"Train: {train_path}")
        print(f"{'='*60}\n")

        cmd = [sys.executable, str(decoder_script)]
        if train_path:
            cmd.extend(["--logits-train-path", train_path])
        if val_path:
            cmd.extend(["--logits-val-path", val_path])
        if test_path:
            cmd.extend(["--logits-test-path", test_path])
        cmd.extend(["--results-filename", results_filename])
        cmd.extend(extra_args)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Warning: {save_name} failed with return code {result.returncode}")


if __name__ == "__main__":
    main()
