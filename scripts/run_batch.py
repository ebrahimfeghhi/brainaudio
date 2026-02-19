#!/usr/bin/env python3
"""
Simple batch runner for run_decoder.py.

Edit VAL_PATHS, TEST_PATHS, and SAVE_NAMES below, then run:
    python scripts/run_batch.py [optional decoder args]

Any additional arguments are passed through to run_decoder.py.
Example:
    python scripts/run_batch.py --beam-size 1200 --acoustic-scale 0.5
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# EDIT THESE LISTS (must be same length, use None to skip val or test)
# =============================================================================

seed_list = [0,1,2,3,4,5,7,8]



TRAIN_PATHS = [None] * len(seed_list)
VAL_PATHS = [f"/home/ebrahim/data2/brain2text/b2t_25/logits/neurips_b2t_25_causal_transformer_v4_prob_1_seed_{seed}/logits_val_chunk:1_context:20.npz" for seed in seed_list]
TEST_PATHS = [f"/home/ebrahim/data2/brain2text/b2t_25/logits/neurips_b2t_25_causal_transformer_v4_prob_1_seed_{seed}/logits_test_chunk:1_context:20.npz" for seed in seed_list]

SAVE_NAMES = [f"" for _ in range(len(seed_list))] # no_finetuning  ,  no_variants  , no_delayed_fusion  ,  llama_3b
# =============================================================================


def main():
    assert len(VAL_PATHS) == len(SAVE_NAMES) == len(TEST_PATHS) == len(TRAIN_PATHS), (
        f"VAL_PATHS ({len(VAL_PATHS)}), TEST_PATHS ({len(TEST_PATHS)}), "
        f"TRAIN_PATHS ({len(TRAIN_PATHS)}), "
        f"and SAVE_NAMES ({len(SAVE_NAMES)}) must have same length"
    )

    script_dir = Path(__file__).parent
    decoder_script = script_dir / "run_decoder.py"

    if not decoder_script.exists():
        print(f"Error: {decoder_script} not found")
        sys.exit(1)

    # Extra args passed through to run_decoder.py
    extra_args = sys.argv[1:]

    timestamp = datetime.now().strftime("%m_%d_%H%M")

    for val_path, test_path, train_path, save_name in zip(VAL_PATHS, TEST_PATHS, TRAIN_PATHS, SAVE_NAMES):
        if val_path is None and test_path is None and train_path is None:
            print(f"Skipping {save_name}: no val, test, or train path")
            continue

        results_filename = f"{save_name}_{timestamp}"

        print(f"\n{'='*60}")
        print(f"Running: {save_name}")
        if train_path:
            print(f"Train: {train_path}")
        if val_path:
            print(f"Val: {val_path}")
        if test_path:
            print(f"Test: {test_path}")
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
