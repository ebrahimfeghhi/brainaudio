#!/usr/bin/env python3
"""
Simple batch runner for run_decoder.py.

Example:
    python scripts/batch_decode.py \
        --dataset b2t_24 \
        --model-mode gru \
        --logits-base /data2/brain2text/b2t_24/logits \
        --model-template "bidirectional_gru_seed_{seed}" \
        --seeds 0 1 2 3 4 \
        --val \
        --device cuda:0

Any additional arguments are passed through to run_decoder.py.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path



def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", choices=["b2t_24", "b2t_25"], required=True,
                        help="Dataset to use.")
    parser.add_argument("--model-mode", choices=["gru", "transformer"], required=True,
                        help="Model mode (sets results directory in decoder_config).")
    parser.add_argument("--logits-base", type=str, required=True,
                        help="Base directory containing per-model logits folders.")
    parser.add_argument("--model-template", type=str, required=True,
                        help="Logits folder name template with {seed} placeholder.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="Seeds to run.")
    parser.add_argument("--val", action="store_true",
                        help="Run on val split.")
    parser.add_argument("--test", action="store_true",
                        help="Run on test split.")
    parser.add_argument("--val-filename", type=str, default="logits_val.npz",
                        help="Val logits filename (default: logits_val.npz).")
    parser.add_argument("--test-filename", type=str, default="logits_test.npz",
                        help="Test logits filename (default: logits_test.npz).")
    parser.add_argument("--device", type=str, default=None,
                        help="Decoder device (e.g. cuda:0, cuda:1).")
    known, extra_args = parser.parse_known_args()



    if not known.val and not known.test:
        print("Error: specify at least one of --val or --test")
        sys.exit(1)

    os.environ["B2T_DATASET"] = known.dataset
    os.environ["B2T_MODEL_MODE"] = known.model_mode

    script_dir = Path(__file__).parent
    decoder_script = script_dir / "run_decoder.py"
    if not decoder_script.exists():
        print(f"Error: {decoder_script} not found")
        sys.exit(1)

    if known.device:
        extra_args = ["--device", known.device] + list(extra_args)

    timestamp = datetime.now().strftime("%m_%d_%H%M")
    splits = "+".join(s for s, f in [("val", known.val), ("test", known.test)] if f)
    print(f"Total runs: {len(known.seeds)} (dataset={known.dataset}, mode={known.model_mode}, splits={splits})")

    for seed in known.seeds:
        model_name = known.model_template.format(seed=seed)
        logits_dir = Path(known.logits_base) / model_name
        val_path  = logits_dir / known.val_filename  if known.val  else None
        test_path = logits_dir / known.test_filename if known.test else None

        results_filename = timestamp

        print(f"\n{'='*60}")
        print(f"Running: {model_name}")
        if val_path:
            print(f"Val:  {val_path}")
        if test_path:
            print(f"Test: {test_path}")
        print(f"{'='*60}\n")

        cmd = [sys.executable, str(decoder_script)]
        if val_path:
            cmd.extend(["--logits-val-path", str(val_path)])
        if test_path:
            cmd.extend(["--logits-test-path", str(test_path)])
        cmd.extend(["--results-filename", results_filename])
        cmd.extend(extra_args)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Warning: {model_name} failed with return code {result.returncode}")


if __name__ == "__main__":
    main()
