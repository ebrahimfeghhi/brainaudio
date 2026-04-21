#!/usr/bin/env python3
"""
Batch runner: optionally generates logits, then runs the decoder across seeds.

If --logits-base is omitted, logits are generated automatically before decoding
and saved to {brain2text-dir}/brain2text/{dataset}/logits/.

If --logits-base is provided, logit generation is skipped and the existing
logits at that path are used directly.

Example (generate logits + decode):
    uv run scripts/batch_decode.py \
        --dataset b2t_24 \
        --model-mode gru \
        --base-path /home/user \
        --brain2text-dir /home/user/data2 \
        --model-template "gru_b2t_24_baseline_brainaudio_seed_{seed}" \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --val \
        --device cuda:0

Example (decode only, logits already exist):
    uv run scripts/batch_decode.py \
        --dataset b2t_24 \
        --model-mode gru \
        --base-path /home/user \
        --brain2text-dir /home/user/data2 \
        --logits-base /home/user/data2/brain2text/b2t_24/logits \
        --model-template "gru_b2t_24_baseline_brainaudio_seed_{seed}" \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
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
    parser.add_argument("--base-path", type=str, required=True,
                        help="Base path for results and adapter directories (e.g. /home/user).")
    parser.add_argument("--brain2text-dir", type=str, required=True,
                        help="Directory containing the brain2text/ folder (e.g. /home/user/data2).")
    parser.add_argument("--logits-base", type=str, default=None,
                        help="Base directory containing per-model logits folders. "
                             "If omitted, logits are generated automatically.")
    parser.add_argument("--model-template", type=str, required=True,
                        help="Model name template with {seed} placeholder.")
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
                        help="Device for both logit generation and decoding (e.g. cuda:0).")
    known, extra_args = parser.parse_known_args()

    if not known.val and not known.test:
        print("Error: specify at least one of --val or --test")
        sys.exit(1)

    os.environ["B2T_DATASET"] = known.dataset
    os.environ["B2T_MODEL_MODE"] = known.model_mode
    os.environ["B2T_BASE_PATH"] = known.base_path
    os.environ["B2T_BRAIN2TEXT_DIR"] = known.brain2text_dir

    script_dir = Path(__file__).parent
    decoder_script = script_dir / "run_decoder.py"
    logits_script = script_dir / "generate_logits.py"

    if not decoder_script.exists():
        print(f"Error: {decoder_script} not found")
        sys.exit(1)

    generate_logits = known.logits_base is None
    logits_base = known.logits_base or f"{known.brain2text_dir}/brain2text/{known.dataset}/logits"

    if generate_logits and not logits_script.exists():
        print(f"Error: {logits_script} not found")
        sys.exit(1)

    if known.device:
        extra_args = ["--device", known.device] + list(extra_args)

    partitions = []
    if known.val:
        partitions.append(("val", known.val_filename))
    if known.test:
        partitions.append(("test", known.test_filename))

    timestamp = datetime.now().strftime("%m_%d_%H%M")
    splits = "+".join(p for p, _ in partitions)
    mode = "generate+decode" if generate_logits else "decode only"
    print(f"Total runs: {len(known.seeds)} (dataset={known.dataset}, mode={known.model_mode}, splits={splits}, {mode})")

    for seed in known.seeds:
        model_name = known.model_template.format(seed=seed)

        print(f"\n{'='*60}")
        print(f"Running: {model_name}")
        print(f"{'='*60}\n")

        if generate_logits:
            for partition, _ in partitions:
                logits_cmd = [
                    sys.executable, str(logits_script),
                    "--model-name", model_name,
                    "--dataset", known.dataset,
                    "--model-type", known.model_mode,
                    "--partition", partition,
                    "--device", known.device or "cuda:0",
                ]
                result = subprocess.run(logits_cmd)
                if result.returncode != 0:
                    print(f"Warning: logit generation failed for {model_name} ({partition}), skipping decode.")
                    continue

        logits_dir = Path(logits_base) / model_name
        cmd = [sys.executable, str(decoder_script)]
        if known.val:
            cmd.extend(["--logits-val-path", str(logits_dir / known.val_filename)])
        if known.test:
            cmd.extend(["--logits-test-path", str(logits_dir / known.test_filename)])
        cmd.extend(["--results-filename", timestamp])
        cmd.extend(extra_args)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Warning: {model_name} decoder failed with return code {result.returncode}")


if __name__ == "__main__":
    main()
