#!/usr/bin/env python3
"""
Run model inference and save logits for downstream decoding.

Example (standalone):
    uv run scripts/generate_logits.py \
        --model-name gru_b2t_24_baseline_brainaudio_seed_0 \
        --dataset b2t_24 \
        --model-type gru \
        --partition val \
        --device cuda:0 \
        --base-path /home/user \
        --brain2text-dir /home/user/data2

When called from batch_decode.py, --base-path and --brain2text-dir are
inherited automatically from the B2T_BASE_PATH / B2T_BRAIN2TEXT_DIR env vars.
"""

import argparse
import os
from brainaudio.inference.load_model_generate_logits import (
    load_transformer_model,
    load_gru_model,
    generate_and_save_logits,
)

WEIGHTS_FILE = "modelWeights_PER"

# Best eval configs per dataset (Transformer only)
EVAL_CONFIGS = {
    "b2t_24": {"chunk_size": 1, "context_sec": 7.5},
    "b2t_25": {"chunk_size": 1, "context_sec": 20},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True,
                        help="Exact model name (no {seed} placeholder).")
    parser.add_argument("--dataset", choices=["b2t_24", "b2t_25"], required=True)
    parser.add_argument("--model-type", choices=["gru", "transformer"], required=True)
    parser.add_argument("--partition", choices=["val", "test"], default="val")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--base-path", type=str, default=os.environ.get("B2T_BASE_PATH"),
                        help="Base path for results (e.g. /home/user). Can also set via B2T_BASE_PATH.")
    parser.add_argument("--brain2text-dir", type=str, default=os.environ.get("B2T_BRAIN2TEXT_DIR"),
                        help="Directory containing brain2text/ folder. Can also set via B2T_BRAIN2TEXT_DIR.")
    args = parser.parse_args()

    if not args.base_path:
        raise RuntimeError("--base-path or B2T_BASE_PATH must be set")
    if not args.brain2text_dir:
        raise RuntimeError("--brain2text-dir or B2T_BRAIN2TEXT_DIR must be set")

    data2_b2t = f"{args.brain2text_dir}/brain2text"
    dataset_dir = f"{data2_b2t}/{args.dataset}"

    # b2t_24 transformer uses log-transformed data; everything else uses raw
    if args.model_type == "transformer" and args.dataset == "b2t_24":
        manifest_path = f"{dataset_dir}/trial_level_data_log/manifest.json"
    else:
        manifest_path = f"{dataset_dir}/trial_level_data/manifest.json"

    load_model_folder = f"{dataset_dir}/outputs/{args.model_name}"
    save_path = f"{dataset_dir}/logits/{args.model_name}"
    os.makedirs(save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating logits: {args.model_name}")
    print(f"Partition: {args.partition} | Device: {args.device}")
    print(f"{'='*60}\n")

    if args.model_type == "transformer":
        eval_cfg = EVAL_CONFIGS[args.dataset]
        model, config = load_transformer_model(
            load_model_folder,
            args.device,
            modelWeightsFile=WEIGHTS_FILE,
            eval_chunk_config=eval_cfg,
        )
    else:
        eval_cfg = None
        model, config = load_gru_model(
            load_model_folder,
            args.device,
            modelWeightsFile=WEIGHTS_FILE,
        )

    per = generate_and_save_logits(
        model=model,
        config=config,
        partition=args.partition,
        device=args.device,
        manifest_paths=[manifest_path],
        save_path=save_path,
        chunk_config=eval_cfg,
    )

    if per is not None:
        print(f"PER ({args.model_name}): {per:.6f}")


if __name__ == "__main__":
    main()
