#!/usr/bin/env python3
"""
Batch decoder runner - runs run_decoder.py across multiple models/seeds.

Example usage:
    # Run transformer (default) on val set
    python scripts/run_batch_decode.py

    # Run with custom hyperparameters
    python scripts/run_batch_decode.py --acoustic-scale 0.5 --llm-weight 1.5 --beam-size 1200

    # Run pretrained RNN on val and test
    python scripts/run_batch_decode.py --preset rnn --mode both

    # Run test only without LLM
    python scripts/run_batch_decode.py --mode test --disable-llm --no-wandb

    # Run custom seeds (val and test)
    python scripts/run_batch_decode.py \
        --base-path /data2/brain2text/b2t_25/logits/baseline_rnn_ucd_npl_seed_ \
        --seeds 1 2 3 4 5 6 7 8 9 \
        --mode both \
        --acoustic-scale 0.4 --temperature 1.0
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Preset configurations
PRESETS = {
    "transformer": {
        "val": "/data2/brain2text/b2t_25/logits/best_chunked_transformer_combined_seed_0/logits_val_chunk:5_context:20.npz",
        "test": "/data2/brain2text/b2t_25/logits/best_chunked_transformer_combined_seed_0/logits_test_chunk:5_context:20.npz",
        "encoder_name": "transformer_chunk5_ctx20",
    },
    "rnn": {
        "val": "/data2/brain2text/b2t_25/logits/pretrained_RNN/logits_val.npz",
        "test": "/data2/brain2text/b2t_25/logits/pretrained_RNN/logits_test.npz",
        "encoder_name": "rnn_pretrained",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run decoder across multiple seeds/models")

    # =======================================================ffffvfff==================
    # BATCH MODE SETTINGS
    # =========================================================================
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default="transformer",
                        help="Use a preset model configuration (default: transformer)")
    parser.add_argument("--mode", type=str, choices=["val", "test", "both"], default="val",
                        help="Which split(s) to run: val, test, or both (default: val)")

    # Custom seed mode (overrides preset)
    parser.add_argument("--base-path", type=str, default=None,
                        help="Base path prefix for custom runs (seed number will be appended)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="List of seeds to run (e.g., --seeds 1 2 3 4 5)")
    parser.add_argument("--logits-val-filename", type=str, default="logits_val.npz",
                        help="Val logits filename within each seed directory (default: logits_val.npz)")
    parser.add_argument("--logits-test-filename", type=str, default="logits_test.npz",
                        help="Test logits filename within each seed directory (default: logits_test.npz)")
    parser.add_argument("--encoder-name-prefix", type=str, default="seed",
                        help="Prefix for encoder model name in output files (default: seed)")

    # =========================================================================
    # TRIAL SELECTION
    # =========================================================================
    parser.add_argument("--random", type=int, default=None,
                        help="Randomly select N trials (fixed seed=42)")
    parser.add_argument("--trial-indices", type=int, nargs="*", default=None,
                        help="List of trial indices to decode")
    parser.add_argument("--start-trial-idx", type=int, default=None,
                        help="Start index (inclusive)")
    parser.add_argument("--end-trial-idx", type=int, default=None,
                        help="End index (exclusive)")

    # =========================================================================
    # LLM SETTINGS (defaults in run_decoder.py)
    # =========================================================================
    parser.add_argument("--model", default=None,
                        help="HuggingFace model ID")
    parser.add_argument("--disable-llm", action="store_true",
                        help="Disable LLM shallow fusion")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization")
    parser.add_argument("--lora-adapter", type=str, default=None,
                        help="LoRA adapter path (auto-selected if not specified)")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Use base model without LoRA adapter")
    parser.add_argument("--lm-rescore-interval", type=int, default=None,
                        help="LLM rescoring interval in frames")
    parser.add_argument("--scoring-chunk-size", type=int, default=None,
                        help="Batch size for LLM scoring")
    parser.add_argument("--length-normalize", action="store_true",
                        help="Apply length normalization at EOS scoring")

    # =========================================================================
    # KEY HYPERPARAMETERS (defaults in run_decoder.py)
    # =========================================================================
    parser.add_argument("--llm-weight", type=float, default=None,
                        help="LLM score weight for rescoring")
    parser.add_argument("--ngram-rescore-weight", type=float, default=None,
                        help="N-gram score weight during LLM rescoring")
    parser.add_argument("--alpha-ngram", type=float, default=None,
                        help="N-gram LM weight during beam search")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for scaling logits")
    parser.add_argument("--acoustic-scale", type=float, default=None,
                        help="Scale factor for acoustic log-probs")

    # =========================================================================
    # BEAM SEARCH SETTINGS (defaults in run_decoder.py)
    # =========================================================================
    parser.add_argument("--beam-size", type=int, default=None,
                        help="CTC beam size")
    parser.add_argument("--num-homophone-beams", type=int, default=None,
                        help="Homophone interpretations per beam")
    parser.add_argument("--beam-prune-threshold", type=float, default=None,
                        help="Beam pruning threshold")
    parser.add_argument("--homophone-prune-threshold", type=float, default=None,
                        help="Homophone pruning threshold")
    parser.add_argument("--beam-beta", type=float, default=None,
                        help="Extension bonus")
    parser.add_argument("--word-boundary-bonus", type=float, default=None,
                        help="Word boundary token bonus")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top beams to display")
    parser.add_argument("--score-combination", type=str, default=None,
                        choices=["max", "logsumexp"],
                        help="Score combination method")

    # =========================================================================
    # OUTPUT SETTINGS
    # =========================================================================
    parser.add_argument("--results-filename", type=str,
                        default=datetime.now().strftime("%m_%d_%H%M"),
                        help="Results filename prefix")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose per-beam output")

    # =========================================================================
    # PATHS
    # =========================================================================
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device (default: cuda:0)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token")

    return parser.parse_args()


def build_decoder_args(args, logits_val_path, logits_test_path, encoder_name):
    """Build list of arguments to pass to run_decoder.py."""
    cmd = ["--encoder-model-name", encoder_name]

    if logits_val_path:
        cmd.extend(["--logits-val-path", str(logits_val_path)])
    if logits_test_path:
        cmd.extend(["--logits-test-path", str(logits_test_path)])

    # Trial selection
    if args.random is not None:
        cmd.extend(["--random", str(args.random)])
    if args.trial_indices is not None:
        cmd.extend(["--trial-indices"] + [str(i) for i in args.trial_indices])
    if args.start_trial_idx is not None:
        cmd.extend(["--start-trial-idx", str(args.start_trial_idx)])
    if args.end_trial_idx is not None:
        cmd.extend(["--end-trial-idx", str(args.end_trial_idx)])

    # LLM settings (only pass if explicitly set)
    if args.model is not None:
        cmd.extend(["--model", args.model])
    if args.disable_llm:
        cmd.append("--disable-llm")
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")
    if args.lora_adapter is not None:
        cmd.extend(["--lora-adapter", args.lora_adapter])
    if args.no_adapter:
        cmd.append("--no-adapter")
    if args.lm_rescore_interval is not None:
        cmd.extend(["--lm-rescore-interval", str(args.lm_rescore_interval)])
    if args.scoring_chunk_size is not None:
        cmd.extend(["--scoring-chunk-size", str(args.scoring_chunk_size)])
    if args.length_normalize:
        cmd.append("--length-normalize")

    # Key hyperparameters (only pass if explicitly set)
    if args.llm_weight is not None:
        cmd.extend(["--llm-weight", str(args.llm_weight)])
    if args.ngram_rescore_weight is not None:
        cmd.extend(["--ngram-rescore-weight", str(args.ngram_rescore_weight)])
    if args.alpha_ngram is not None:
        cmd.extend(["--alpha-ngram", str(args.alpha_ngram)])
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.acoustic_scale is not None:
        cmd.extend(["--acoustic-scale", str(args.acoustic_scale)])

    # Beam search settings (only pass if explicitly set)
    if args.beam_size is not None:
        cmd.extend(["--beam-size", str(args.beam_size)])
    if args.num_homophone_beams is not None:
        cmd.extend(["--num-homophone-beams", str(args.num_homophone_beams)])
    if args.beam_prune_threshold is not None:
        cmd.extend(["--beam-prune-threshold", str(args.beam_prune_threshold)])
    if args.homophone_prune_threshold is not None:
        cmd.extend(["--homophone-prune-threshold", str(args.homophone_prune_threshold)])
    if args.beam_beta is not None:
        cmd.extend(["--beam-beta", str(args.beam_beta)])
    if args.word_boundary_bonus is not None:
        cmd.extend(["--word-boundary-bonus", str(args.word_boundary_bonus)])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.score_combination is not None:
        cmd.extend(["--score-combination", args.score_combination])

    # Output settings
    cmd.extend(["--results-filename", args.results_filename])
    if args.no_wandb:
        cmd.append("--no-wandb")
    if args.verbose:
        cmd.append("--verbose")

    # Paths
    cmd.extend(["--device", args.device])
    if args.hf_token:
        cmd.extend(["--hf-token", args.hf_token])

    return cmd


def run_decoder(decoder_script, args, logits_val_path, logits_test_path, encoder_name):
    """Run the decoder script with given paths."""
    cmd = [sys.executable, str(decoder_script)]
    cmd.extend(build_decoder_args(args, logits_val_path, logits_test_path, encoder_name))

    print(f"\n{'='*60}")
    print(f"Running: {encoder_name}")
    if logits_val_path:
        print(f"Val: {logits_val_path}")
    if logits_test_path:
        print(f"Test: {logits_test_path}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    decoder_script = script_dir / "run_decoder.py"

    if not decoder_script.exists():
        print(f"Error: {decoder_script} not found")
        sys.exit(1)

    # Custom seed mode
    if args.base_path and args.seeds:
        for seed in args.seeds:
            seed_dir = Path(f"{args.base_path}{seed}")
            encoder_name = f"{args.encoder_name_prefix}_{seed}"

            # Determine which logits files to use based on mode
            if args.mode == "val":
                logits_val = seed_dir / args.logits_val_filename
                logits_test = None
            elif args.mode == "test":
                logits_val = None
                logits_test = seed_dir / args.logits_test_filename
            else:  # both
                logits_val = seed_dir / args.logits_val_filename
                logits_test = seed_dir / args.logits_test_filename

            returncode = run_decoder(
                decoder_script, args,
                logits_val_path=logits_val,
                logits_test_path=logits_test,
                encoder_name=encoder_name,
            )

            if returncode != 0:
                print(f"Warning: Seed {seed} failed with return code {returncode}")

    # Preset mode
    else:
        preset = PRESETS[args.preset]

        if args.mode == "val":
            logits_val = preset["val"]
            logits_test = None
        elif args.mode == "test":
            logits_val = None
            logits_test = preset["test"]
        else:  # both
            logits_val = preset["val"]
            logits_test = preset["test"]

        run_decoder(
            decoder_script, args,
            logits_val_path=logits_val,
            logits_test_path=logits_test,
            encoder_name=preset["encoder_name"],
        )


if __name__ == "__main__":
    main()
