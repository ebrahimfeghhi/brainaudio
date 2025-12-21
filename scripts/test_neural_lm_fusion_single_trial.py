

import argparse
import gc
import time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from datetime import datetime

# Assuming this is available as requested
from brainaudio.inference.eval_metrics import _cer_and_wer 

from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer,
    VectorizedLexiconConstraint,
    HuggingFaceLMFusion,
)
from brainaudio.inference.decoder.beam_helpers import (
    decode_beam_texts,
    load_log_probs,
    load_phoneme_to_word_mapping,
    load_token_to_phoneme_mapping,
    pick_device,
)

"""
This run tests neural LM fusion with CTC batched beam search. Adjust this string to describe experiment details, settings, or observations for this run.
"""
"""Tiny end-to-end CTC beam-search check with neural LM fusion.

Loops through a range of trials, decodes them, tracks performance,
and computes final WER/CER metrics.
"""
DEFAULT_ENCODER_MODEL_NAME = "pretrained_RNN"
DEFAULT_TOKENS = "/data2/brain2text/lm/units_pytorch.txt"
DEFAULT_LEXICON = "/data2/brain2text/lm/lexicon_phonemes.txt"
TRANSCRIPTS_PKL = Path("/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CTC beam search + HF LM fusion loop")

    parser.add_argument("--encoder-model-name", type=str, default=DEFAULT_ENCODER_MODEL_NAME,
                        help="Name of the encoder model (used for logits path and wandb logging)")
    parser.add_argument("--start-trial-idx", type=int, default=0
                        , help="Start index (inclusive)")
    parser.add_argument("--end-trial-idx", type=int, default=None, help="End index (exclusive); defaults to all trials in logits file")
    parser.add_argument("--beam-size", type=int, default=100, help="CTC beam size")
    parser.add_argument("--model", default="google/gemma-3-270m", help="HF causal LM checkpoint")
    parser.add_argument("--hf-token", default=None, help="Optional HF token")
    parser.add_argument("--lm-weight", type=float, default=1, help="Fusion weight")
    parser.add_argument("--word-insertion-bonus", type=float, default=1, help="Bonus at boundaries")
    parser.add_argument("--max-context-length", type=int, default=512, help="Token budget")
    parser.add_argument("--device", default="cuda:1", help="Torch device")
    parser.add_argument("--logits", type=Path, default=None, help="NPZ logits file (default: derived from encoder-model-name)")
    parser.add_argument("--tokens", type=Path, default=Path(DEFAULT_TOKENS), help="units file")
    parser.add_argument("--lexicon", type=Path, default=Path(DEFAULT_LEXICON), help="lexicon file")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top beams to display per trial")
    parser.add_argument("--num-homophone-beams", type=int, default=2, help="Number of text interpretations (homophones) to track per beam")
    parser.add_argument("--beam-prune-threshold", type=float, default=20, help="Prune beams that are more than this many log-prob points below the best.")
    parser.add_argument("--homophone-prune-threshold", type=float, default=10.0, help="Prune homophones more than this many log-prob points below the best.")
    parser.add_argument(
        "--results-filename",
        type=str,
        default=datetime.now().strftime("%m/%d_%H%M"),
        help="Filename for saving results (will be placed in /home/ebrahim/brainaudio/results directory)"
    )
    
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B logging (default: enabled)"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization (requires bitsandbytes)"
    )

    return parser.parse_args()



notes = "Running with simplified apply_lm_fusion_post_selection which only allows for one homophone per beam."

def main():
    args = parse_args()

    # Set default logits path based on encoder_model_name if not provided
    if args.logits is None:
        args.logits = Path(f"/data2/brain2text/b2t_25/logits/{args.encoder_model_name}/logits_val.npz")

    # Initialize wandb run (before any wandb.log or Table calls)
    if not args.disable_wandb:
        import wandb
        # REMOVE: wandb.online() 
        
        wandb.init(
            project="brainaudio-neural-lm-fusion",
            config=vars(args),
            name=f"{args.results_filename}",
            mode="online",    # <--- Force online mode here
            notes=notes       # <--- Pass notes here so they save as run metadata
        )
        
        wandb.log({"notes": notes})
            
    else:
        print("[wandb] W&B logging disabled")

    device = pick_device(args.device)
    
    K = args.top_k

    # Determine number of trials in logits file if end-trial-idx is None
    if args.end_trial_idx is None:
        logits_npz = np.load(args.logits)
        trial_keys = [k for k in logits_npz.keys() if k.startswith("arr_")]
        trial_indices = [int(k.split("_")[1]) for k in trial_keys]
        if not trial_indices:
            raise ValueError(f"No trial arrays found in {args.logits}")
        args.end_trial_idx = max(trial_indices) + 1
        print(f"Auto-set end-trial-idx to {args.end_trial_idx} (all trials in logits file)")

    print(f"Loading resources on {device}...")
    
    lexicon = VectorizedLexiconConstraint.from_file_paths(
        tokens_file=str(args.tokens),
        lexicon_file=str(args.lexicon),
        device=device,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)

    if args.load_in_4bit:
        # 4-bit quantization: device must be set via device_map at load time
        # Use bfloat16 for compute dtype - float16 can cause NaN with some models
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},  # place entire model on specified device
            token=args.hf_token,
        )
        print(f"[INFO] Loaded {args.model} in 4-bit quantization on {device}")
    else:
        # Standard loading: load then move to device
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            token=args.hf_token,
        ).to(device)
        print(f"[INFO] Loaded {args.model} (full precision) on {device}")

    lm_fusion = HuggingFaceLMFusion(
        model=model,
        tokenizer=tokenizer,
        weight=args.lm_weight,
        homophone_aggregation="max",
        device=device,
        max_context_length=args.max_context_length,
        word_insertion_bonus=args.word_insertion_bonus
    )

    decoder = BatchedBeamCTCComputer(
        blank_index=lexicon.blank_index,
        beam_size=args.beam_size,
        lexicon=lexicon,
        lm_fusion=lm_fusion,
        allow_cuda_graphs=False,
        num_homophone_beams=args.num_homophone_beams,
        beam_threshold=args.beam_prune_threshold,
        homophone_prune_threshold=args.homophone_prune_threshold,
    )

    transcripts = None
    if TRANSCRIPTS_PKL.exists():
        transcripts = pd.read_pickle(TRANSCRIPTS_PKL)
    else:
        print(f"Warning: Transcripts file not found at {TRANSCRIPTS_PKL}")

    # Results Containers
    decoded_sentences = []
    ground_truth_sentences = []
    
    total_start_time = time.perf_counter()

    print(f"\n=== Starting Decode Loop: Trials {args.start_trial_idx} to {args.end_trial_idx} ===")
    print(f"[INFO] Decoding from start-trial-idx={args.start_trial_idx} to end-trial-idx={args.end_trial_idx}")
    
    for trial_idx in range(args.start_trial_idx, args.end_trial_idx):
        # Load single trial
        log_probs, _, lengths = load_log_probs(args.logits, [trial_idx], device)

        # Run Decoder
        if device.type == "cuda":
            torch.cuda.synchronize()
        trial_start = time.perf_counter()

        result = decoder(log_probs, lengths)

        if device.type == "cuda":
            torch.cuda.synchronize()
        trial_elapsed = time.perf_counter() - trial_start

        # Select top K beams that do not have -inf score
        scores = result.scores[0].cpu().numpy()
        valid_indices = [i for i, s in enumerate(scores) if s != -float('inf')]
        # Sort valid indices by descending score
        valid_indices_sorted = sorted(valid_indices, key=lambda i: scores[i], reverse=True)
        topk_indices = valid_indices_sorted[:K]
        # context_texts[batch][beam] is now List[Tuple[float, str]], get best text (index 0)
        decoded_beams = [result.context_texts[0][i][0][1] for i in topk_indices]
        best_text = decoded_beams[0] if decoded_beams else ""
        decoded_sentences.append(best_text)

        # Handle Ground Truth
        ground_truth = ""
        if isinstance(transcripts, (list, tuple)) and trial_idx < len(transcripts):
            ground_truth = transcripts[trial_idx]
        elif hasattr(transcripts, 'get'):
            ground_truth = transcripts.get(trial_idx, "")
        elif hasattr(transcripts, 'iloc'):
            ground_truth = transcripts.iloc[trial_idx]
        ground_truth_sentences.append(ground_truth)

        # Print Status
        best_score = result.scores[0, 0].item()
        print(f"Trial {trial_idx:3d} | {trial_elapsed*1000:.1f}ms | Score: {best_score:.2f}")
        print(f"  GT:   {ground_truth}")
        print(f"  Best: {best_text}")
        print(f"  Top {K} beams (with {args.num_homophone_beams} homophone interpretations each):")
        for rank, i in enumerate(topk_indices):
            beam_score = result.scores[0, i].item()
            all_texts = result.context_texts[0][i]  # List of (lm_score, text) tuples
            print(f"  #{rank:02d} | beam_score={beam_score:.4f}")
            for k_idx, (lm_score, text) in enumerate(all_texts):
                print(f"       H{k_idx}: lm={lm_score:.4f} | {text}")
        print("-" * 60)

        # Explicit memory cleanup to prevent accumulation across trials
        del result
        del log_probs
        gc.collect()
        torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - total_start_time

    print("\n=== Final Results Summary ===")
    print(f"Processed {len(decoded_sentences)} trials in {total_elapsed:.2f}s")
    print(f"Beam size: {args.beam_size}, Homophone beams: {args.num_homophone_beams}, Prune threshold: {args.homophone_prune_threshold}")

# Compute Metrics
    try:
        # Passing lists of strings as expected by standard WER calculators
        _, wer, _ = _cer_and_wer(decoded_sentences, ground_truth_sentences)
        print(f"\nAggregate WER: {wer:.4f}")
    except Exception as e:
        print(f"\nError computing WER: {e}")

    # Save results to CSV file
    results_dir = Path("/home/ebrahim/brainaudio/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create filename: {encoder_model_name}_{results_filename}.csv
    # Replace "/" with "_" in results_filename to make it a valid filename
    safe_results_filename = args.results_filename.replace("/", "_")
    csv_filename = f"{args.encoder_model_name}_{safe_results_filename}.csv"
    csv_path = results_dir / csv_filename

    # Create DataFrame with id and text columns (same format as RNN baseline)
    results_df = pd.DataFrame({
        "id": list(range(args.start_trial_idx, args.start_trial_idx + len(decoded_sentences))),
        "text": decoded_sentences
    })
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to {csv_path}")

    # --- wandb logging ---
    if not args.disable_wandb:
        try:
            import wandb
            # Log WER to wandb
            if 'wer' in locals() and wer is not None:
                wandb.log({"WER": wer})
            # Log predicted and ground truth sentences as a table to wandb
            if decoded_sentences and ground_truth_sentences:
                data = list(zip(range(len(decoded_sentences)), ground_truth_sentences, decoded_sentences))
                table = wandb.Table(data=data, columns=["idx", "ground_truth", "predicted"])
                wandb.log({"predictions_vs_ground_truth": table})
            wandb.finish()
        except Exception as e:
            print(f"[wandb] Logging failed: {e}")
    else:
        print("[wandb] Skipping W&B logging (disabled)")

if __name__ == "__main__":
    main()