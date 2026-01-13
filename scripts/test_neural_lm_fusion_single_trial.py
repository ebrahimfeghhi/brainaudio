import argparse
import time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from datetime import datetime
import sys

from brainaudio.inference.eval_metrics import _cer_and_wer

from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer,
    VectorizedLexiconConstraint,
    HuggingFaceLMFusion,
)
from brainaudio.inference.decoder.beam_helpers import (
    load_log_probs,
    pick_device
)
from brainaudio.inference.decoder.word_ngram_lm_optimized_v2 import FastNGramLM, WordHistory

"""
CTC beam search with word-level N-gram LM fusion and optional LLM shallow fusion.
"""

# Default paths (phoneme mode)
DEFAULT_ENCODER_MODEL_NAME = "pretrained_RNN"
DEFAULT_TOKENS = "/data2/brain2text/lm/units_pytorch.txt"
DEFAULT_LEXICON = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt"
DEFAULT_WORD_LM_PATH = "/data2/brain2text/lm/lm_dec19_huge_4gram.kenlm"
TRANSCRIPTS_PKL = Path("/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CTC beam search + word N-gram LM + optional LLM fusion")

    parser.add_argument("--encoder-model-name", type=str, default=DEFAULT_ENCODER_MODEL_NAME,
                        help="Name of the encoder model (used for logits path and wandb logging)")
    parser.add_argument("--trial-indices", type=int, nargs="*", default=None,
                        help="List of trial indices to decode (e.g., --trial-indices 0 5 10). Default: all trials in logits file")
    parser.add_argument("--start-trial-idx", type=int, default=None,
                        help="Start index (inclusive). Use with --end-trial-idx for a range.")
    parser.add_argument("--end-trial-idx", type=int, default=None,
                        help="End index (exclusive). Use with --start-trial-idx for a range.")
    parser.add_argument("--beam-size", type=int, default=300, help="CTC beam size")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M", help="HF causal LM checkpoint")
    parser.add_argument("--hf-token", default=None, help="Optional HF token")
    parser.add_argument("--lm-weight", type=float, default=1, help="Neural LM fusion weight")
    parser.add_argument("--word-insertion-bonus", type=float, default=1.5, help="Bonus at boundaries")
    parser.add_argument("--max-context-length", type=int, default=50, help="Token budget")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--logits", type=Path, default=None, help="NPZ logits file (default: derived from encoder-model-name)")
    parser.add_argument("--tokens", type=Path, default=Path(DEFAULT_TOKENS), help="units file")
    parser.add_argument("--lexicon", type=Path, default=Path(DEFAULT_LEXICON), help="lexicon file")
    parser.add_argument("--top-k", type=int, default=1, help="Number of top beams to display per trial")
    parser.add_argument("--num-homophone-beams", type=int, default=3, help="Number of text interpretations (homophones) to track per beam")
    parser.add_argument("--beam-prune-threshold", type=float, default=12, help="Prune beams that are more than this many log-prob points below the best.")
    parser.add_argument("--homophone-prune-threshold", type=float, default=4, help="Prune homophones more than this many log-prob points below the best.")
    parser.add_argument("--beam-beta", type=float, default=np.log(7), help="Bonus added to extending beams (not blank/repeat).")
    parser.add_argument("--beam-blank-penalty", type=float, default=0, help="Penalty subtracted from blank emissions.")
    parser.add_argument("--logit-scale", type=float, default=0.6, help="Scalar multiplier for encoder logits.")
    parser.add_argument(
        "--results-filename",
        type=str,
        default=datetime.now().strftime("%m_%d_%H%M"),
        help="Filename for saving results (will be placed in /home/ebrahim/brainaudio/results directory)"
    )
    parser.add_argument("--enable-wandb", action="store_true", help="Enable W&B logging (default: disabled)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization (requires bitsandbytes)")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM shallow fusion (useful for testing n-gram LM alone)")
    parser.add_argument("--test-mode", action="store_true", help="Use logits_test.npz and skip WER computation")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbose output (skip per-beam printing)")

    # Word N-gram LM arguments
    parser.add_argument("--word-lm-path", type=str, default=DEFAULT_WORD_LM_PATH,
                        help="Path to word-level KenLM file (.kenlm)")
    parser.add_argument("--alpha-ngram", type=float, default=2,
                        help="Weight for word N-gram LM scores (default: 0.5)")
    parser.add_argument("--beta-ngram", type=float, default=1.0,
                        help="Word insertion bonus for word N-gram LM (default: 1.0)")

    return parser.parse_args()


notes = "Word-level N-gram LM fusion with optional LLM shallow fusion."


def main():
    args = parse_args()

    print(f"[INFO] Tokens: {args.tokens}")
    print(f"[INFO] Lexicon: {args.lexicon}")
    print(f"[INFO] Word LM: {args.word_lm_path}")
    print(f"[INFO] Alpha (ngram weight): {args.alpha_ngram}")
    print(f"[INFO] Beta (word insertion bonus): {args.beta_ngram}")

    # Set default logits path based on encoder_model_name if not provided
    if args.logits is None:
        logits_filename = "logits_test.npz" if args.test_mode else "logits_val.npz"
        args.logits = Path(f"/data2/brain2text/b2t_25/logits/{args.encoder_model_name}/{logits_filename}")

    print(f"[INFO] Logits: {args.logits}")

    # Initialize wandb run (before any wandb.log or Table calls)
    if args.enable_wandb:
        import wandb
        wandb.init(
            project="brainaudio-neural-lm-fusion",
            config=vars(args),
            name=f"{args.results_filename}",
            mode="online",
            notes=notes
        )
        wandb.log({"notes": notes})
    else:
        print("[wandb] W&B logging disabled")

    device = pick_device(args.device)
    K = args.top_k

    # Determine trial indices to decode
    if args.trial_indices is not None:
        print(f"Using {len(args.trial_indices)} explicitly provided trial indices")
    elif args.start_trial_idx is not None or args.end_trial_idx is not None:
        logits_npz = np.load(args.logits)
        trial_keys = [k for k in logits_npz.keys() if k.startswith("arr_")]
        all_indices = sorted([int(k.split("_")[1]) for k in trial_keys])
        start = args.start_trial_idx if args.start_trial_idx is not None else 0
        end = args.end_trial_idx if args.end_trial_idx is not None else max(all_indices) + 1
        args.trial_indices = list(range(start, end))
        print(f"Using trial range [{start}, {end}) = {len(args.trial_indices)} trials")
    else:
        logits_npz = np.load(args.logits)
        trial_keys = [k for k in logits_npz.keys() if k.startswith("arr_")]
        args.trial_indices = sorted([int(k.split("_")[1]) for k in trial_keys])
        if not args.trial_indices:
            raise ValueError(f"No trial arrays found in {args.logits}")
        print(f"Auto-detected {len(args.trial_indices)} trials in logits file")

    print(f"Loading resources on {device}...")

    lexicon = VectorizedLexiconConstraint.from_file_paths(
        tokens_file=str(args.tokens),
        lexicon_file=str(args.lexicon),
        device=device,
    )
    

    # Load LLM for shallow fusion (skip if --disable-llm)
    lm_fusion = None
    if not args.disable_llm:
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token, trust_remote_code=True)

        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=quantization_config,
                dtype=torch.bfloat16,
                device_map={"": device},
                token=args.hf_token,
            )
            print(f"[INFO] Loaded {args.model} in 4-bit quantization on {device}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=torch.bfloat16,
                token=args.hf_token,
                trust_remote_code=True
            ).to(device)
            print(f"[INFO] Loaded {args.model} on {device}")

        lm_fusion = HuggingFaceLMFusion(
            model=model,
            tokenizer=tokenizer,
            weight=args.lm_weight,
            homophone_aggregation="max",
            device=device,
            max_context_length=args.max_context_length,
            word_insertion_bonus=args.word_insertion_bonus,
        )
    else:
        print("[INFO] LLM shallow fusion disabled (--disable-llm)")

    # Initialize word-level N-gram LM
    word_ngram_lm = FastNGramLM(args.word_lm_path, alpha=args.alpha_ngram, beta=args.beta_ngram)
    word_history = WordHistory()
 
    print(f"[INFO] Word N-gram LM loaded with alpha={args.alpha_ngram}, beta={args.beta_ngram}")

    decoder = BatchedBeamCTCComputer(
        blank_index=lexicon.blank_index,
        beam_size=args.beam_size,
        lexicon=lexicon,
        lm_fusion=lm_fusion,
        allow_cuda_graphs=False,
        num_homophone_beams=args.num_homophone_beams,
        beam_threshold=args.beam_prune_threshold,
        homophone_prune_threshold=args.homophone_prune_threshold,
        beam_beta=args.beam_beta,
        beam_blank_penalty=args.beam_blank_penalty,
        word_ngram_lm=word_ngram_lm,
        word_history = word_history
    )

    transcripts = None
    if not args.test_mode:
        if TRANSCRIPTS_PKL.exists():
            transcripts = pd.read_pickle(TRANSCRIPTS_PKL)
        else:
            print(f"Warning: Transcripts file not found at {TRANSCRIPTS_PKL}")

    # Results Containers
    decoded_sentences = []
    ground_truth_sentences = []
    gt_in_beams_count = 0

    total_start_time = time.perf_counter()

    print(f"\n=== Starting Decode Loop: {len(args.trial_indices)} trials ===")
    print(f"[INFO] Decoding trial indices: {args.trial_indices}")

    # Pre-load NPZ file once (avoid opening file per trial)
    logits_npz = np.load(args.logits)

    # Move import outside loop
    from brainaudio.inference.decoder import materialize_beam_transcript, collapse_ctc_sequence

    for trial_idx in args.trial_indices:
        # Load single trial from pre-opened NPZ (much faster than load_log_probs per trial)
        logits = torch.from_numpy(logits_npz[f"arr_{trial_idx}"]).to(device).unsqueeze(0)
        lengths = torch.tensor([logits.shape[1]], device=device)
        # Convert raw logits to log probabilities, then apply acoustic scale
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1) * args.logit_scale

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

        # Get decoded text from context_texts (populated by WordNGramLM)
        #decoded_beams = [result.context_texts[0][i][0][1] for i in topk_indices]
        #best_text = decoded_beams[0] if decoded_beams else ""
        decoded_beams = []
        for i in topk_indices:
            context_list = result.context_texts[0][i]
            if context_list:
                # Tuple is (score, lm_id, history_id) -> We want index 2
                hist_id = context_list[0][2]
                text = word_history.get_text(hist_id)
                decoded_beams.append(text)
            else:
                decoded_beams.append("")

        best_text = decoded_beams[0] if decoded_beams else ""
        decoded_sentences.append(best_text)

        # Handle Ground Truth (skip in test mode)
        ground_truth = ""
        if not args.test_mode:
            if isinstance(transcripts, (list, tuple)) and trial_idx < len(transcripts):
                ground_truth = transcripts[trial_idx]
            elif hasattr(transcripts, 'get'):
                ground_truth = transcripts.get(trial_idx, "")
            elif hasattr(transcripts, 'iloc'):
                ground_truth = transcripts.iloc[trial_idx]
            ground_truth_sentences.append(ground_truth)

        # Print Status
        best_score = scores[topk_indices[0]] if topk_indices else 0
        print(f"Trial {trial_idx:3d} | {trial_elapsed*1000:.1f}ms | Score: {best_score:.2f}")
        if not args.test_mode:
            print(f"  GT:   {ground_truth}")
        print(f"  Best: {best_text}")

        # Verbose per-beam printing (skip with --quiet)
        if not args.quiet:
            print(f"  Top {K} beams (with {args.num_homophone_beams} homophone interpretations each):")
            for rank, i in enumerate(topk_indices):
                beam_score = result.scores[0, i].item()

                # Get token sequence for this beam
                seq_raw = materialize_beam_transcript(result, 0, i)
                seq_collapsed = collapse_ctc_sequence(seq_raw.tolist(), lexicon.blank_index)
                token_names_collapsed = [lexicon.token_to_symbol.get(idx, f"?{idx}") for idx in seq_collapsed]

                # Raw sequence (before CTC merging) - shows blanks and repeats
                token_names_raw = [lexicon.token_to_symbol.get(idx, f"?{idx}") for idx in seq_raw.tolist()]

                print(f"  #{rank:02d} | score={beam_score:.2f} | {' '.join(token_names_collapsed)}")
                print(f"       Raw ({len(seq_raw)} frames): {' '.join(token_names_raw[-30:])}")

                # Print homophone interpretations
                all_texts = result.context_texts[0][i]
                for k_idx, (lm_score, _, hist_id) in enumerate(all_texts):
                    text_str = word_history.get_text(hist_id)
                    print(f"       H{k_idx}: lm={lm_score:.4f} | {text_str}")

            print("-" * 60)

        # Cleanup references (let Python GC handle it naturally)
        del result
        del log_probs

    total_elapsed = time.perf_counter() - total_start_time

    print("\n=== Final Results Summary ===")
    print(f"Processed {len(decoded_sentences)} trials in {total_elapsed:.2f}s")
    print(f"Beam size: {args.beam_size}, Homophone beams: {args.num_homophone_beams}, Prune threshold: {args.homophone_prune_threshold}")
    if not args.test_mode:
        print(f"Ground truth in beams: {gt_in_beams_count}/{len(decoded_sentences)} ({100*gt_in_beams_count/len(decoded_sentences):.1f}%)")
    if lm_fusion is not None:
        print(f"LLM forward passes: {lm_fusion.get_call_count()}")
    else:
        print("LLM: disabled")

    # Compute Metrics (skip in test mode)
    wer = None
    if not args.test_mode:
        try:
            _, wer, _ = _cer_and_wer(decoded_sentences, ground_truth_sentences)
            print(f"\nAggregate WER: {wer:.4f}")
        except Exception as e:
            print(f"\nError computing WER: {e}")
    else:
        print("\n[test-mode] Skipping WER computation")

    # Save results to CSV file
    results_dir = Path("/home/ebrahim/brainaudio/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    safe_results_filename = args.results_filename.replace("/", "_")
    csv_filename = f"{args.encoder_model_name}_{safe_results_filename}.csv"
    csv_path = results_dir / csv_filename

    results_df = pd.DataFrame({
        "id": args.trial_indices,
        "text": decoded_sentences
    })
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to {csv_path}")

    # --- wandb logging ---
    if args.enable_wandb:
        try:
            import wandb
            if 'wer' in locals() and wer is not None:
                wandb.log({"WER": wer})
            if decoded_sentences and ground_truth_sentences:
                data = list(zip(args.trial_indices, ground_truth_sentences, decoded_sentences))
                table = wandb.Table(data=data, columns=["trial_idx", "ground_truth", "predicted"])
                wandb.log({"predictions_vs_ground_truth": table})
            wandb.finish()
        except Exception as e:
            print(f"[wandb] Logging failed: {e}")
    else:
        print("[wandb] Skipping W&B logging (disabled)")


if __name__ == "__main__":
    main()
