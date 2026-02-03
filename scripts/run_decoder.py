import argparse
import os
import time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
from datetime import datetime
import psutil

from brainaudio.inference.eval_metrics import _cer_and_wer

from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer,
    VectorizedLexiconConstraint,
    LLMRescorer,
)
from brainaudio.inference.decoder.beam_helpers import pick_device
from brainaudio.inference.decoder.word_ngram_lm_optimized import FastNGramLM, WordHistory

"""
CTC beam search with word-level N-gram LM fusion and optional LLM shallow fusion.
"""

<<<<<<< HEAD
# Default paths (phoneme mode)
DEFAULT_ENCODER_MODEL_NAME = "best_chunked_transformer_combined_seed_0" #"pretrained_RNN"
=======
# Default paths
<<<<<<< Updated upstream
>>>>>>> 4112d88840496a933a5cfcf4eaac44a5100b5700
DEFAULT_TOKENS = "/data2/brain2text/lm/units_pytorch.txt"
DEFAULT_LEXICON = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme_with_variants.txt"
DEFAULT_WORD_LM_PATH = "/data2/brain2text/lm/lm_dec19_huge_4gram.kenlm"
TRANSCRIPTS_PKL = Path("/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl")
=======
BASE_PATH = "/home/ebrahim"
DEFAULT_TOKENS = f"{BASE_PATH}/data2/brain2text/lm/units_pytorch.txt"
DEFAULT_LEXICON = f"{BASE_PATH}/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme_with_variants.txt"
DEFAULT_WORD_LM_PATH = f"{BASE_PATH}/data2/brain2text/lm/lm_dec19_huge_4gram.kenlm"
TRANSCRIPTS_PKL = Path(f"{BASE_PATH}/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl")
>>>>>>> Stashed changes

# LoRA adapter paths (auto-selected based on model)
LORA_ADAPTER_1B = "/home/ebrahim/brainaudio/finetune_llm/llama-3.2-1b-hf-finetuned-normalized"
LORA_ADAPTER_3B = "/home/ebrahim/brainaudio/finetune_llm/llama-3.2-3b-hf-finetuned-normalized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CTC beam search + word N-gram LM + optional LLM fusion")

    # =========================================================================
    # COMMONLY USED ARGUMENTS
    # =========================================================================

    # Logits paths (required)
    parser.add_argument("--logits-val-path", type=Path, default=None,
                        help="Path to validation logits NPZ file")
    parser.add_argument("--logits-test-path", type=Path, default=None,
                        help="Path to test logits NPZ file (if provided, runs val then test)")

    # Trial selection
    parser.add_argument("--random", type=int, default=None,
                        help="Randomly select N trials (fixed seed=42 for reproducibility), e.g. --random 300")
    parser.add_argument("--trial-indices", type=int, nargs="*", default=None,
                        help="List of trial indices to decode (e.g., --trial-indices 0 5 10)")
    parser.add_argument("--start-trial-idx", type=int, default=None,
                        help="Start index (inclusive). Use with --end-trial-idx for a range.")
    parser.add_argument("--end-trial-idx", type=int, default=None,
                        help="End index (exclusive). Use with --start-trial-idx for a range.")

    # LLM model
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B)")
    parser.add_argument("--disable-llm", action="store_true",
                        help="Disable LLM shallow fusion (n-gram LM only)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization")

    # Key hyperparameters
    parser.add_argument("--llm-weight", type=float, default=1.2,
                        help="LLM score weight for rescoring")
    parser.add_argument("--ngram-rescore-weight", type=float, default=0.0,
                        help="N-gram score weight during LLM rescoring (interpolation)")
    parser.add_argument("--alpha-ngram", type=float, default=1.0,
                        help="N-gram LM weight (during beam search)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for scaling logits before softmax (lower = sharper)")
    parser.add_argument("--acoustic-scale", type=float, default=0.4,
                        help="Scale factor for acoustic log-probs after softmax")
 
    # Output
    parser.add_argument("--results-filename", type=str,
                        default=datetime.now().strftime("%m_%d_%H%M"),
                        help="Results filename (saved to /home/ebrahim/brainaudio/results/)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging (enabled by default)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Verbose per-beam output")


    # LLM settings
    parser.add_argument("--lora-adapter", type=str, default=None,
                        help="LoRA adapter path (auto-selected if not specified)")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Use base model without LoRA adapter")
    parser.add_argument("--lm-rescore-interval", type=int, default=15,
                        help="LLM rescoring interval in frames (0 = end only)")
    parser.add_argument("--scoring-chunk-size", type=int, default=256,
                        help="Batch size for LLM scoring")
    parser.add_argument("--length-normalize", action="store_true", default=False,
                        help="Apply length normalization (divide score by word count) at EOS scoring")

    # N-gram LM settings
    parser.add_argument("--word-lm-path", type=str, default=DEFAULT_WORD_LM_PATH,
                        help="Path to KenLM file")

    # Beam search settings
    parser.add_argument("--beam-size", type=int, default=900,
                        help="CTC beam size")
    parser.add_argument("--num-homophone-beams", type=int, default=3,
                        help="Homophone interpretations per beam")
    parser.add_argument("--beam-prune-threshold", type=float, default=18,
                        help="Beam pruning threshold (log-prob)")
    parser.add_argument("--homophone-prune-threshold", type=float, default=4,
                        help="Homophone pruning threshold (log-prob)")
    parser.add_argument("--beam-beta", type=float, default=1.5,
                        help="Extension bonus (non-blank/repeat)")
    parser.add_argument("--word-boundary-bonus", type=float, default=1,
                        help="Word boundary token bonus")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top beams to display")
    parser.add_argument("--score-combination", type=str, default="max",
                        choices=["max", "logsumexp"],
                        help="Method for combining scores of equivalent hypotheses: 'max' (faster) or 'logsumexp' (sums probabilities)")

    # Paths
    parser.add_argument("--encoder-model-name", type=str, default="model",
                        help="Encoder model name (used for output CSV naming)")
    parser.add_argument("--tokens", type=Path, default=Path(DEFAULT_TOKENS),
                        help="Units file")
    parser.add_argument("--lexicon", type=Path, default=Path(DEFAULT_LEXICON),
                        help="Lexicon file")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token")

    return parser.parse_args()


def run_decode_pass(
    args, decoder, device, word_history, lexicon, lm_fusion, process,
    logits_path, trial_indices, transcripts, is_test_mode, results_filename
):
    """Run a single decode pass (val or test) and save results."""
    from brainaudio.inference.decoder import materialize_beam_transcript, collapse_ctc_sequence
    from brainaudio.inference.eval_metrics import clean_string

    K = args.top_k

    # Results Containers
    decoded_sentences = []
    ground_truth_sentences = []
    rtf_values = []
    peak_cpu_memory = process.memory_info().rss

    total_start_time = time.perf_counter()

    mode_str = "TEST" if is_test_mode else "VAL"
    print(f"\n=== Starting {mode_str} Decode Loop: {len(trial_indices)} trials ===")
    print(f"[INFO] Decoding trial indices: {trial_indices}")
    print(f"[INFO] Logits path: {logits_path}")

    # Pre-load NPZ file once
    logits_npz = np.load(logits_path)

    for trial_idx in trial_indices:
        logits = torch.from_numpy(logits_npz[f"arr_{trial_idx}"]).to(device).unsqueeze(0)

        num_frames_original = logits.shape[1]
        trial_duration_seconds = num_frames_original * 0.08

        num_padding_frames = 2
        vocab_size = logits.shape[-1]
        padding = torch.zeros((1, num_padding_frames, vocab_size), device=device, dtype=logits.dtype)
        logits = torch.cat([logits, padding], dim=1)

        lengths = torch.tensor([logits.shape[1]], device=device)
        log_probs = torch.nn.functional.log_softmax(logits / args.temperature, dim=-1) * args.acoustic_scale

        if device.type == "cuda":
            torch.cuda.synchronize()
        trial_start = time.perf_counter()

        result = decoder(log_probs, lengths)

        if device.type == "cuda":
            torch.cuda.synchronize()
        trial_elapsed = time.perf_counter() - trial_start

        rtf = trial_elapsed / trial_duration_seconds
        rtf_values.append(rtf)

        scores = result.scores[0].cpu().numpy()
        valid_indices = [i for i, s in enumerate(scores) if s != -float('inf')]
        valid_indices_sorted = sorted(valid_indices, key=lambda i: scores[i], reverse=True)
        topk_indices = valid_indices_sorted[:K]

        decoded_beams = []
        lm_scores = []
        for i in topk_indices:
            context_list = result.context_texts[0][i]
            if context_list:
                lm_scores.append(context_list[0][0])
                hist_id = context_list[0][2]
                text = word_history.get_text(hist_id)
                decoded_beams.append(text)
            else:
                decoded_beams.append("")

        best_text = decoded_beams[0] if decoded_beams else ""
        best_lm_score = lm_scores[0] if lm_scores else 0
        decoded_sentences.append(best_text)

        ground_truth = ""
        if not is_test_mode:
            if isinstance(transcripts, (list, tuple)) and trial_idx < len(transcripts):
                ground_truth = transcripts[trial_idx]
            elif hasattr(transcripts, 'get'):
                ground_truth = transcripts.get(trial_idx, "")
            elif hasattr(transcripts, 'iloc'):
                ground_truth = transcripts.iloc[trial_idx]
            ground_truth_sentences.append(ground_truth)

        best_score = scores[topk_indices[0]] if topk_indices else 0
        print(f"Trial {trial_idx:3d} | {trial_elapsed*1000:.1f}ms | RTF: {rtf:.3f} | Score: {best_score:.2f} | LM Score: {best_lm_score:.2f}")
        if not is_test_mode:
            print(f"  GT:   {ground_truth}")
        print(f"  Best: {best_text}")

        if args.verbose:
            # Greedy decoding (phonemes)
            greedy_seq = log_probs[0].argmax(dim=-1).cpu().tolist()
            greedy_collapsed = collapse_ctc_sequence(greedy_seq, lexicon.blank_index)
            greedy_phonemes = [lexicon.token_to_symbol.get(idx, f"?{idx}") for idx in greedy_collapsed]
            print(f"  Greedy (phonemes): {' '.join(greedy_phonemes)}")

            print(f"  Top {K} beams (with {args.num_homophone_beams} homophone interpretations each):")
            for rank, i in enumerate(topk_indices):
                beam_score = result.scores[0, i].item()
                seq_raw = materialize_beam_transcript(result, 0, i)
                seq_collapsed = collapse_ctc_sequence(seq_raw.tolist(), lexicon.blank_index)
                token_names_collapsed = [lexicon.token_to_symbol.get(idx, f"?{idx}") for idx in seq_collapsed]
                token_names_raw = [lexicon.token_to_symbol.get(idx, f"?{idx}") for idx in seq_raw.tolist()]

                print(f"  #{rank:02d} | score={beam_score:.2f} | {' '.join(token_names_collapsed)}")
                print(f"       Raw ({len(seq_raw)} frames): {' '.join(token_names_raw[-30:])}")

                all_texts = result.context_texts[0][i]
                for k_idx, (lm_score, _, hist_id) in enumerate(all_texts):
                    text_str = word_history.get_text(hist_id)
                    print(f"       H{k_idx}: lm={lm_score:.4f} | {text_str}")
            print("-" * 60)

        del result
        del log_probs

        current_cpu_memory = process.memory_info().rss
        peak_cpu_memory = max(peak_cpu_memory, current_cpu_memory)

    total_elapsed = time.perf_counter() - total_start_time

    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_gb = peak_vram_bytes / (1024 ** 3)
    peak_cpu_memory_gb = peak_cpu_memory / (1024 ** 3)

    print(f"\n=== {mode_str} Results Summary ===")
    print(f"Processed {len(decoded_sentences)} trials in {total_elapsed:.2f}s")
    print(f"Beam size: {args.beam_size}, Homophone beams: {args.num_homophone_beams}, Prune threshold: {args.homophone_prune_threshold}")
    if lm_fusion is not None:
        print(f"LLM forward passes: {lm_fusion.llm_call_count}")
    else:
        print("LLM: disabled")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"Peak CPU Memory: {peak_cpu_memory_gb:.2f} GB")

    rtf_array = np.array(rtf_values)
    print(f"RTF - Mean: {rtf_array.mean():.3f}, Median: {np.median(rtf_array):.3f}, Min: {rtf_array.min():.3f}, Max: {rtf_array.max():.3f}")

    wer = None
    if not is_test_mode:
        try:
            _, wer, _ = _cer_and_wer(decoded_sentences, ground_truth_sentences)
            print(f"\nAggregate WER: {wer:.4f}")
        except Exception as e:
            print(f"\nError computing WER: {e}")
    else:
        print("\n[test-mode] Skipping WER computation")

    # Save results
    if is_test_mode:
        results_dir = Path("/home/ebrahim/brainaudio/results/test_files")
        cleaned_sentences = [clean_string(sent) for sent in decoded_sentences]
    else:
        results_dir = Path("/home/ebrahim/brainaudio/results")
        cleaned_sentences = decoded_sentences
    results_dir.mkdir(parents=True, exist_ok=True)

    safe_results_filename = results_filename.replace("/", "_")
    csv_filename = f"{args.encoder_model_name}_{safe_results_filename}_{os.getpid()}.csv"
    csv_path = results_dir / csv_filename

    results_df = pd.DataFrame({
        "id": trial_indices,
        "text": cleaned_sentences
    })
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to {csv_path}")

    rtf_path = csv_path.with_suffix('.npy')
    np.save(rtf_path, rtf_array)
    print(f"Saved RTF values to {rtf_path}")

    return {
        "decoded_sentences": decoded_sentences,
        "ground_truth_sentences": ground_truth_sentences,
        "trial_indices": trial_indices,
        "wer": wer,
        "peak_vram_gb": peak_vram_gb,
        "peak_cpu_memory_gb": peak_cpu_memory_gb,
        "total_elapsed": total_elapsed,
    }


def get_trial_indices_from_npz(npz_path):
    """Extract sorted trial indices from an NPZ file."""
    npz = np.load(npz_path)
    trial_keys = [k for k in npz.keys() if k.startswith("arr_")]
    return sorted([int(k.split("_")[1]) for k in trial_keys])


def load_llm(args, device):
    """Load LLM and tokenizer for shallow fusion. Returns (lm_fusion, None) or (None, None) if disabled."""
    if args.disable_llm:
        print("[INFO] LLM shallow fusion disabled (--disable-llm)")
        return None

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # Load LoRA adapter unless --no-adapter is specified
    if not args.no_adapter and args.lora_adapter is not None:
        model = PeftModel.from_pretrained(model, args.lora_adapter)
        model = model.merge_and_unload()
        print(f"[INFO] Loaded and merged LoRA adapter from {args.lora_adapter}")

    return LLMRescorer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        llm_weight=args.llm_weight,
        ngram_weight=args.ngram_rescore_weight,
        scoring_chunk_size=args.scoring_chunk_size,
        length_normalize=args.length_normalize,
    )


def main():
    args = parse_args()

    # Validate inputs
    if args.logits_val_path is None and args.logits_test_path is None:
        raise ValueError("At least one of --logits-val-path or --logits-test-path is required")

    device = pick_device(args.device)
    process = psutil.Process()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Auto-select LoRA adapter based on model
    if args.lora_adapter is None and not args.no_adapter and not args.disable_llm:
        model_lower = args.model.lower()
        if "1b" in model_lower:
            args.lora_adapter = LORA_ADAPTER_1B
        elif "3b" in model_lower:
            args.lora_adapter = LORA_ADAPTER_3B

    # Initialize wandb
    if not args.no_wandb:
        import wandb
        wandb.init(
            project="brainaudio-neural-lm-fusion",
            config=vars(args),
            name=args.results_filename,
            mode="online",
        )
    else:
        print("[wandb] W&B logging disabled")

    print(f"Loading resources on {device}...")

    lexicon = VectorizedLexiconConstraint.from_file_paths(
        tokens_file=str(args.tokens),
        lexicon_file=str(args.lexicon),
        device=device,
    )

    lm_fusion = load_llm(args, device)

    # Initialize word-level N-gram LM
    word_ngram_lm = FastNGramLM(args.word_lm_path, alpha=args.alpha_ngram)
    word_history = WordHistory()

    print(f"[INFO] Word N-gram LM loaded with alpha={args.alpha_ngram}")

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
        word_boundary_bonus=args.word_boundary_bonus,
        word_ngram_lm=word_ngram_lm,
        word_history=word_history,
        lm_rescore_interval=args.lm_rescore_interval,
        score_combination=args.score_combination,
    )

    # Load transcripts for validation
    transcripts = None
    if TRANSCRIPTS_PKL.exists():
        transcripts = pd.read_pickle(TRANSCRIPTS_PKL)
    else:
        print(f"Warning: Transcripts file not found at {TRANSCRIPTS_PKL}")

    val_results = None

    # Run validation if val path provided
    if args.logits_val_path:
        val_trial_indices = get_trial_indices_from_npz(args.logits_val_path)
        if args.trial_indices is not None:
            val_trial_indices = args.trial_indices
        elif args.random is not None:
            rng = np.random.default_rng(seed=42)
            val_trial_indices = sorted(rng.choice(val_trial_indices, size=min(args.random, len(val_trial_indices)), replace=False).tolist())
        elif args.start_trial_idx is not None or args.end_trial_idx is not None:
            start = args.start_trial_idx or 0
            end = args.end_trial_idx or max(val_trial_indices) + 1
            val_trial_indices = [i for i in val_trial_indices if start <= i < end]

        val_results = run_decode_pass(
            args, decoder, device, word_history, lexicon, lm_fusion, process,
            logits_path=args.logits_val_path,
            trial_indices=val_trial_indices,
            transcripts=transcripts,
            is_test_mode=False,
            results_filename=args.results_filename
        )

    # Run test if test path provided
    if args.logits_test_path:
        if args.logits_val_path:
            word_history.reset()
        test_trial_indices = get_trial_indices_from_npz(args.logits_test_path)
        run_decode_pass(
            args, decoder, device, word_history, lexicon, lm_fusion, process,
            logits_path=args.logits_test_path,
            trial_indices=test_trial_indices,
            transcripts=None,
            is_test_mode=True,
            results_filename=args.results_filename + "_test"
        )

    # Log to wandb
    if not args.no_wandb and val_results:
        import wandb
        if val_results["wer"] is not None:
            wandb.log({"WER": val_results["wer"]})
        wandb.log({
            "peak_vram_gb": val_results["peak_vram_gb"],
            "peak_cpu_memory_gb": val_results["peak_cpu_memory_gb"],
            "total_time_seconds": val_results["total_elapsed"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
