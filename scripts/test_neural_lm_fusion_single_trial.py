"""Tiny end-to-end CTC beam-search check with neural LM fusion.

Loops through a range of trials, decodes them, tracks performance,
and computes final WER/CER metrics.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

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

DEFAULT_LOGITS = "/data2/brain2text/b2t_25/logits/tm_transformer_b2t_24+25_large_wide_bidir_grad_clip_cosine_decay/logits_val.npz"
DEFAULT_TOKENS = "/data2/brain2text/lm/units_pytorch.txt"
DEFAULT_LEXICON = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt"
TRANSCRIPTS_PKL = Path("/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CTC beam search + HF LM fusion loop")
    
    parser.add_argument("--start-trial-idx", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end-trial-idx", type=int, default=None, help="End index (exclusive); defaults to all trials in logits file")
    parser.add_argument("--beam-size", type=int, default=100, help="CTC beam size")
    parser.add_argument("--model", default="google/gemma-3-270m", help="HF causal LM checkpoint")
    parser.add_argument("--hf-token", default=None, help="Optional HF token")
    parser.add_argument("--lm-weight", type=float, default=1.5, help="Fusion weight")
    parser.add_argument("--word-insertion-bonus", type=float, default=5, help="Bonus at boundaries")
    parser.add_argument("--max-context-length", type=int, default=512, help="Token budget")
    parser.add_argument("--device", default=None, help="Torch device")
    parser.add_argument("--logits", type=Path, default=Path(DEFAULT_LOGITS), help="NPZ logits file")
    parser.add_argument("--tokens", type=Path, default=Path(DEFAULT_TOKENS), help="units file")
    parser.add_argument("--lexicon", type=Path, default=Path(DEFAULT_LEXICON), help="lexicon file")
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)

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
    token_table = load_token_to_phoneme_mapping(args.tokens)
    phoneme_to_word = load_phoneme_to_word_mapping(args.lexicon)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        token=args.hf_token,
    ).to(device)

    lm_fusion = HuggingFaceLMFusion(
        model=model,
        tokenizer=tokenizer,
        weight=args.lm_weight,
        homophone_aggregation="max",
        device=device,
        max_context_length=args.max_context_length,
    )

    decoder = BatchedBeamCTCComputer(
        blank_index=lexicon.blank_index,
        beam_size=args.beam_size,
        lexicon=lexicon,
        lm_fusion=lm_fusion,
        word_insertion_bonus=args.word_insertion_bonus,
        allow_cuda_graphs=False,
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

        # Use context_texts for decoded beams (like test_neural_lm_fusion)
        top_k = min(5, result.scores.shape[1])  # Print up to 5 beams for inspection
        decoded_beams = result.context_texts[0][:top_k]
        best_text = decoded_beams[0] if decoded_beams else ""
        decoded_sentences.append(best_text)

        # Handle Ground Truth
        ground_truth = ""
        if transcripts is not None:
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
        print("   Top beams:")
        for beam_rank, beam_text in enumerate(decoded_beams):
            beam_score = result.scores[0, beam_rank].item()
            print(f"     #{beam_rank:02d} | log {beam_score:.4f} | {beam_text}")
        print("-" * 40)

    total_elapsed = time.perf_counter() - total_start_time

    print("\n=== Final Results Summary ===")
    print(f"Processed {len(decoded_sentences)} trials in {total_elapsed:.2f}s")

    # Compute Metrics
    if transcripts is not None:
        try:
            # Passing lists of strings as expected by standard WER calculators
            _, wer, _ = _cer_and_wer(decoded_sentences, ground_truth_sentences)
            print(f"\nAggregate WER: {wer:.4f}")
        except Exception as e:
            print(f"\nError computing WER: {e}")
            
    # Save decoded and ground truth sentences to file
    output_path = Path("decoded_sentences_12_11_beamsize_100.txt")
    # Compute WER for file header
    wer_str = "WER: N/A"
    if transcripts is not None:
        try:
            _, wer, _ = _cer_and_wer(decoded_sentences, ground_truth_sentences)
            wer_str = f"WER: {wer:.4f}"
        except Exception as e:
            wer_str = f"WER: ERROR ({e})"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(wer_str + "\n\n")
        for idx, (gt, pred) in enumerate(zip(ground_truth_sentences, decoded_sentences)):
            f.write(f"{idx}\n")
            f.write(f"gt: {gt}\n")
            f.write(f"pred: {pred}\n\n")
    print(f"\nSaved decoded and ground truth sentences to {output_path}")

if __name__ == "__main__":
    main()