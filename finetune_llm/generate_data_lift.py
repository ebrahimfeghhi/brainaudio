"""
Generate finetuning dataset for LLM-based generative corrector.

This script:
1. Loads trained encoder model(s) for b2t_24 and b2t_25
2. Runs beam search decoding with large beam size on training data
3. Extracts top N beam hypotheses for each sample
4. Creates instruction-tuning dataset: input=top N beams, output=ground truth sentence
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer,
    VectorizedLexiconConstraint,
    materialize_beam_transcript,
    collapse_ctc_sequence,
)
from brainaudio.inference.decoder.word_ngram_lm_optimized import FastNGramLM, WordHistory
from brainaudio.inference.decoder.beam_helpers import pick_device


def parse_args():
    parser = argparse.ArgumentParser("Generate dataset for LLM generative corrector")

    # Data paths
    parser.add_argument("--logits-path", type=Path, required=True,
                       help="Path to logits NPZ file (training set)")
    parser.add_argument("--transcripts-path", type=Path, required=True,
                       help="Path to ground truth transcripts pickle file")
    parser.add_argument("--output-path", type=Path, required=True,
                       help="Output path for generated dataset (JSONL)")

    # Decoder resources
    parser.add_argument("--tokens", type=Path, required=True,
                       help="Path to tokens/units file")
    parser.add_argument("--lexicon", type=Path, required=True,
                       help="Path to lexicon file")
    parser.add_argument("--word-lm-path", type=str, required=True,
                       help="Path to KenLM n-gram model")

    # Beam search settings
    parser.add_argument("--beam-size", type=int, default=300,
                       help="Beam size for decoding (should be >= top-n)")
    parser.add_argument("--top-n", type=int, default=100,
                       help="Number of top beams to extract per sample")
    parser.add_argument("--num-homophone-beams", type=int, default=1,
                       help="Number of homophone alternatives per beam")
    parser.add_argument("--beam-prune-threshold", type=float, default=25.0,
                       help="Beam pruning threshold")
    parser.add_argument("--homophone-prune-threshold", type=float, default=5.0,
                       help="Homophone pruning threshold")

    # Acoustic settings
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for logits")
    parser.add_argument("--acoustic-scale", type=float, default=0.4,
                       help="Acoustic scaling factor")

    # N-gram LM settings
    parser.add_argument("--alpha-ngram", type=float, default=1.0,
                       help="N-gram LM weight")
    parser.add_argument("--beam-beta", type=float, default=1.5,
                       help="Beam extension bonus")
    parser.add_argument("--word-boundary-bonus", type=float, default=1.0,
                       help="Word boundary bonus")

    # Processing
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--trial-indices", type=int, nargs="*", default=None,
                       help="Specific trial indices to process (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process")

    # Instruction format
    parser.add_argument("--instruction-template", type=str,
                       default="Given the following {n} candidate transcriptions from a brain-to-text decoder, generate the most accurate and coherent transcription:",
                       help="Instruction template for the LLM")

    return parser.parse_args()


def format_beams_as_input(beams: List[str], instruction: str) -> str:
    """Format top N beams as input for instruction finetuning."""
    beam_text = "\n".join([f"{i+1}. {beam}" for i, beam in enumerate(beams)])
    return f"{instruction}\n\n{beam_text}\n\nGenerate the corrected transcription:"


def extract_top_beams(
    decoder_result,
    word_history: WordHistory,
    lexicon: VectorizedLexiconConstraint,
    batch_idx: int,
    top_n: int,
) -> List[str]:
    """Extract top N decoded beam hypotheses as text."""
    scores = decoder_result.scores[batch_idx].cpu().numpy()

    # Get valid beams (score != -inf)
    valid_indices = [i for i, s in enumerate(scores) if s != -float('inf')]

    # Sort by score descending
    valid_indices_sorted = sorted(valid_indices, key=lambda i: scores[i], reverse=True)

    # Take top N
    topk_indices = valid_indices_sorted[:top_n]

    beams = []
    for i in topk_indices:
        context_list = decoder_result.context_texts[batch_idx][i]
        if context_list:
            # Get the best homophone interpretation for this beam
            hist_id = context_list[0][2]  # (lm_score, _, hist_id)
            text = word_history.get_text(hist_id)
            beams.append(text)
        else:
            beams.append("")  # Empty beam

    return beams


def main():
    args = parse_args()

    print(f"Generating LLM corrector dataset...")
    print(f"  Logits: {args.logits_path}")
    print(f"  Transcripts: {args.transcripts_path}")
    print(f"  Output: {args.output_path}")
    print(f"  Beam size: {args.beam_size}, Top-N: {args.top_n}")

    device = pick_device(args.device)
    print(f"Using device: {device}")

    # Load resources
    print("Loading lexicon and LM...")
    lexicon = VectorizedLexiconConstraint.from_file_paths(
        tokens_file=str(args.tokens),
        lexicon_file=str(args.lexicon),
        device=device,
    )

    word_ngram_lm = FastNGramLM(args.word_lm_path, alpha=args.alpha_ngram)
    word_history = WordHistory()

    # Initialize decoder (no LLM fusion for dataset generation)
    print("Initializing beam search decoder...")
    decoder = BatchedBeamCTCComputer(
        blank_index=lexicon.blank_index,
        beam_size=args.beam_size,
        lexicon=lexicon,
        lm_fusion=None,  # No LLM fusion during data generation
        allow_cuda_graphs=False,
        num_homophone_beams=args.num_homophone_beams,
        beam_threshold=args.beam_prune_threshold,
        homophone_prune_threshold=args.homophone_prune_threshold,
        beam_beta=args.beam_beta,
        word_boundary_bonus=args.word_boundary_bonus,
        word_ngram_lm=word_ngram_lm,
        word_history=word_history,
        lm_rescore_interval=0,  # No rescoring
        score_combination="max",
    )

    # Load transcripts
    print(f"Loading ground truth transcripts from {args.transcripts_path}...")
    transcripts = pd.read_pickle(args.transcripts_path)
    print(f"Loaded {len(transcripts)} transcripts")

    # Load logits
    print(f"Loading logits from {args.logits_path}...")
    logits_npz = np.load(args.logits_path)

    # Get trial indices
    if args.trial_indices is not None:
        trial_indices = args.trial_indices
    else:
        trial_keys = [k for k in logits_npz.keys() if k.startswith("arr_")]
        trial_indices = sorted([int(k.split("_")[1]) for k in trial_keys])

    if args.max_samples is not None:
        trial_indices = trial_indices[:args.max_samples]

    print(f"Processing {len(trial_indices)} trials...")

    # Generate dataset
    dataset = []
    instruction = args.instruction_template.format(n=args.top_n)

    for trial_idx in tqdm(trial_indices, desc="Processing trials"):
        # Load logits for this trial
        logits = torch.from_numpy(logits_npz[f"arr_{trial_idx}"]).to(device).unsqueeze(0)

        # Add padding frames
        num_padding_frames = 2
        vocab_size = logits.shape[-1]
        padding = torch.zeros((1, num_padding_frames, vocab_size), device=device, dtype=logits.dtype)
        logits = torch.cat([logits, padding], dim=1)

        lengths = torch.tensor([logits.shape[1]], device=device)

        # Apply temperature and acoustic scaling
        log_probs = torch.nn.functional.log_softmax(
            logits / args.temperature, dim=-1
        ) * args.acoustic_scale

        # Run beam search
        result = decoder(log_probs, lengths)

        # Extract top N beams
        top_beams = extract_top_beams(
            decoder_result=result,
            word_history=word_history,
            lexicon=lexicon,
            batch_idx=0,
            top_n=args.top_n,
        )

        # Get ground truth
        if isinstance(transcripts, (list, tuple)):
            ground_truth = transcripts[trial_idx]
        elif hasattr(transcripts, 'get'):
            ground_truth = transcripts.get(trial_idx, "")
        elif hasattr(transcripts, 'iloc'):
            ground_truth = transcripts.iloc[trial_idx]
        else:
            ground_truth = str(transcripts[trial_idx])

        # Create dataset entry
        dataset_entry = {
            "trial_id": trial_idx,
            "input_beams": top_beams,
            "ground_truth": ground_truth,
            "instruction": instruction,
            "formatted_input": format_beams_as_input(top_beams, instruction),
        }
        dataset.append(dataset_entry)

        # Clean up
        del result, log_probs, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save dataset
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving dataset to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"Dataset generation complete!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Output file: {args.output_path}")

    # Print example
    if dataset:
        print("\n=== Example Entry ===")
        example = dataset[0]
        print(f"Trial ID: {example['trial_id']}")
        print(f"\nTop {min(3, len(example['input_beams']))} beams:")
        for i, beam in enumerate(example['input_beams'][:3]):
            print(f"  {i+1}. {beam}")
        print(f"\nGround truth: {example['ground_truth']}")


if __name__ == "__main__":
    main()
