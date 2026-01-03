#!/usr/bin/env python3
"""Compute WER for specific trial indices."""

import argparse
import pandas as pd
from pathlib import Path

from brainaudio.inference.eval_metrics import _cer_and_wer


DEFAULT_TRANSCRIPTS_PKL = "/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute WER for specific trials")
    parser.add_argument(
        "predictions",
        type=Path,
        help="Path to CSV file with predictions (columns: id, text)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        nargs="+",
        required=True,
        help="Trial indices to evaluate"
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=Path(DEFAULT_TRANSCRIPTS_PKL),
        help="Path to ground truth transcripts pickle file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sentence comparisons"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load predictions
    print(f"Loading predictions from {args.predictions}")
    predictions_df = pd.read_csv(args.predictions)
    predictions_df = predictions_df.set_index("id")
    print(f"  Loaded {len(predictions_df)} predictions")

    # Load ground truth transcripts
    print(f"Loading transcripts from {args.transcripts}")
    transcripts = pd.read_pickle(args.transcripts)
    print(f"  Loaded {len(transcripts)} transcripts")

    # Get predictions and ground truth for specified trials
    trial_indices = args.trials
    print(f"\nEvaluating {len(trial_indices)} trials: {trial_indices[:5]}{'...' if len(trial_indices) > 5 else ''}")

    predicted_sentences = []
    ground_truth_sentences = []

    for idx in trial_indices:
        if idx in predictions_df.index:
            predicted_sentences.append(predictions_df.loc[idx, "text"])
            ground_truth_sentences.append(transcripts[idx])
        else:
            print(f"  Warning: Trial {idx} not found in predictions")

    # Verbose output
    if args.verbose:
        print("\n" + "=" * 80)
        print("Per-sentence comparison:")
        print("=" * 80)
        for i, (trial_idx, gt, pred) in enumerate(zip(trial_indices, ground_truth_sentences, predicted_sentences)):
            print(f"\n[Trial {trial_idx}]")
            print(f"  GT:   {gt}")
            print(f"  Pred: {pred}")

    # Compute WER
    print("\n" + "=" * 80)
    print("Computing metrics...")
    print("=" * 80)

    cer, wer, per_sentence_wer = _cer_and_wer(predicted_sentences, ground_truth_sentences)

    print(f"\nResults:")
    print(f"  CER: {cer:.4f} ({cer * 100:.2f}%)")
    print(f"  WER: {wer:.4f} ({wer * 100:.2f}%)")
    print(f"  Total sentences: {len(predicted_sentences)}")


if __name__ == "__main__":
    main()
