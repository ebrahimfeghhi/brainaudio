"""
Compute WER from a CSV submission file by comparing against ground truth transcripts.
"""

import argparse
import pandas as pd
from pathlib import Path

from brainaudio.inference.eval_metrics import _cer_and_wer


DEFAULT_PREDICTIONS_CSV = "/home/ebrahim/nejm-brain-to-text/model_training/rnn_baseline_submission_file_valsplit.csv"
DEFAULT_TRANSCRIPTS_PKL = "/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute WER from CSV submission file")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(DEFAULT_PREDICTIONS_CSV),
        help="Path to CSV file with predictions (columns: id, text)"
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
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start trial index (inclusive). Default: 0"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End trial index (exclusive). Default: all trials"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load predictions
    print(f"Loading predictions from {args.predictions}")
    predictions_df = pd.read_csv(args.predictions)
    print(f"  Loaded {len(predictions_df)} predictions")

    # Load ground truth transcripts
    print(f"Loading transcripts from {args.transcripts}")
    transcripts = pd.read_pickle(args.transcripts)
    print(f"  Loaded {len(transcripts)} transcripts")

    # Ensure predictions are sorted by id
    predictions_df = predictions_df.sort_values("id").reset_index(drop=True)

    # Filter by trial index range if specified
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(transcripts)

    # Filter predictions by id range
    predictions_df = predictions_df[(predictions_df["id"] >= start_idx) & (predictions_df["id"] < end_idx)]
    print(f"  Filtered to trials [{start_idx}, {end_idx}): {len(predictions_df)} predictions")

    # Extract predicted texts in order
    predicted_sentences = predictions_df["text"].tolist()
    ground_truth_sentences = list(transcripts[start_idx:end_idx])

    # Ensure same length
    if len(predicted_sentences) != len(ground_truth_sentences):
        print(f"WARNING: Prediction count ({len(predicted_sentences)}) != transcript count ({len(ground_truth_sentences)})")
        min_len = min(len(predicted_sentences), len(ground_truth_sentences))
        predicted_sentences = predicted_sentences[:min_len]
        ground_truth_sentences = ground_truth_sentences[:min_len]
        print(f"  Using first {min_len} entries")

    # Verbose output
    if args.verbose:
        print("\n" + "=" * 80)
        print("Per-sentence comparison:")
        print("=" * 80)
        for i, (gt, pred) in enumerate(zip(ground_truth_sentences, predicted_sentences)):
            print(f"\n[{i}]")
            print(f"  GT:   {gt}")
            print(f"  Pred: {pred}")

    # Compute WER
    print("\n" + "=" * 80)
    print("Computing metrics...")
    print("=" * 80)

    cer, wer, wer_details = _cer_and_wer(predicted_sentences, ground_truth_sentences)

    print(f"\nResults:")
    print(f"  CER: {cer:.4f} ({cer * 100:.2f}%)")
    print(f"  WER: {wer:.4f} ({wer * 100:.2f}%)")
    print(f"\n  Total sentences: {len(predicted_sentences)}")


if __name__ == "__main__":
    main()
