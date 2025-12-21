"""
Compare predictions from LLM beam search against RNN baseline.
Prints sentences where the WER differs between the two methods.
"""

import argparse
import pandas as pd
from pathlib import Path
from jiwer import wer as compute_wer


DEFAULT_PREDICTIONS_CSV = "/home/ebrahim/nejm-brain-to-text/model_training/rnn_baseline_submission_file_valsplit.csv"
DEFAULT_TRANSCRIPTS_PKL = "/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl"
RESULTS_DIR = Path("/home/ebrahim/brainaudio/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LLM beam search predictions against RNN baseline")
    parser.add_argument(
        "results_csv",
        type=str,
        help="Name of the CSV file in the results folder (e.g., 'pretrained_RNN_12_21_1430.csv')"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(DEFAULT_PREDICTIONS_CSV),
        help="Path to baseline predictions CSV"
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=Path(DEFAULT_TRANSCRIPTS_PKL),
        help="Path to ground truth transcripts pickle file"
    )
    parser.add_argument(
        "--show-better",
        action="store_true",
        help="Show sentences where LLM beam search is better than baseline"
    )
    parser.add_argument(
        "--show-worse",
        action="store_true",
        help="Show sentences where LLM beam search is worse than baseline"
    )
    parser.add_argument(
        "--show-all-diff",
        action="store_true",
        help="Show all sentences where predictions differ (default if no filter specified)"
    )
    return parser.parse_args()


def sentence_wer(prediction: str, reference: str) -> float:
    """Compute WER for a single sentence (case-insensitive)."""
    if not reference.strip():
        return 0.0 if not prediction.strip() else 1.0
    try:
        return compute_wer(reference.lower(), prediction.lower())
    except:
        return 1.0


def main():
    args = parse_args()

    # Default to showing all differences if no filter specified
    if not args.show_better and not args.show_worse and not args.show_all_diff:
        args.show_all_diff = True

    # Create output filename based on input
    input_stem = Path(args.results_csv).stem
    output_csv_path = RESULTS_DIR / f"{input_stem}_comparison.csv"

    # Load LLM beam search predictions
    results_path = RESULTS_DIR / args.results_csv
    if not results_path.exists():
        # Try with .csv extension
        results_path = RESULTS_DIR / f"{args.results_csv}.csv"

    if not results_path.exists():
        print(f"Error: Could not find {args.results_csv} in {RESULTS_DIR}")
        print(f"Available files:")
        for f in RESULTS_DIR.glob("*.csv"):
            print(f"  {f.name}")
        return

    print(f"Loading LLM predictions from {results_path}")
    llm_df = pd.read_csv(results_path)
    llm_df = llm_df.sort_values("id").reset_index(drop=True)
    print(f"  Loaded {len(llm_df)} predictions")

    # Load baseline predictions
    print(f"Loading baseline predictions from {args.baseline}")
    baseline_df = pd.read_csv(args.baseline)
    baseline_df = baseline_df.sort_values("id").reset_index(drop=True)
    print(f"  Loaded {len(baseline_df)} predictions")

    # Load ground truth
    print(f"Loading ground truth from {args.transcripts}")
    transcripts = pd.read_pickle(args.transcripts)
    print(f"  Loaded {len(transcripts)} transcripts")

    # Get the range of IDs from LLM predictions
    llm_ids = set(llm_df["id"].tolist())

    # Compare predictions
    better_count = 0
    worse_count = 0
    same_count = 0
    both_perfect = 0

    results = []

    for _, row in llm_df.iterrows():
        idx = row["id"]
        llm_pred = str(row["text"]) if pd.notna(row["text"]) else ""

        # Get baseline prediction for this ID
        baseline_row = baseline_df[baseline_df["id"] == idx]
        if baseline_row.empty:
            continue
        baseline_pred = str(baseline_row["text"].values[0]) if pd.notna(baseline_row["text"].values[0]) else ""

        # Get ground truth
        if idx >= len(transcripts):
            continue
        ground_truth = transcripts[idx]

        # Compute WER for each
        llm_wer = sentence_wer(llm_pred, ground_truth)
        baseline_wer = sentence_wer(baseline_pred, ground_truth)

        # Categorize
        if llm_wer < baseline_wer:
            category = "BETTER"
            better_count += 1
        elif llm_wer > baseline_wer:
            category = "WORSE"
            worse_count += 1
        else:
            category = "SAME"
            same_count += 1
            if llm_wer == 0:
                both_perfect += 1

        results.append({
            "idx": idx,
            "ground_truth": ground_truth,
            "llm_pred": llm_pred,
            "baseline_pred": baseline_pred,
            "llm_wer": llm_wer,
            "baseline_wer": baseline_wer,
            "category": category,
        })

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total sentences compared: {len(results)}")
    print(f"  LLM better than baseline: {better_count} ({100*better_count/len(results):.1f}%)")
    print(f"  LLM worse than baseline:  {worse_count} ({100*worse_count/len(results):.1f}%)")
    print(f"  Same WER:                 {same_count} ({100*same_count/len(results):.1f}%)")
    print(f"    (Both perfect:          {both_perfect})")

    # Compute aggregate WER (case-insensitive)
    llm_preds_all = [r["llm_pred"].lower() for r in results]
    baseline_preds_all = [r["baseline_pred"].lower() for r in results]
    gts_all = [r["ground_truth"].lower() for r in results]

    llm_agg_wer = compute_wer(gts_all, llm_preds_all)
    baseline_agg_wer = compute_wer(gts_all, baseline_preds_all)

    print(f"\nAggregate WER:")
    print(f"  LLM beam search: {llm_agg_wer:.4f} ({llm_agg_wer*100:.2f}%)")
    print(f"  RNN baseline:    {baseline_agg_wer:.4f} ({baseline_agg_wer*100:.2f}%)")

    # Print detailed comparisons
    if args.show_better or args.show_all_diff:
        better_results = [r for r in results if r["category"] == "BETTER"]
        if better_results:
            print("\n" + "=" * 80)
            print(f"LLM BETTER THAN BASELINE ({len(better_results)} sentences)")
            print("=" * 80)
            for r in better_results:
                print(f"\n[{r['idx']}] LLM WER: {r['llm_wer']:.2f} vs Baseline WER: {r['baseline_wer']:.2f}")
                print(f"  GT:       {r['ground_truth']}")
                print(f"  LLM:      {r['llm_pred']}")
                print(f"  Baseline: {r['baseline_pred']}")

    if args.show_worse or args.show_all_diff:
        worse_results = [r for r in results if r["category"] == "WORSE"]
        if worse_results:
            print("\n" + "=" * 80)
            print(f"LLM WORSE THAN BASELINE ({len(worse_results)} sentences)")
            print("=" * 80)
            for r in worse_results:
                print(f"\n[{r['idx']}] LLM WER: {r['llm_wer']:.2f} vs Baseline WER: {r['baseline_wer']:.2f}")
                print(f"  GT:       {r['ground_truth']}")
                print(f"  LLM:      {r['llm_pred']}")
                print(f"  Baseline: {r['baseline_pred']}")

    # Save sentences with different WER to CSV
    diff_results = [r for r in results if r["category"] != "SAME"]
    if diff_results:
        comparison_df = pd.DataFrame([
            {
                "id": r["idx"],
                "llm_wer": round(r["llm_wer"], 2),
                "baseline_wer": round(r["baseline_wer"], 2),
                "ground_truth": r["ground_truth"],
                "llm_prediction": r["llm_pred"],
                "baseline_prediction": r["baseline_pred"],
            }
            for r in diff_results
        ])
        comparison_df.to_csv(output_csv_path, index=False)
        print(f"\nSaved {len(diff_results)} differing sentences to {output_csv_path}")


if __name__ == "__main__":
    main()
