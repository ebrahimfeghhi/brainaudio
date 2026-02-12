"""
Compare predictions from LLM beam search against RNN baseline.
Prints sentences where the predictions differ between the two methods.
"""

import argparse
import random
import re
import pandas as pd
from pathlib import Path
from jiwer import wer as compute_wer, process_words

RANDOM_SEED = 42

base_path = "/home/ebrahim/"

DEFAULT_PREDICTIONS_CSV = "/home/ebrahim/nejm-brain-to-text/model_training/rnn_baseline_submission_file_valsplit.csv"
DEFAULT_TRANSCRIPTS_PKL = f"{base_path}/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl"
RESULTS_DIR = Path("/home/ebrahim/brainaudio/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LLM beam search predictions against RNN baseline")
    parser.add_argument(
        "--results-csv",
        type=str,
        default="best_chunked_transformer_combined_seed_0_02_11_2151_1430117.csv",
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


def clean_string(text: str) -> str:
    """Normalize text: keep only letters, hyphens, apostrophes, and spaces."""
    text = re.sub(r"[^a-zA-Z\- \']", "", text)
    text = text.replace("--", "").lower().strip()
    return text


def sentence_wer(prediction: str, reference: str) -> float:
    """Compute WER for a single sentence (case-insensitive, keeps apostrophes/hyphens)."""

    # Normalization: keep letters, hyphens, apostrophes, spaces
    p_clean = clean_string(prediction)
    r_clean = clean_string(reference)
    
    # 2. Handle empty references (e.g., silence or only punctuation)
    if not r_clean:
        return 0.0 if not p_clean else 1.0
        
    try:
        # compute_wer must be defined in your scope (e.g. from jiwer)
        return compute_wer(r_clean, p_clean)
    except Exception as e:
        # It is often better to print the error for debugging than to silently return 1.0
        print(f"WER Calculation Error: {e}")
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

        # Check if predictions differ (normalized)
        llm_normalized = clean_string(llm_pred)
        baseline_normalized = clean_string(baseline_pred)
        text_differs = llm_normalized != baseline_normalized

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
            "text_differs": text_differs,
        })

    # Count sentences with different text
    diff_text_count = sum(1 for r in results if r["text_differs"])
    same_wer_diff_text_count = sum(1 for r in results if r["category"] == "SAME" and r["text_differs"])

    # Compute aggregate WER and error types (normalized)
    llm_preds_all = [clean_string(r["llm_pred"]) for r in results]
    baseline_preds_all = [clean_string(r["baseline_pred"]) for r in results]
    gts_all = [clean_string(r["ground_truth"]) for r in results]

    # Process words to get detailed error counts
    llm_output = process_words(gts_all, llm_preds_all)
    baseline_output = process_words(gts_all, baseline_preds_all)

    # Print error analysis first
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    print(f"{'Metric':<20} {'LLM Beam Search':>18} {'RNN Baseline':>18} {'Difference':>12}")
    print("-" * 80)
    print(f"{'Substitutions':<20} {llm_output.substitutions:>18} {baseline_output.substitutions:>18} {llm_output.substitutions - baseline_output.substitutions:>+12}")
    print(f"{'Insertions':<20} {llm_output.insertions:>18} {baseline_output.insertions:>18} {llm_output.insertions - baseline_output.insertions:>+12}")
    print(f"{'Deletions':<20} {llm_output.deletions:>18} {baseline_output.deletions:>18} {llm_output.deletions - baseline_output.deletions:>+12}")
    print("-" * 80)
    llm_total_errors = llm_output.substitutions + llm_output.insertions + llm_output.deletions
    baseline_total_errors = baseline_output.substitutions + baseline_output.insertions + baseline_output.deletions
    print(f"{'Total Errors':<20} {llm_total_errors:>18} {baseline_total_errors:>18} {llm_total_errors - baseline_total_errors:>+12}")
    print(f"{'Hits (Correct)':<20} {llm_output.hits:>18} {baseline_output.hits:>18} {llm_output.hits - baseline_output.hits:>+12}")
    print("-" * 80)
    print(f"{'WER':<20} {llm_output.wer:>17.2%} {baseline_output.wer:>17.2%} {(llm_output.wer - baseline_output.wer)*100:>+11.2f}%")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total sentences compared: {len(results)}")
    print(f"  LLM better than baseline: {better_count} ({100*better_count/len(results):.1f}%)")
    print(f"  LLM worse than baseline:  {worse_count} ({100*worse_count/len(results):.1f}%)")
    print(f"  Same WER:                 {same_count} ({100*same_count/len(results):.1f}%)")
    print(f"    (Both perfect:          {both_perfect})")
    print(f"  Different text:           {diff_text_count} ({100*diff_text_count/len(results):.1f}%)")
    print(f"    (Same WER, diff text:   {same_wer_diff_text_count})")

    # Print detailed comparisons (only sentences where text differs)
    if args.show_better or args.show_all_diff:
        better_results = [r for r in results if r["category"] == "BETTER" and r["text_differs"]]
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
        worse_results = [r for r in results if r["category"] == "WORSE" and r["text_differs"]]
        if worse_results:
            print("\n" + "=" * 80)
            print(f"LLM WORSE THAN BASELINE ({len(worse_results)} sentences)")
            print("=" * 80)
            for r in worse_results:
                print(f"\n[{r['idx']}] LLM WER: {r['llm_wer']:.2f} vs Baseline WER: {r['baseline_wer']:.2f}")
                print(f"  GT:       {r['ground_truth']}")
                print(f"  LLM:      {r['llm_pred']}")
                print(f"  Baseline: {r['baseline_pred']}")

            # Print indices in copy-pasteable format
            worse_indices = [r['idx'] for r in worse_results]
            print("\n" + "-" * 80)
            print(f"Copy-pasteable trial indices ({len(worse_indices)} trials):")
            print(' '.join(str(i) for i in worse_indices))

    if args.show_all_diff:
        # Show sentences where text differs but WER is the same
        same_wer_diff_text = [r for r in results if r["category"] == "SAME" and r["text_differs"]]
        if same_wer_diff_text:
            print("\n" + "=" * 80)
            print(f"SAME WER BUT DIFFERENT TEXT ({len(same_wer_diff_text)} sentences)")
            print("=" * 80)
            for r in same_wer_diff_text:
                print(f"\n[{r['idx']}] WER: {r['llm_wer']:.2f} (same for both)")
                print(f"  GT:       {r['ground_truth']}")
                print(f"  LLM:      {r['llm_pred']}")
                print(f"  Baseline: {r['baseline_pred']}")

    # Save sentences with different text to CSV (includes both different WER and same WER but different text)
    diff_results = [r for r in results if r["text_differs"]]
    if diff_results:
        # Get worse trial indices and compute WER for worse trials only
        worse_results_for_csv = [r for r in diff_results if r["category"] == "WORSE"]
        worse_indices = [r['idx'] for r in worse_results_for_csv]

        # Compute WER for worse trials subset
        if worse_results_for_csv:
            worse_llm_preds = [clean_string(r["llm_pred"]) for r in worse_results_for_csv]
            worse_baseline_preds = [clean_string(r["baseline_pred"]) for r in worse_results_for_csv]
            worse_gts = [clean_string(r["ground_truth"]) for r in worse_results_for_csv]
            worse_llm_output = process_words(worse_gts, worse_llm_preds)
            worse_baseline_output = process_words(worse_gts, worse_baseline_preds)
        else:
            worse_llm_output = None
            worse_baseline_output = None

        # Compute WER for random subset of 50 worse trials (reproducible)
        random_subset_size_50 = 50
        random_subset_50_output = None
        random_subset_50_baseline_output = None
        random_subset_50_indices = []
        if len(worse_results_for_csv) >= random_subset_size_50:
            random.seed(RANDOM_SEED)
            random_subset_50 = random.sample(worse_results_for_csv, random_subset_size_50)
            random_subset_50_indices = [r['idx'] for r in random_subset_50]
            random_50_llm_preds = [clean_string(r["llm_pred"]) for r in random_subset_50]
            random_50_baseline_preds = [clean_string(r["baseline_pred"]) for r in random_subset_50]
            random_50_gts = [clean_string(r["ground_truth"]) for r in random_subset_50]
            random_subset_50_output = process_words(random_50_gts, random_50_llm_preds)
            random_subset_50_baseline_output = process_words(random_50_gts, random_50_baseline_preds)

        # Compute WER for random subset of 25 worse trials (reproducible)
        random_subset_size_25 = 25
        random_subset_25_output = None
        random_subset_25_baseline_output = None
        random_subset_25_indices = []
        if len(worse_results_for_csv) >= random_subset_size_25:
            random.seed(RANDOM_SEED)
            random_subset_25 = random.sample(worse_results_for_csv, random_subset_size_25)
            random_subset_25_indices = [r['idx'] for r in random_subset_25]
            random_25_llm_preds = [clean_string(r["llm_pred"]) for r in random_subset_25]
            random_25_baseline_preds = [clean_string(r["baseline_pred"]) for r in random_subset_25]
            random_25_gts = [clean_string(r["ground_truth"]) for r in random_subset_25]
            random_subset_25_output = process_words(random_25_gts, random_25_llm_preds)
            random_subset_25_baseline_output = process_words(random_25_gts, random_25_baseline_preds)

        # Compute WER for random subset of 10 worse trials (reproducible)
        random_subset_size_10 = 10
        random_subset_10_output = None
        random_subset_10_baseline_output = None
        random_subset_10_indices = []
        if len(worse_results_for_csv) >= random_subset_size_10:
            random.seed(RANDOM_SEED)
            random_subset_10 = random.sample(worse_results_for_csv, random_subset_size_10)
            random_subset_10_indices = [r['idx'] for r in random_subset_10]
            random_10_llm_preds = [clean_string(r["llm_pred"]) for r in random_subset_10]
            random_10_baseline_preds = [clean_string(r["baseline_pred"]) for r in random_subset_10]
            random_10_gts = [clean_string(r["ground_truth"]) for r in random_subset_10]
            random_subset_10_output = process_words(random_10_gts, random_10_llm_preds)
            random_subset_10_baseline_output = process_words(random_10_gts, random_10_baseline_preds)

        # Build error analysis header
        header_lines = [
            "# ERROR ANALYSIS",
            f"# {'Metric':<20} {'LLM Beam Search':>18} {'RNN Baseline':>18} {'Difference':>12}",
            f"# {'-'*70}",
            f"# {'Substitutions':<20} {llm_output.substitutions:>18} {baseline_output.substitutions:>18} {llm_output.substitutions - baseline_output.substitutions:>+12}",
            f"# {'Insertions':<20} {llm_output.insertions:>18} {baseline_output.insertions:>18} {llm_output.insertions - baseline_output.insertions:>+12}",
            f"# {'Deletions':<20} {llm_output.deletions:>18} {baseline_output.deletions:>18} {llm_output.deletions - baseline_output.deletions:>+12}",
            f"# {'-'*70}",
            f"# {'Total Errors':<20} {llm_total_errors:>18} {baseline_total_errors:>18} {llm_total_errors - baseline_total_errors:>+12}",
            f"# {'Hits (Correct)':<20} {llm_output.hits:>18} {baseline_output.hits:>18} {llm_output.hits - baseline_output.hits:>+12}",
            f"# {'-'*70}",
            f"# {'WER':<20} {llm_output.wer:>17.2%} {baseline_output.wer:>17.2%} {(llm_output.wer - baseline_output.wer)*100:>+11.2f}%",
            "#",
            f"# WER ON WORSE TRIALS ONLY ({len(worse_indices)} trials):",
            f"# {'LLM WER':<20} {worse_llm_output.wer:>17.2%}" if worse_llm_output else "# No worse trials",
            f"# {'Baseline WER':<20} {worse_baseline_output.wer:>17.2%}" if worse_baseline_output else "",
            "#",
            f"# WER ON RANDOM SUBSET OF {random_subset_size_50} WORSE TRIALS (seed={RANDOM_SEED}):" if random_subset_50_output else f"# (Fewer than {random_subset_size_50} worse trials, skipping random subset)",
            f"# {'LLM WER':<20} {random_subset_50_output.wer:>17.2%}" if random_subset_50_output else "",
            f"# {'Baseline WER':<20} {random_subset_50_baseline_output.wer:>17.2%}" if random_subset_50_baseline_output else "",
            "#",
            f"# RANDOM 50 SUBSET INDICES ({len(random_subset_50_indices)} trials):" if random_subset_50_indices else "",
            f"{' '.join(str(i) for i in sorted(random_subset_50_indices))}" if random_subset_50_indices else "",
            "#" if random_subset_50_indices else "",
            f"# WER ON RANDOM SUBSET OF {random_subset_size_25} WORSE TRIALS (seed={RANDOM_SEED}):" if random_subset_25_output else f"# (Fewer than {random_subset_size_25} worse trials, skipping random subset)",
            f"# {'LLM WER':<20} {random_subset_25_output.wer:>17.2%}" if random_subset_25_output else "",
            f"# {'Baseline WER':<20} {random_subset_25_baseline_output.wer:>17.2%}" if random_subset_25_baseline_output else "",
            "#",
            f"# RANDOM 25 SUBSET INDICES ({len(random_subset_25_indices)} trials):" if random_subset_25_indices else "",
            f"{' '.join(str(i) for i in sorted(random_subset_25_indices))}" if random_subset_25_indices else "",
            "#" if random_subset_25_indices else "",
            f"# WER ON RANDOM SUBSET OF {random_subset_size_10} WORSE TRIALS (seed={RANDOM_SEED}):" if random_subset_10_output else f"# (Fewer than {random_subset_size_10} worse trials, skipping random subset)",
            f"# {'LLM WER':<20} {random_subset_10_output.wer:>17.2%}" if random_subset_10_output else "",
            f"# {'Baseline WER':<20} {random_subset_10_baseline_output.wer:>17.2%}" if random_subset_10_baseline_output else "",
            "#",
            f"# RANDOM 10 SUBSET INDICES ({len(random_subset_10_indices)} trials):" if random_subset_10_indices else "",
            f"{' '.join(str(i) for i in sorted(random_subset_10_indices))}" if random_subset_10_indices else "",
            "#" if random_subset_10_indices else "",
            f"# WORSE TRIAL INDICES ({len(worse_indices)} trials):",
            f"{' '.join(str(i) for i in worse_indices)}",
            "#",
        ]

        comparison_df = pd.DataFrame([
            {
                "id": r["idx"],
                "llm_wer": round(r["llm_wer"], 2),
                "baseline_wer": round(r["baseline_wer"], 2),
                "category": r["category"],
                "ground_truth": r["ground_truth"],
                "llm_prediction": r["llm_pred"],
                "baseline_prediction": r["baseline_pred"],
            }
            for r in worse_results_for_csv
        ])

        # Write header then CSV
        with open(output_csv_path, 'w') as f:
            f.write('\n'.join(header_lines) + '\n')
            comparison_df.to_csv(f, index=False)

        print(f"\nSaved {len(worse_results_for_csv)} worse sentences to {output_csv_path}")


if __name__ == "__main__":
    main()
