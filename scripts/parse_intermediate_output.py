"""
Parse intermediate decoder output file and extract predictions.

Extracts trial IDs and "Best:" predictions from verbose decoder output,
then saves as CSV in the same format as compare_predictions.py expects.

Usage:
    python parse_intermediate_output.py intermediate_file_outputs.csv
    python parse_intermediate_output.py intermediate_file_outputs.csv --output parsed_results.csv
"""

import argparse
import re
import pandas as pd
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse intermediate decoder output")
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to intermediate output file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: input_file with _parsed suffix)"
    )
    return parser.parse_args()


def parse_intermediate_file(input_path: Path) -> list[dict]:
    """
    Parse the intermediate output file and extract trial IDs and best predictions.

    Returns:
        List of dicts with 'id' and 'text' keys
    """
    results = []

    with open(input_path, 'r') as f:
        content = f.read()

    # Pattern to match trial blocks
    # Trial   0 | 19682.3ms | Score: 24.41
    #   GT:   you can see the code at this point as well
    #   Best: You can see the code at this point as well.

    trial_pattern = re.compile(
        r'Trial\s+(\d+)\s*\|.*?\n'  # Trial line with ID
        r'\s+GT:\s+.*?\n'           # GT line (skip)
        r'\s+Best:\s+(.*?)\n',      # Best line (capture)
        re.MULTILINE
    )

    for match in trial_pattern.finditer(content):
        trial_id = int(match.group(1))
        best_text = match.group(2).strip()
        results.append({
            'id': trial_id,
            'text': best_text
        })

    return results


def main():
    args = parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return

    # Parse the file
    results = parse_intermediate_file(args.input_file)

    if not results:
        print("Error: No trials found in input file")
        return

    print(f"Parsed {len(results)} trials")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.input_file.parent / f"{args.input_file.stem}_parsed.csv"

    # Save as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Print first few results as sanity check
    print("\nFirst 5 results:")
    for r in results[:5]:
        print(f"  [{r['id']}] {r['text'][:60]}...")


if __name__ == "__main__":
    main()
