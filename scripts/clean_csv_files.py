#!/usr/bin/env python3
"""Script to clean text in CSV files using the clean_string function."""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brainaudio.inference.eval_metrics import clean_string


def process_csv(input_path: Path, output_path: Path) -> None:
    """Process a CSV file and clean the text column."""
    with open(input_path, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if "text" not in fieldnames:
            print(f"Warning: 'text' column not found in {input_path}, skipping.")
            return

        rows = []
        for row in reader:
            row["text"] = clean_string(row["text"])
            rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed: {input_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Clean text in CSV files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/test_files"),
        help="Input directory containing CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: overwrites input files)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_cleaned",
        help="Suffix to add to output filenames (used when output-dir is not specified)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        sys.exit(0)

    for csv_file in csv_files:
        if args.output_dir:
            output_path = output_dir / csv_file.name
        else:
            output_path = output_dir / f"{csv_file.stem}{args.suffix}.csv"

        process_csv(csv_file, output_path)

    print(f"\nProcessed {len(csv_files)} file(s)")


if __name__ == "__main__":
    main()
