#!/usr/bin/env python3
"""Script to find sentences with repeated words in CSV files."""

import argparse
import csv
from pathlib import Path


def has_consecutive_repeated_words(text: str) -> list[str]:
    """Return list of words that appear consecutively."""
    words = text.lower().split()
    repeated = []
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and words[i] not in repeated:
            repeated.append(words[i])
    return repeated


def find_repeated_words_in_csv(csv_path: Path) -> None:
    """Find and print sentences with repeated words from a CSV file."""
    print(f"\n=== {csv_path.name} ===")
    found = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "")
            repeated = has_consecutive_repeated_words(text)
            if repeated:
                found += 1
                print(f"ID {row['id']}: {text}")
                print(f"  Repeated: {', '.join(repeated)}")

    print(f"Found {found} sentence(s) with repeated words")


def main():
    parser = argparse.ArgumentParser(description="Find sentences with repeated words")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="CSV files to process (default: results/test_files/*_cleaned.csv)",
    )
    args = parser.parse_args()

    if args.files:
        csv_files = args.files
    else:
        csv_files = list(Path("results/test_files").glob("*_cleaned.csv"))

    if not csv_files:
        print("No CSV files found")
        return

    for csv_file in csv_files:
        find_repeated_words_in_csv(csv_file)


if __name__ == "__main__":
    main()
