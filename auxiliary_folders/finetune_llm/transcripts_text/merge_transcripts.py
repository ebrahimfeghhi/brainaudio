#!/usr/bin/env python3
"""
Merge transcript files and remove duplicate sentences.
Preserves train/validation split structure.
"""

import argparse
from pathlib import Path


def load_transcripts(file_path: str) -> tuple[list[str], list[str]]:
    """Load transcripts, split by VALIDATION marker."""
    train = []
    val = []
    in_val = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#') and 'VALIDATION' in line.upper():
                in_val = True
                continue
            if line.startswith('#'):
                continue
            if in_val:
                val.append(line)
            else:
                train.append(line)

    return train, val


def main():
    parser = argparse.ArgumentParser(description="Merge transcript files, removing duplicates")
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=[
            "/home/ebrahim/brainaudio/data/transcripts_all_24.txt",
            "/home/ebrahim/brainaudio/data/transcripts_all_25.txt",
        ],
    )
    parser.add_argument(
        "--output-file",
        default="/home/ebrahim/brainaudio/data/transcripts_merged.txt",
    )
    args = parser.parse_args()

    all_train = []
    all_val = []

    for f in args.input_files:
        print(f"Loading: {f}")
        t, v = load_transcripts(f)
        print(f"  Train: {len(t)}, Val: {len(v)}")
        all_train.extend(t)
        all_val.extend(v)

    print(f"\nBefore deduplication:")
    print(f"  Train: {len(all_train)}")
    print(f"  Val: {len(all_val)}")

    # Deduplicate while preserving order
    seen = set()
    unique_train = []
    for s in all_train:
        if s not in seen:
            seen.add(s)
            unique_train.append(s)

    unique_val = []
    for s in all_val:
        if s not in seen:  # Also exclude if already in train
            seen.add(s)
            unique_val.append(s)

    print(f"\nAfter deduplication:")
    print(f"  Train: {len(unique_train)}")
    print(f"  Val: {len(unique_val)}")
    print(f"  Total unique: {len(unique_train) + len(unique_val)}")

    # Write merged file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for s in unique_train:
            f.write(s + '\n')
        f.write('\n# VALIDATION\n\n')
        for s in unique_val:
            f.write(s + '\n')

    print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()
