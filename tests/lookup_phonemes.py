#!/usr/bin/env python3
"""
Interactive script to look up phoneme sequences for words in the lexicon.
Returns all pronunciation variants (e.g., live, live(2), live(3)).
"""

import re
from pathlib import Path
from collections import defaultdict

DEFAULT_LEXICON = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme_with_variants.txt"


def load_lexicon(lexicon_path: str) -> dict[str, list[tuple[str, list[str]]]]:
    """
    Load lexicon file and return a dictionary mapping base words to their variants.

    Returns:
        dict mapping base_word -> [(variant_name, [phonemes]), ...]
        e.g., "live" -> [("live", ["L", "IH", "V"]), ("live(2)", ["L", "AY", "V"])]
    """
    lexicon = defaultdict(list)

    with open(lexicon_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: "word phoneme1 phoneme2 ... |"
            parts = line.split()
            if len(parts) < 2:
                continue

            word = parts[0]
            # Remove trailing "|" if present
            phonemes = [p for p in parts[1:] if p != "|"]

            # Extract base word (remove variant number like "(2)")
            base_word = re.sub(r'\(\d+\)$', '', word).lower()

            lexicon[base_word].append((word, phonemes))

    return dict(lexicon)


def lookup_word(lexicon: dict, word: str) -> list[tuple[str, list[str]]]:
    """Look up all pronunciation variants for a word."""
    return lexicon.get(word.lower(), [])


def format_phonemes(phonemes: list[str]) -> str:
    """Format phoneme list for display."""
    return " ".join(phonemes)


def main():
    lexicon_path = DEFAULT_LEXICON

    print(f"Loading lexicon from {lexicon_path}...")
    lexicon = load_lexicon(lexicon_path)
    print(f"Loaded {len(lexicon)} unique words with {sum(len(v) for v in lexicon.values())} total variants.\n")

    print("Enter a word to look up its phoneme sequences.")
    print("Type 'quit' or 'exit' to exit.\n")

    while True:
        try:
            word = input("Word> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not word:
            continue

        if word.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        variants = lookup_word(lexicon, word)

        if not variants:
            print(f"  '{word}' not found in lexicon.\n")
            continue

        print(f"  Found {len(variants)} variant(s) for '{word}':")
        for variant_name, phonemes in variants:
            print(f"    {variant_name:<20} {format_phonemes(phonemes)}")
        print()


if __name__ == "__main__":
    main()
