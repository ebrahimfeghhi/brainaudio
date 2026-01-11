"""Check for words in the lexicon that share the same character sequence."""

from collections import defaultdict
from pathlib import Path

LEXICON_PATH = "/data2/brain2text/lm/char_lm/lexicon_char_cleaned.txt"


def main():
    # Map: character sequence -> list of words with that sequence
    char_seq_to_words = defaultdict(list)

    with open(LEXICON_PATH) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            word = parts[0]
            # Character sequence is everything after the word, excluding the final "|"
            char_seq = tuple(p for p in parts[1:] if p != "|")
            char_seq_to_words[char_seq].append(word)

    # Find sequences with multiple words
    duplicates = {seq: words for seq, words in char_seq_to_words.items() if len(words) > 1}

    print(f"Total unique character sequences: {len(char_seq_to_words)}")
    print(f"Sequences with multiple words: {len(duplicates)}")
    print()

    if duplicates:
        # Sort by number of words sharing the sequence (descending)
        sorted_dups = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)

        print("=== Words sharing the same character sequence ===")
        for char_seq, words in sorted_dups:
            seq_str = " ".join(char_seq)
            print(f"[{seq_str}] -> {words}")
    else:
        print("No duplicate character sequences found.")


if __name__ == "__main__":
    main()
