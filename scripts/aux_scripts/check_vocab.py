#!/usr/bin/env python3
"""Check if all words in transcripts are within the vocabulary."""

import re

def load_vocab(vocab_path):
    """Load vocabulary from file, returning a set of lowercase words."""
    vocab = set()
    with open(vocab_path, 'r') as f:
        for line in f:
            word = line.strip().lower()
            if word:
                vocab.add(word)
    return vocab

def extract_words(text):
    """Extract words from text, removing punctuation."""
    # Remove punctuation and split into words
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
    return words

def main():
    transcript_path = "/home/ebrahim/brainaudio/data/transcripts_all_25_val.txt"
    vocab_path = "/data2/brain2text/lm/vocab_lower_100k.txt"

    # Load vocabulary
    vocab = load_vocab(vocab_path)
    print(f"Loaded {len(vocab)} words from vocabulary")

    # Track out-of-vocabulary words
    oov_words = set()
    oov_occurrences = {}

    with open(transcript_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            words = extract_words(line)
            for word in words:
                word_lower = word.lower()
                if word_lower not in vocab:
                    oov_words.add(word_lower)
                    if word_lower not in oov_occurrences:
                        oov_occurrences[word_lower] = []
                    oov_occurrences[word_lower].append((line_num, line.strip()))

    # Report results
    if oov_words:
        print(f"\nFound {len(oov_words)} out-of-vocabulary words:\n")
        for word in sorted(oov_words):
            print(f"  '{word}' (appears on {len(oov_occurrences[word])} line(s))")
            # Show first occurrence
            line_num, line_text = oov_occurrences[word][0]
            print(f"      Example (line {line_num}): {line_text[:80]}...")
    else:
        print("\nAll words are within the vocabulary!")

if __name__ == "__main__":
    main()
