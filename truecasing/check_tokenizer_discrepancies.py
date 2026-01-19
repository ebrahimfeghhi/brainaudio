#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers",
#     "sentencepiece",
# ]
# ///
"""
Check for tokenization discrepancies between capitalized and non-capitalized words
using the LLaMA 3.2 3B tokenizer.
"""

from transformers import AutoTokenizer

VOCAB_FILE = "/data2/brain2text/lm/vocab_lower_100k.txt"
MODEL_ID = "meta-llama/Llama-3.2-3B"


def load_vocab(vocab_path: str) -> list[str]:
    """Load words from vocabulary file."""
    words = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                words.append(word)
    return words


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading vocabulary from: {VOCAB_FILE}")
    words = load_vocab(VOCAB_FILE)
    print(f"Loaded {len(words):,} words")

    discrepancies = []

    for word in words:
        lower_word = word.lower()
        capitalized_word = word.capitalize()

        lower_tokens = tokenizer.encode(lower_word, add_special_tokens=False)
        capitalized_tokens = tokenizer.encode(capitalized_word, add_special_tokens=False)

        lower_count = len(lower_tokens)
        capitalized_count = len(capitalized_tokens)

        if lower_count != capitalized_count:
            discrepancy = capitalized_count - lower_count
            discrepancies.append({
                "word": word,
                "lower": lower_word,
                "capitalized": capitalized_word,
                "lower_tokens": lower_count,
                "capitalized_tokens": capitalized_count,
                "discrepancy": discrepancy,
                "lower_token_ids": lower_tokens,
                "capitalized_token_ids": capitalized_tokens,
            })

    print(f"\n{'='*60}")
    print(f"RESULTS: Found {len(discrepancies)} words with discrepancies")
    print(f"{'='*60}\n")

    # Sort by absolute discrepancy (largest first)
    discrepancies.sort(key=lambda x: abs(x["discrepancy"]), reverse=True)

    for d in discrepancies:
        lower_decoded = [tokenizer.decode([t]) for t in d["lower_token_ids"]]
        cap_decoded = [tokenizer.decode([t]) for t in d["capitalized_token_ids"]]

        print(f"Word: '{d['word']}'")
        print(f"  Lower '{d['lower']}': {d['lower_tokens']} tokens -> {lower_decoded}")
        print(f"  Capitalized '{d['capitalized']}': {d['capitalized_tokens']} tokens -> {cap_decoded}")
        print(f"  Discrepancy: {d['discrepancy']:+d}")
        print()

    # Summary statistics
    if discrepancies:
        positive = sum(1 for d in discrepancies if d["discrepancy"] > 0)
        negative = sum(1 for d in discrepancies if d["discrepancy"] < 0)
        print(f"{'='*60}")
        print("SUMMARY:")
        print(f"  Total discrepancies: {len(discrepancies)}")
        print(f"  Capitalized has MORE tokens: {positive}")
        print(f"  Capitalized has FEWER tokens: {negative}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
