#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers",
#     "sentencepiece",
# ]
# ///
"""
Interactive tool to check tokenization for capitalized vs non-capitalized words.
"""

from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-3B"


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Ready! Type a word to see tokenization (Ctrl+C to exit)\n")

    while True:
        try:
            word = input("Enter word: ").strip()
            if not word:
                continue

            lower_word = word.lower()
            capitalized_word = word.capitalize()

            lower_tokens = tokenizer.encode(lower_word, add_special_tokens=False)
            capitalized_tokens = tokenizer.encode(capitalized_word, add_special_tokens=False)

            lower_decoded = [tokenizer.decode([t]) for t in lower_tokens]
            cap_decoded = [tokenizer.decode([t]) for t in capitalized_tokens]

            print(f"  Lower '{lower_word}': {len(lower_tokens)} tokens -> {lower_decoded}")
            print(f"  Capitalized '{capitalized_word}': {len(capitalized_tokens)} tokens -> {cap_decoded}")

            diff = len(capitalized_tokens) - len(lower_tokens)
            if diff != 0:
                print(f"  Discrepancy: {diff:+d}")
            print()

        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()
