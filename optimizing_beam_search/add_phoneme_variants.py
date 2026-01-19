"""
Add pronunciation variants to the phoneme vocab file.
For words with multiple CMU dict variants, append lines for variants (2), (3), etc.
"""

import re
import nltk
nltk.download('cmudict', quiet=True)
from nltk.corpus import cmudict

# Load the CMU dictionary
d = cmudict.dict()

def process_phoneme(p):
    """Remove stress markers and return phoneme if valid, else None."""
    p = re.sub(r'[0-9]', '', p)  # remove stress
    if re.match(r'[A-Z]+', p):   # keep only phoneme labels
        return p
    return None

def get_variants(word):
    """Returns list of pronunciation variants for a word."""
    return d.get(word.lower(), [])

def format_phonemes(variant):
    """Convert CMU variant to space-separated phoneme string."""
    phonemes = []
    for p in variant:
        processed = process_phoneme(p)
        if processed:
            phonemes.append(processed)
    return ' '.join(phonemes)

def main():
    input_path = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt"
    output_path = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme_with_variants.txt"

    # Read existing phoneme vocab
    with open(input_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    # Track stats
    words_with_variants = 0
    total_variants_added = 0

    # Process and collect new variant lines
    new_lines = []
    for line in lines:
        new_lines.append(line)  # Keep original line

        # Parse word from line (format: "word PH1 PH2 ... |")
        parts = line.split()
        if not parts:
            continue
        word = parts[0]

        # Get CMU variants
        variants = get_variants(word)

        # If multiple variants, add lines for unique variant 2, 3, etc.
        if len(variants) > 1:
            # Get phonemes from original file (between word and |)
            original_phonemes = ' '.join(parts[1:-1]) if len(parts) > 2 else ''

            seen_phonemes = set()
            seen_phonemes.add(original_phonemes)  # What's already in the file

            unique_variants = []
            for variant in variants:
                phoneme_str = format_phonemes(variant)
                if phoneme_str not in seen_phonemes:
                    seen_phonemes.add(phoneme_str)
                    unique_variants.append(phoneme_str)

            if unique_variants:
                words_with_variants += 1
                for i, phoneme_str in enumerate(unique_variants, start=2):
                    new_line = f"{word}({i}) {phoneme_str} |"
                    new_lines.append(new_line)
                    total_variants_added += 1

    # Write output
    with open(output_path, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')

    print(f"Original lines: {len(lines)}")
    print(f"Words with multiple variants: {words_with_variants}")
    print(f"Variant lines added: {total_variants_added}")
    print(f"Total lines in output: {len(new_lines)}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
