"""
Count CMU dictionary pronunciation variants for each word in vocab.
"""

import nltk
nltk.download('cmudict', quiet=True)
from nltk.corpus import cmudict

# Load the CMU dictionary
d = cmudict.dict()

def get_variant_count(word):
    """Returns the number of pronunciation variants for a word (0 if not in dict)."""
    variants = d.get(word.lower(), [])
    return len(variants)

def main():
    vocab_path = "/data2/brain2text/lm/vocab_lower_100k.txt"
    output_path = "/home/ebrahim/brainaudio/optimizing_beam_search/vocab_cmu_variant_counts.txt"

    with open(vocab_path, 'r') as f:
        words = [line.strip() for line in f]

    results = []
    for word in words:
        count = get_variant_count(word)
        results.append((word, count))

    # Save results: word<tab>count
    with open(output_path, 'w') as f:
        for word, count in results:
            f.write(f"{word}\t{count}\n")

    # Print summary stats
    total = len(results)
    in_cmu = sum(1 for _, c in results if c > 0)
    not_in_cmu = total - in_cmu
    multi_variant = sum(1 for _, c in results if c > 1)

    print(f"Total words: {total}")
    print(f"In CMU dict: {in_cmu} ({100*in_cmu/total:.1f}%)")
    print(f"Not in CMU dict: {not_in_cmu} ({100*not_in_cmu/total:.1f}%)")
    print(f"Multiple variants: {multi_variant} ({100*multi_variant/total:.1f}%)")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
