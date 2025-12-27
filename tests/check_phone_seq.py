from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

DEFAULT_LEXICON = Path("/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt")

# =============================================================================
# Define your phoneme sequences to check here:
# =============================================================================
QUERIES: List[List[str]] = [
    ["P", "R", "AH", "F", "EH", "S", "ER", "|"]
]
# =============================================================================


def load_lexicon(path: Path) -> Tuple[Dict[Tuple[str, ...], Set[str]], Dict[str, List[str]]]:
    """
    Load lexicon and build mappings.

    Returns:
        phonemes_to_words: Maps phoneme tuple -> set of words with that pronunciation
        word_to_phonemes: Maps word -> list of phonemes
    """
    phonemes_to_words: Dict[Tuple[str, ...], Set[str]] = {}
    word_to_phonemes: Dict[str, List[str]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            phonemes = parts[1:]

            # Store word -> phonemes
            word_to_phonemes[word] = phonemes

            # Store phonemes -> words (for exact match lookup)
            key = tuple(phonemes)
            if key not in phonemes_to_words:
                phonemes_to_words[key] = set()
            phonemes_to_words[key].add(word)

    return phonemes_to_words, word_to_phonemes


def find_words_for_phonemes(phonemes: List[str], phonemes_to_words: Dict[Tuple[str, ...], Set[str]]) -> Set[str]:
    """Find all words that have exactly this phoneme sequence."""
    key = tuple(phonemes)
    return phonemes_to_words.get(key, set())


def find_prefix_matches(prefix: List[str], phonemes_to_words: Dict[Tuple[str, ...], Set[str]]) -> Set[str]:
    """Find all words whose phoneme sequence starts with this prefix."""
    prefix_tuple = tuple(prefix)
    prefix_len = len(prefix_tuple)
    matches = set()

    for phoneme_seq, words in phonemes_to_words.items():
        if phoneme_seq[:prefix_len] == prefix_tuple:
            matches.update(words)

    return matches


if __name__ == "__main__":
    if not DEFAULT_LEXICON.exists():
        raise FileNotFoundError(f"Lexicon file not found: {DEFAULT_LEXICON}")

    print(f"Loading lexicon from {DEFAULT_LEXICON}...")
    phonemes_to_words, word_to_phonemes = load_lexicon(DEFAULT_LEXICON)
    print(f"Loaded {len(word_to_phonemes)} words\n")

    for query in QUERIES:
        phoneme_str = " ".join(query)

        # Exact matches
        exact_words = find_words_for_phonemes(query, phonemes_to_words)

        # Prefix matches (words that start with this sequence)
        prefix_words = find_prefix_matches(query, phonemes_to_words)

        print(f"Query: {phoneme_str}")
        print(f"  Exact matches: {exact_words if exact_words else '{none}'}")
        print(f"  Prefix matches ({len(prefix_words)}): {sorted(prefix_words)[:20]}{'...' if len(prefix_words) > 20 else ''}")
        print()