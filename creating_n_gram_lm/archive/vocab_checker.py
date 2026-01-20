"""Utility to check if words are in the vocabulary."""

VOCAB_FILE = "/data2/brain2text/lm/vocab_lower_100k.txt"

_vocab_cache = None

def load_vocab(vocab_path=VOCAB_FILE):
    """Load vocabulary into a set for O(1) lookup."""
    global _vocab_cache
    if _vocab_cache is not None:
        return _vocab_cache
    
    vocab = set()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word:
                vocab.add(word)
    _vocab_cache = vocab
    print(f"Loaded {len(vocab):,} words from {vocab_path}")
    return vocab

def is_in_vocab(word, vocab=None):
    """Check if a word is in the vocabulary (case-insensitive)."""
    if vocab is None:
        vocab = load_vocab()
    return word in vocab

def check_words(words, vocab=None):
    """Check multiple words, return dict of word -> in_vocab."""
    if vocab is None:
        vocab = load_vocab()
    return {w: w.lower() in vocab for w in words}


if __name__ == "__main__":
    # Quick test
    vocab = load_vocab()
    
    test_words = ["hello", "world", "xyzabc", "the", "Apple"]
    for word in test_words:
        status = "✓" if is_in_vocab(word, vocab) else "✗"
        print(f"  {status} {word}")
