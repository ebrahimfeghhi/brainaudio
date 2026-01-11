import pickle
from transformers import AutoTokenizer


def load_sentences(pkl_path: str) -> list[str]:
    """Load sentences from a pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Handle different data formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # If it's a dict, try to extract sentences from values
        sentences = []
        for v in data.values():
            if isinstance(v, str):
                sentences.append(v)
            elif isinstance(v, list):
                sentences.extend([s for s in v if isinstance(s, str)])
        return sentences
    else:
        raise ValueError(f"Unexpected data type in pickle: {type(data)}")


def test_sentence(tokenizer, sentence_text: str) -> tuple[bool, str]:
    """
    Test tokenization consistency for a single sentence.
    Returns (passed, message).
    """
    # Split into words and ensure each has a leading space
    words = [" " + w for w in sentence_text.split()]

    # Reconstruct the full string
    full_text = "".join(words)

    # Method A: Tokenize the entire string at once
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    # Method B: Tokenize word by word and concatenate
    streamed_tokens = []
    for word in words:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        streamed_tokens.extend(word_ids)

    if full_tokens == streamed_tokens:
        return True, ""
    else:
        # Find the first mismatch for debugging
        min_len = min(len(full_tokens), len(streamed_tokens))
        for i in range(min_len):
            if full_tokens[i] != streamed_tokens[i]:
                msg = (f"First mismatch at index {i}: "
                       f"Full={full_tokens[i]} ({tokenizer.decode([full_tokens[i]])}), "
                       f"Streamed={streamed_tokens[i]} ({tokenizer.decode([streamed_tokens[i]])})")
                return False, msg

        msg = f"Length mismatch: Full={len(full_tokens)}, Streamed={len(streamed_tokens)}"
        return False, msg


def test_tokenization_consistency():
    # 1. Load the tokenizer
    model_id = "fla-hub/rwkv7-0.1B-g1"
    print(f"Loading tokenizer from {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 2. Load sentences from pickle file
    pkl_path = "/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl"
    print(f"Loading sentences from {pkl_path}...")
    try:
        sentences = load_sentences(pkl_path)
    except Exception as e:
        print(f"Error loading sentences: {e}")
        return

    print(f"Loaded {len(sentences)} sentences\n")

    # 3. Test each sentence
    passed = 0
    failed = 0
    failed_examples = []

    for i, sentence in enumerate(sentences):
        success, msg = test_sentence(tokenizer, sentence)
        if success:
            passed += 1
        else:
            failed += 1
            if len(failed_examples) < 5:  # Store first 5 failures for display
                failed_examples.append((sentence, msg))

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(sentences)} sentences...")

    # 4. Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total sentences: {len(sentences)}")
    print(f"Passed: {passed} ({100*passed/len(sentences):.1f}%)")
    print(f"Failed: {failed} ({100*failed/len(sentences):.1f}%)")

    if failed > 0:
        print(f"\n{'='*60}")
        print("FAILED EXAMPLES (first 5):")
        print(f"{'='*60}")
        for sentence, msg in failed_examples:
            print(f"\nSentence: '{sentence}'")
            print(f"  {msg}")

    if failed == 0:
        print("\n SUCCESS: All sentences passed tokenization consistency test!")
    else:
        print(f"\n MISMATCH: {failed} sentences have inconsistent tokenization.")


if __name__ == "__main__":
    test_tokenization_consistency()
