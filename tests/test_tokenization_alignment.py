"""
Test script to verify tokenization alignment in HuggingFaceLMFusion.

Verifies that the token IDs selected for scoring actually correspond to the candidate word.
"""

import torch
from transformers import AutoTokenizer
import truecase


def get_word_tokens_using_our_logic(tokenizer, context: str, word: str):
    """
    Replicate the logic from score_continuations to get the word tokens.

    Returns:
        tuple: (full_text_truecased, start_idx, word_token_ids, decoded_word)
    """
    # Step 1: Construct full text (same logic as in score_continuations)
    if not context:
        full_text = word
    elif context.endswith(" ") or word.startswith(" "):
        full_text = f"{context}{word}"
    else:
        full_text = f"{context} {word}"

    # Step 2: Apply truecase to full text
    full_text_truecased = truecase.get_true_case(full_text)

    # Step 3: Extract truecased context (everything before the last word)
    truecased_context = full_text_truecased.rsplit(" ", 1)[0] if " " in full_text_truecased else ""

    # Step 4: Compute start_idx from truecased context
    prefix_ids = tokenizer.encode(truecased_context, add_special_tokens=True)
    start_idx = len(prefix_ids) - 1

    # Step 5: Tokenize full text and extract word tokens
    full_ids = tokenizer.encode(full_text_truecased, add_special_tokens=True)

    # The word tokens start at start_idx + 1 (since we score from start_idx to end)
    # In the actual code, we sum log probs from start_idx to valid_seq_len
    # This means tokens at indices [start_idx, start_idx+1, ..., valid_seq_len-1]
    # But the first token at start_idx is the last context token, so word tokens are [start_idx+1:]
    word_token_ids = full_ids[start_idx + 1:]
    decoded_word = tokenizer.decode(word_token_ids).strip()

    return full_text_truecased, start_idx, word_token_ids, decoded_word, truecased_context


def test_word_token_extraction(tokenizer, context: str, word: str, verbose: bool = True):
    """
    Test that the tokens we select actually correspond to the candidate word.

    Args:
        tokenizer: HuggingFace tokenizer
        context: The context string
        word: The candidate word to add

    Returns:
        bool: True if the extracted tokens match the expected word
    """
    full_text, start_idx, word_token_ids, decoded_word, truecased_context = \
        get_word_tokens_using_our_logic(tokenizer, context, word)

    # The expected word is the last word in the truecased full text
    expected_word = full_text.rsplit(" ", 1)[-1] if " " in full_text else full_text

    # Check if decoded word matches expected
    passed = decoded_word == expected_word

    if verbose:
        print(f"\n{'='*70}")
        print(f"Context:             '{context}'")
        print(f"Candidate word:      '{word}'")
        print(f"Full text truecased: '{full_text}'")
        print(f"Truecased context:   '{truecased_context}'")
        print(f"Start idx:           {start_idx}")
        print(f"Word token IDs:      {word_token_ids}")
        print(f"Decoded word:        '{decoded_word}'")
        print(f"Expected word:       '{expected_word}'")

        if passed:
            print(f"✅ PASS: Extracted tokens correctly represent the candidate word")
        else:
            print(f"❌ FAIL: Mismatch! Decoded '{decoded_word}' != Expected '{expected_word}'")

    return passed


def run_tests():
    """Run test suite with various context and word combinations."""

    print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test cases: (context, word, description)
    test_cases = [
        # Basic cases
        ("", "hello", "Empty context, single word"),
        ("the", "cat", "Simple one-word context"),
        ("I saw the", "dog", "Multi-word context"),

        # Truecase changes capitalization of context
        ("he is", "happy", "Truecase capitalizes 'He'"),
        ("she said", "hello", "Truecase capitalizes 'She'"),

        # The original problematic case
        ("he is also a member of the", "royal", "Original case - capitalization change"),
        ("he is also a member of the", "real", "Original case variant"),

        # Proper nouns
        ("i went to", "paris", "Truecase capitalizes 'I' and possibly 'Paris'"),
        ("john said", "yes", "Name in context"),

        # Longer contexts
        ("the quick brown fox jumps over the lazy", "dog", "Long context"),

        # Words that might have tricky tokenization
        ("I love", "programming", "Multi-token word"),
        ("she is", "extraordinary", "Long multi-token word"),

        # Edge cases
        ("a", "test", "Single letter context"),
        ("hello world", "goodbye", "Two word context"),
    ]

    print(f"\nRunning {len(test_cases)} test cases...")

    results = []
    for context, word, description in test_cases:
        print(f"\n--- Test: {description} ---")
        passed = test_word_token_extraction(tokenizer, context, word, verbose=True)
        results.append((description, context, word, passed))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for r in results if r[3])
    failed_count = len(results) - passed_count

    for description, context, word, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} [{context[:20]}...] + [{word}]: {description}")

    print(f"\nTotal: {passed_count}/{len(results)} passed, {failed_count} failed")

    return failed_count == 0


if __name__ == "__main__":
    all_passed = run_tests()

    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed - review output above")
