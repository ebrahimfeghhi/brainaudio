"""Test the _score_words_with_kv_cache method against the original implementation."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from brainaudio.inference.decoder.neural_lm_fusion import HuggingFaceLMFusion

MODEL_NAME = "meta-llama/Llama-3.2-3B"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load the model and tokenizer."""
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
    )
    return model, tokenizer


def test_kv_cache_method():
    """Test that _score_words_with_kv_cache produces same results as score_continuations."""
    model, tokenizer = load_model()
    lm_fusion = HuggingFaceLMFusion(
        model, tokenizer,
        weight=1.0,
        word_insertion_bonus=0.0,
        device=None
    )

    # Test cases: (context, words)
    # Include single-token and multi-token words
    test_cases = [
        # Single-token words
        ("He is also a member of the", ["royal", "Royal", "real", "Real"]),
        # Multi-token words (e.g., "committee" is likely multiple tokens)
        ("She is the head of the", ["committee", "Committee", "organization", "Organization"]),
        # Mix of single and multi-token
        ("The quick brown fox", ["jumps", "jumped", "jumping"]),
        # Empty context (sentence start)
        ("", ["The", "the", "Hello", "hello"]),
        # Longer context
        ("I went to the store yesterday and bought", ["apples", "Apples", "groceries", "Groceries"]),
    ]

    print("\n" + "="*70)
    print("Testing _score_words_with_kv_cache vs score_continuations")
    print("="*70)

    all_passed = True

    for context, words in test_cases:
        print(f"\nContext: \"{context}\"")
        print(f"Words: {words}")

        # Get scores using original method
        original_scores = lm_fusion.score_continuations([context], [words])[0]

        # Get scores using KV cache method
        try:
            kv_scores = lm_fusion._score_words_with_kv_cache(context, words)
        except NotImplementedError:
            print("  _score_words_with_kv_cache not yet implemented, skipping...")
            continue

        # Compare scores
        print(f"  {'Word':<20} {'Original':>12} {'KV Cache':>12} {'Diff':>12}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

        for word, orig, kv in zip(words, original_scores, kv_scores):
            diff = abs(orig - kv)
            status = "OK" if diff < 1e-4 else "MISMATCH"
            print(f"  {word:<20} {orig:>12.6f} {kv:>12.6f} {diff:>12.6f} {status}")
            if diff >= 1e-4:
                all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - scores don't match")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    test_kv_cache_method()
