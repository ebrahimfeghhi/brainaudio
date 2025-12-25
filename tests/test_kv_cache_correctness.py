"""Test to verify KV caching produces identical outputs to the original implementation."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from brainaudio.inference.decoder.neural_lm_fusion import HuggingFaceLMFusion
import time
import json

MODEL_NAME = "meta-llama/Llama-3.2-3B"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# Test cases: (contexts, candidate_words)
TEST_CASES = [
    # Single context, multiple candidates (homophones + capitalization)
    (
        ["He is also a member of the"],
        [["royal", "Royal", "real", "Real", "reel", "Reel"]]
    ),
    # Multiple identical contexts (should benefit from KV cache reuse)
    (
        ["The quick brown fox", "The quick brown fox"],
        [["jumps", "Jumps"], ["runs", "Runs"]]
    ),
    # Empty context (sentence start)
    (
        [""],
        [["The", "the", "A", "a"]]
    ),
    # Longer context
    (
        ["I went to the store to buy some groceries and then I came back home to cook"],
        [["dinner", "Dinner", "lunch", "Lunch"]]
    ),
    # Multiple different contexts
    (
        ["She said", "He replied", "They answered"],
        [["hello", "Hello"], ["goodbye", "Goodbye"], ["maybe", "Maybe"]]
    ),
]


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


def run_tests(lm_fusion):
    """Run all test cases and return results."""
    results = []

    for i, (contexts, candidates) in enumerate(TEST_CASES):
        start = time.perf_counter()
        scores = lm_fusion.score_continuations(contexts, candidates)
        elapsed = time.perf_counter() - start

        results.append({
            "test_case": i,
            "contexts": contexts,
            "candidates": candidates,
            "scores": scores,
            "time_ms": elapsed * 1000
        })

        print(f"\nTest case {i}:")
        for ctx, cands, sc in zip(contexts, candidates, scores):
            print(f"  Context: \"{ctx[:50]}...\"" if len(ctx) > 50 else f"  Context: \"{ctx}\"")
            for word, score in zip(cands, sc):
                print(f"    {word:15s}: {score:.6f}")
        print(f"  Time: {elapsed*1000:.1f}ms")

    return results


def compare_results(original, kv_cached, tolerance=1e-4):
    """Compare two sets of results and report differences."""
    all_match = True

    for orig, kv in zip(original, kv_cached):
        for ctx_idx, (orig_scores, kv_scores) in enumerate(zip(orig["scores"], kv["scores"])):
            for word_idx, (o, k) in enumerate(zip(orig_scores, kv_scores)):
                diff = abs(o - k)
                if diff > tolerance:
                    print(f"MISMATCH in test {orig['test_case']}, ctx {ctx_idx}, word {word_idx}:")
                    print(f"  Original: {o:.6f}, KV Cached: {k:.6f}, Diff: {diff:.6f}")
                    all_match = False

    return all_match


if __name__ == "__main__":
    model, tokenizer = load_model()
    lm_fusion = HuggingFaceLMFusion(
        model, tokenizer,
        weight=1.0,
        word_insertion_bonus=0.0,
        device=None
    )

    print("\n" + "="*60)
    print("Running tests with CURRENT implementation")
    print("="*60)

    results = run_tests(lm_fusion)

    # Save results for later comparison
    total_time = sum(r["time_ms"] for r in results)
    print(f"\n{'='*60}")
    print(f"Total time: {total_time:.1f}ms")
    print(f"{'='*60}")

    # Save scores to file for comparison
    scores_only = [[r["scores"] for r in results]]
    with open("/tmp/original_scores.json", "w") as f:
        json.dump(scores_only, f)
    print("\nScores saved to /tmp/original_scores.json")
