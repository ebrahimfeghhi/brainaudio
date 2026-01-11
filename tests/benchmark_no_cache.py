"""
Benchmark RWKV vs SmolLM2 WITHOUT caching.
Simulates beam search behavior where full context is re-encoded each time.
"""
import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM

# --- CONFIGURATION ---
BATCH_SIZE = 128      # Typical beam search might have batch * beams * homophones
CONTEXT_LEN = 20      # Tokens already decoded (e.g., "I saw my aunt at the")
WORD_LEN = 4          # New word to score (e.g., "unbelievable")
NUM_ITERS = 100
WARMUP = 10

# Define Models
MODELS = {
    "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M",
    "RWKV7-0.1B": "fla-hub/rwkv7-0.1B-g1"
}

def benchmark_model_no_cache(name, repo_id):
    """
    Benchmark without caching - re-encode full context + word each time.
    This simulates the current beam search behavior.
    """
    print(f"\n--- Benchmarking {name} (NO CACHE) ---")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).cuda()
        model.eval()

        # Full sequence = context + word
        total_len = CONTEXT_LEN + WORD_LEN
        input_ids = torch.randint(0, 1000, (BATCH_SIZE, total_len)).cuda()

        # Warmup
        print(f"Warming up... (seq_len={total_len})")
        for _ in range(WARMUP):
            with torch.no_grad():
                _ = model(input_ids)
        torch.cuda.synchronize()

        # Benchmark - NO CACHING, full re-encode each time
        print(f"Running {NUM_ITERS} iterations (full re-encode)...")
        times = []

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(NUM_ITERS):
                start_event.record()

                # Re-encode FULL sequence (context + word) from scratch
                # This is what score_continuations currently does
                _ = model(input_ids, use_cache=False)

                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"Results for {name} (NO CACHE):")
        print(f"  Seq Length: {total_len} (context={CONTEXT_LEN}, word={WORD_LEN})")
        print(f"  Batch Size: {BATCH_SIZE}")
        print(f"  Avg Time: {avg_time:.4f} ms")
        print(f"  Std Dev:  {std_time:.4f} ms")

        del model
        torch.cuda.empty_cache()

        return avg_time

    except Exception as e:
        print(f"Failed to benchmark {name}: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')


def benchmark_model_with_cache(name, repo_id):
    """
    Benchmark WITH caching - encode context once, then only encode word.
    This is what we SHOULD be doing in beam search.
    """
    print(f"\n--- Benchmarking {name} (WITH CACHE) ---")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).cuda()
        model.eval()

        # Separate context and word
        context_ids = torch.randint(0, 1000, (BATCH_SIZE, CONTEXT_LEN)).cuda()
        word_ids = torch.randint(0, 1000, (BATCH_SIZE, WORD_LEN)).cuda()

        # Warmup - encode context and get cache
        print(f"Warming up... (context={CONTEXT_LEN}, word={WORD_LEN})")
        for _ in range(WARMUP):
            with torch.no_grad():
                ctx_out = model(context_ids, use_cache=True)
                _ = model(word_ids, past_key_values=ctx_out.past_key_values, use_cache=True)
        torch.cuda.synchronize()

        # Pre-compute context cache (done once in real beam search)
        with torch.no_grad():
            ctx_out = model(context_ids, use_cache=True)
            cached_state = ctx_out.past_key_values

        # Benchmark - only encode the WORD using cached context
        print(f"Running {NUM_ITERS} iterations (word only, using cached context)...")
        times = []

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(NUM_ITERS):
                start_event.record()

                # Only encode the NEW WORD, using cached context
                _ = model(word_ids, past_key_values=cached_state, use_cache=True)

                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"Results for {name} (WITH CACHE):")
        print(f"  Word Length: {WORD_LEN}")
        print(f"  Batch Size: {BATCH_SIZE}")
        print(f"  Avg Time: {avg_time:.4f} ms")
        print(f"  Std Dev:  {std_time:.4f} ms")

        del model
        torch.cuda.empty_cache()

        return avg_time

    except Exception as e:
        print(f"Failed to benchmark {name}: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')


# --- MAIN ---
if __name__ == "__main__":
    print("="*60)
    print("BENCHMARK: NO CACHE (current beam search behavior)")
    print("="*60)

    no_cache_results = {}
    for name, repo in MODELS.items():
        no_cache_results[name] = benchmark_model_no_cache(name, repo)

    print("\n" + "="*60)
    print("BENCHMARK: WITH CACHE (optimal beam search behavior)")
    print("="*60)

    with_cache_results = {}
    for name, repo in MODELS.items():
        with_cache_results[name] = benchmark_model_with_cache(name, repo)

    # Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    print("\nNO CACHE (re-encode full context each time):")
    baseline = no_cache_results["SmolLM2-135M"]
    for name, t in no_cache_results.items():
        diff = ((t - baseline) / baseline) * 100
        print(f"  {name}: {t:.2f} ms ({diff:+.1f}% vs SmolLM2)")

    print("\nWITH CACHE (encode word only):")
    baseline = with_cache_results["SmolLM2-135M"]
    for name, t in with_cache_results.items():
        diff = ((t - baseline) / baseline) * 100
        print(f"  {name}: {t:.2f} ms ({diff:+.1f}% vs SmolLM2)")

    print("\nSPEEDUP FROM CACHING:")
    for name in MODELS.keys():
        speedup = no_cache_results[name] / with_cache_results[name]
        print(f"  {name}: {speedup:.2f}x faster with cache")
