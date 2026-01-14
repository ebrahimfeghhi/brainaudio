import torch
import time
from transformers import AutoModelForCausalLM
import gc
import os

# === CONFIGURATION ===
MODEL_NAME = "meta-llama/Llama-3.2-1B" 
BATCH_SIZE = 300
SEQ_LEN = 30  # Increased to 30
DTYPE = torch.bfloat16

# Check if cuda:1 exists, otherwise fallback to cuda:0
if torch.cuda.device_count() > 1:
    DEVICE = "cuda:1"
else:
    print("Warning: 'cuda:1' not found. Defaulting to 'cuda:0'")
    DEVICE = "cuda:0"

def print_memory(label):
    torch.cuda.synchronize(DEVICE)
    allocated = torch.cuda.memory_allocated(DEVICE) / 1024**3
    reserved = torch.cuda.memory_reserved(DEVICE) / 1024**3
    print(f"[{label}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

def benchmark():
    print(f"--- Benchmarking {MODEL_NAME} ---")
    print(f"Batch: {BATCH_SIZE} | Seq Len: {SEQ_LEN} | Dtype: {DTYPE} | Device: {DEVICE}")
    
    # 1. Load Model
    # CRITICAL FIX: Removed device_map to avoid accelerate bugs.
    # We load to CPU first (default) or "meta" then move to GPU manually.
    try:
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=DTYPE, # Reverted to standard arg name for safety, change if strict
            attn_implementation="sdpa" 
        )
        model.to(DEVICE) # Explicit move ensures GPU usage
    except OSError:
        print(f"Could not load {MODEL_NAME}. Using fallback...")
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            torch_dtype=DTYPE
        )
        model.to(DEVICE)

    model.eval()
    print_memory("Model Loaded")

    # 2. Prepare Input
    input_ids = torch.randint(
        0, model.config.vocab_size, 
        (BATCH_SIZE, SEQ_LEN), 
        device=DEVICE
    )

    # 3. Warmup
    print("\nWarming up...")
    with torch.no_grad():
        _ = model(input_ids)
    torch.cuda.synchronize(DEVICE)
    
    torch.cuda.reset_peak_memory_stats(DEVICE)
    gc.collect()
    torch.cuda.empty_cache()
    
    # 4. Run Benchmark
    print("\nRunning Forward Pass...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    with torch.no_grad():
        # A. Forward Pass
        # logits_to_keep=1 OPTIMIZATION:
        # This tells HuggingFace to ONLY compute the logits for the last token.
        # This saves massive VRAM and Compute for Batch 900.
        try:
            outputs = model(input_ids, use_cache=True, logits_to_keep=1)
            next_token_logits = outputs.logits[:, -1, :].clone()
        except TypeError:
            # Fallback for older transformers versions without logits_to_keep
            outputs = model(input_ids, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :].clone()
        
        del outputs
    
    end_event.record()
    torch.cuda.synchronize(DEVICE)
    
    elapsed_ms = start_event.elapsed_time(end_event)
    
    # 5. Results
    print("-" * 30)
    print(f"Time per forward pass: {elapsed_ms:.2f} ms")
    print("-" * 30)
    
    peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
    print(f"Peak VRAM used: {peak_mem:.2f} GB")
    
    del next_token_logits
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    benchmark()