import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_ID = "meta-llama/Llama-3.2-1B" 
BATCH_SIZE = 1000
# We will let the tokenizer decide the length based on the text
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Loading {MODEL_ID} on {DEVICE.upper()}...")
    
    # 1. Load Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # Load in float16 for GPU, or float32 for CPU (CPU doesn't always love fp16)
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Prepare Data
    text = "The quick brown fox jumps over the lazy dog repeatedly."
    # Don't hardcode length. Let the tokenizer tell us how long the sentence is.
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    # Repeat for batch size
    input_ids = input_ids.repeat(BATCH_SIZE, 1).to(DEVICE)
    
    # DYNAMIC LENGTH: Use the actual shape of the tensor!
    SEQ_LEN = input_ids.shape[1] 
    
    print(f"\nProcessing Batch Size: {BATCH_SIZE}, Actual Seq Length: {SEQ_LEN}")
    print("-" * 50)

    # --- CASE 1: WITHOUT KV CACHE ---
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    # Iterate from 1 because we need context to predict the next token
    for i in range(1, SEQ_LEN):
        # Slicing from 0 to i forces re-computation of the whole prefix
        current_input = input_ids[:, :i] 
        with torch.no_grad():
            outputs = model(current_input, use_cache=False)
            # We don't need to store logits, just burn the compute

    no_cache_time = time.time() - start_time
    # Get memory only if on CUDA
    no_cache_mem = torch.cuda.max_memory_allocated() / 1024**3 if DEVICE == "cuda" else 0.0
    
    print(f"NO CACHE Results:")
    print(f"  Time: {no_cache_time:.4f} seconds")
    if DEVICE == "cuda":
        print(f"  Peak VRAM: {no_cache_mem:.4f} GB")
    else:
        print(f"  Peak VRAM: N/A (Running on CPU)")

    # --- CASE 2: WITH KV CACHE ---
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    past_key_values = None
    # Start with the first token (Index 0)
    current_input_token = input_ids[:, 0].unsqueeze(1) 
    
    for i in range(1, SEQ_LEN):
        with torch.no_grad():
            # Pass only the NEW token + history cache
            outputs = model(current_input_token, past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            
            # Prepare the next token for the NEXT iteration
            # We use 'i' here because that's the token we just predicted (or are forcing)
            # CRITICAL FIX: Ensure we don't go out of bounds
            if i < SEQ_LEN: 
                 current_input_token = input_ids[:, i].unsqueeze(1)

    cache_time = time.time() - start_time
    cache_mem = torch.cuda.max_memory_allocated() / 1024**3 if DEVICE == "cuda" else 0.0

    print(f"\nWITH CACHE Results:")
    print(f"  Time: {cache_time:.4f} seconds")
    if DEVICE == "cuda":
        print(f"  Peak VRAM: {cache_mem:.4f} GB")
    else:
        print(f"  Peak VRAM: N/A (Running on CPU)")

    # --- COMPARISON ---
    print("-" * 50)
    print(f"Speedup: {no_cache_time / cache_time:.2f}x faster with Cache")

if __name__ == "__main__":
    main()