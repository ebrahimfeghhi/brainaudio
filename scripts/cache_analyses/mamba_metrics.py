import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
# COMPARISON: Using Mamba2-2.7B (closest size to Llama 3.2 3B)
# You could also use "mistralai/Mamba-Codestral-7B-v0.1"
MODEL_ID ="state-spaces/mamba-2.8b-hf" 
BATCH_SIZE = 500
SEQ_LENGTH = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_memory_gb():
    """Returns current memory allocated in GB"""
    return torch.cuda.memory_allocated() / 1024**3

def run_benchmark():
    # Clear any existing cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"--- Loading Model: {MODEL_ID} ---")
    
    # 1. Load Model
    # Mamba 2 loads via AutoModelForCausalLM just like Llama
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16,
            device_map=DEVICE,
            trust_remote_code=True # Often needed for newer Mamba2 implementations
        )
    except Exception as e:
        print(f"Error loading {MODEL_ID}: {e}")
        return

    model.eval()
    
    # Measure Static Memory (Weights only)
    static_mem = get_memory_gb()
    print(f"Model Loaded. Static VRAM: {static_mem:.2f} GB")

    print(f"\n--- Preparing Batch ({BATCH_SIZE} seqs x {SEQ_LENGTH} tokens) ---")
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LENGTH)).to(DEVICE)

    # Warmup
    print("Warming up CUDA kernels...")
    with torch.no_grad():
        _ = model(input_ids[:, 0:1], use_cache=True)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    
    print("Starting Sequential Scoring (SSM State enabled)...")
    start_time = time.time()
    
    # In Mamba, 'past_key_values' holds the RNN-like state (SSM state).
    # CRITICAL DIFFERENCE: This object does NOT grow. It stays fixed size.
    past_key_values = None
    
    with torch.no_grad():
        for t in range(SEQ_LENGTH):
            # Select current token
            current_input = input_ids[:, t].unsqueeze(1)
            
            # Forward Pass - API is identical to Transformer
            outputs = model(
                input_ids=current_input, 
                past_key_values=past_key_values, 
                use_cache=True
            )
            
            # Pass the state to the next step
            past_key_values = outputs.past_key_values
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Measure Stats
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    total_time = end_time - start_time
    throughput = (BATCH_SIZE * SEQ_LENGTH) / total_time
    
    print(f"\n--- Results ---")
    print(f"Throughput:       {throughput:.0f} tokens/second")
    print(f"Total Time:       {total_time:.4f} s")
    print(f"-----------------------------")
    print(f"Static Model VRAM: {static_mem:.2f} GB")
    print(f"Peak VRAM Usage:   {peak_mem:.2f} GB")
    print(f"Memory Growth:     {peak_mem - static_mem:.2f} GB") 
    print(f"                   (Expect ~0.00 GB growth for Mamba)")

if __name__ == "__main__":
    run_benchmark()