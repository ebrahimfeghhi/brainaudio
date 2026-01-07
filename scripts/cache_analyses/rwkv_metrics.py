import time
import torch
from transformers import AutoModelForCausalLM

# --- Configuration ---
MODEL_ID = "fla-hub/rwkv7-0.1B-g1"
BATCH_SIZE = 500
SEQ_LENGTH = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_memory_gb():
    """Returns current memory allocated in GB"""
    return torch.cuda.memory_allocated() / 1024**3

def run_benchmark():
    # Clear any existing cache to get accurate readings
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"--- Loading Model: {MODEL_ID} ---")
    
    # 1. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    
    # Measure Static Memory (Weights only)
    static_mem = get_memory_gb()
    print(f"Model Loaded. Static VRAM: {static_mem:.2f} GB")

    print(f"\n--- Preparing Batch ({BATCH_SIZE} seqs x {SEQ_LENGTH} tokens) ---")
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LENGTH)).to(DEVICE)

    # Warmup
    with torch.no_grad():
        _ = model(input_ids[:, 0:1], use_cache=True)
    torch.cuda.synchronize()
    
    # Reset peak stats specifically to capture the INFERENCE peak
    torch.cuda.reset_peak_memory_stats()
    
    print("Starting Sequential Scoring...")
    start_time = time.time()
    
    state = None
    
    with torch.no_grad():
        for t in range(SEQ_LENGTH):
            current_input = input_ids[:, t].unsqueeze(1)
            
            outputs = model(
                input_ids=current_input, 
                past_key_values=state, 
                use_cache=True
            )
            
            state = outputs.past_key_values
            
            # (Optional) access logits to ensure computation happens
            _ = outputs.logits

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
    print(f"Memory Growth:     {peak_mem - static_mem:.2f} GB (due to Batch States)")

if __name__ == "__main__":
    run_benchmark()