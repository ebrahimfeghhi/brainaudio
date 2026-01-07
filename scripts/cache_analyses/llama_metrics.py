import time
import torch
from transformers import AutoModelForCausalLM

# --- Configuration ---
# The base (non-instruct) version of Llama 3.2 3B
MODEL_ID = "meta-llama/Llama-3.2-3B"
BATCH_SIZE = 1000
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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16,
            device_map=DEVICE,
            attn_implementation="sdpa" # Use Scaled Dot Product Attention (Fastest)
        )
    except OSError:
        print(f"Error: Could not load {MODEL_ID}.")
        print("Please ensure you have accepted the license on Hugging Face and ran 'huggingface-cli login'.")
        return

    model.eval()
    
    # Measure Static Memory (Weights only)
    static_mem = get_memory_gb()
    print(f"Model Loaded. Static VRAM: {static_mem:.2f} GB")

    print(f"\n--- Preparing Batch ({BATCH_SIZE} seqs x {SEQ_LENGTH} tokens) ---")
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LENGTH)).to(DEVICE)

    # Warmup
    # We run a tiny forward pass to wake up the CUDA kernels
    print("Warming up CUDA kernels...")
    with torch.no_grad():
        _ = model(input_ids[:, 0:1], use_cache=True)
    torch.cuda.synchronize()
    
    # Reset peak stats to capture only the INFERENCE/SCORING phase
    torch.cuda.reset_peak_memory_stats()
    
    print("Starting Sequential Scoring (KV Cache enabled)...")
    start_time = time.time()
    
    # 'past_key_values' is the Transformer equivalent of an RNN state
    # However, unlike RWKV, this 'state' GROWS with every token.
    past_key_values = None
    
    with torch.no_grad():
        for t in range(SEQ_LENGTH):
            # Select current token for all 500 sentences [Batch, 1]
            current_input = input_ids[:, t].unsqueeze(1)
            
            # Forward Pass
            outputs = model(
                input_ids=current_input, 
                past_key_values=past_key_values, 
                use_cache=True
            )
            
            # Update the KV Cache for the next step
            past_key_values = outputs.past_key_values
            
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
    print(f"Memory Growth:     {peak_mem - static_mem:.2f} GB (Weights vs Peak)")

if __name__ == "__main__":
    run_benchmark()