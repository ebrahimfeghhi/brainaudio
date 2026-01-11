import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_mem_mb():
    """Returns currently allocated CUDA memory in MB."""
    return torch.cuda.memory_allocated() / 1024 / 1024

def measure_smollm_cache_cost():
    # Configuration
    MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
    TOTAL_BEAMS = 5000     # Your target beam count
    SEQ_LEN = 200          # Target sequence length
    DTYPE = torch.float16  # Using FP16
    
    print(f"--- SmolLM2 KV Cache Memory Test (Beams={TOTAL_BEAMS}, Len={SEQ_LEN}) ---")
    
    # 1. Load Model
    print("1. Loading model...", end="")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(" [WARNING: No GPU found, measuring CPU RAM]")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(" Done.")
    
    baseline_mem = get_mem_mb()
    print(f"   Base Model VRAM: {baseline_mem:.2f} MB")

    # 2. Simulate Cache Growth
    print(f"\n2. Allocating Cache for {TOTAL_BEAMS} beams x {SEQ_LEN} tokens...")
    
    # Create dummy input of shape [Batch, SeqLen]
    dummy_input = torch.randint(0, 1000, (TOTAL_BEAMS, SEQ_LEN), device=device)
    
    try:
        with torch.no_grad():
            out = model(dummy_input, use_cache=True)
            kv_cache = out.past_key_values
            
            # --- CRITICAL OPTIMIZATION ---
            # Extract ONLY the last token's logits: [2000, 1, Vocab]
            last_token_logits = out.logits[:, -1, :].clone()
            
            # Delete the full output object to free the massive [2000, 50, Vocab] tensor
            del out
            torch.cuda.empty_cache()

        # 3. Measure Memory
        torch.cuda.synchronize()
        current_mem = get_mem_mb()
        cache_cost = current_mem - baseline_mem
        
        print(f"   SUCCESS!")
        print(f"   Actual KV Cache Size: {cache_cost:.2f} MB")
        
        # 4. Theoretical Calculation (Sanity Check)
        config = model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = config.hidden_size // config.num_attention_heads
        
        # Size = 2 (K+V) * Batch * SeqLen * Layers * KVHeads * HeadDim * 2 bytes
        theoretical_bytes = 2 * TOTAL_BEAMS * SEQ_LEN * num_layers * num_kv_heads * head_dim * 2
        theoretical_mb = theoretical_bytes / 1024 / 1024
        
        print(f"   Theoretical Size:     {theoretical_mb:.2f} MB")
        print(f"   Difference (Overhead): {cache_cost - theoretical_mb:.2f} MB (Likely fragmentation)")

        # 5. Simulate the "Next Token" Step (Verification)
        print(f"\n3. Simulating Next Token Generation (Step {SEQ_LEN+1})...")
        next_token = torch.randint(0, 1000, (TOTAL_BEAMS, 1), device=device)
        
        with torch.no_grad():
            out = model(next_token, past_key_values=kv_cache, use_cache=True)
            new_logits = out.logits[:, -1, :]
            
        print("   Forward pass successful.")
        
    except torch.cuda.OutOfMemoryError:
        print("   FAILED: CUDA Out Of Memory!")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    measure_smollm_cache_cost()