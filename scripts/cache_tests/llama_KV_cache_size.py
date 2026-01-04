import torch
from transformers import AutoConfig

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
BEAM_SIZE = 500
SEQ_LEN = 20  # Tokens per beam
PRECISION_BYTES = 2  # float16/bfloat16 = 2 bytes

print(f"Fetching config for {MODEL_ID}...")
try:
    config = AutoConfig.from_pretrained(MODEL_ID)
except OSError:
    print("Error: Could not fetch config from Hugging Face.")
    print("Please ensure you are logged in via 'huggingface-cli login' if the model is gated.")
    exit()

# --- 1. GET ARCHITECTURE DETAILS ---
# Llama 3.2 uses Grouped Query Attention (GQA), so we check num_key_value_heads
n_layers = config.num_hidden_layers
head_dim = config.hidden_size // config.num_attention_heads
n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)

print(f"\n--- MODEL SPECS ---")
print(f"Layers:        {n_layers}")
print(f"Head Dim:      {head_dim}")
print(f"KV Heads:      {n_kv_heads} (GQA)")

# --- 2. CALCULATE PER-TOKEN COST ---
# Formula: 2 (K+V) * Layers * KV_Heads * Head_Dim * Bytes
bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * PRECISION_BYTES

print(f"\n--- CACHE COST ---")
print(f"Cache per Token: {bytes_per_token / 1024:.2f} KB")

# --- 3. SCALING PREDICTION ---
# Worst case: All 500 beams are completely distinct (no shared prefix)
total_tokens = BEAM_SIZE * SEQ_LEN
total_cache_bytes = total_tokens * bytes_per_token
total_cache_gb = total_cache_bytes / (1024**3)

print(f"\n--- SCENARIO: 500 Beams @ Length {SEQ_LEN} ---")
print(f"Total KV Cache:  {total_cache_gb:.4f} GB")

# --- COMPARISON ---
print(f"\n--- VS RWKV (Reference) ---")
print(f"RWKV (Fixed):    ~10.00 GB")
print(f"Llama (Growing): {total_cache_gb:.4f} GB")

if total_cache_gb < 1.0:
    print("\n✅ VERDICT: Llama wins huge here (Short Sequence).")
    print("Since the sequence is short (20 tokens), the 'Growing Cache' is tiny.")
else:
    print("\n✅ VERDICT: RWKV wins.")