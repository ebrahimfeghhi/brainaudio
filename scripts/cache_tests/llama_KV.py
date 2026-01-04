import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def measure_exact_vram_with_logits():
    print("--- 1. LOADING MODEL ---")
    model_id = "meta-llama/Llama-3.2-3B"
    
    # Config: 1000 Beams
    NUM_BEAMS = 5000
    
    # Load in bfloat16 (Standard for Llama 3)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
    )

    # Measure Model Weights (Ground Truth)
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model Weights: {model_bytes / (1024**3):.2f} GB")

    print("\n--- 2. RUNNING 20 TOKENS ---")
    text = "The quick brown fox jumps over the lazy dog and then decides to learn about transformers."
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    inputs['input_ids'] = inputs['input_ids'][:, :20] # Force 20 tokens
    
    # Run Forward Pass
    with torch.no_grad():
        outputs = model(inputs['input_ids'], use_cache=True)
    
    print("\n--- 3. MEASURING EXACT CACHE ---")
    # A. Measure KV Cache (Per Beam)
    past_key_values = outputs.past_key_values
    breakpoint()
    kv_bytes_one_beam = 0
    for layer in past_key_values:
        key_tensor, value_tensor = layer
        kv_bytes_one_beam += key_tensor.numel() * key_tensor.element_size()
        kv_bytes_one_beam += value_tensor.numel() * value_tensor.element_size()

    # B. Measure Logits Cache (Per Beam)
    # We store the logits for the LAST token of the beam to score the next step instantly.
    # Shape: [1, Vocab_Size] per beam
    vocab_size = model.config.vocab_size
    bytes_per_logit = 2 # float16/bfloat16
    logits_bytes_one_beam = vocab_size * bytes_per_logit
    
    print(f"Vocab Size: {vocab_size}")
    print(f"Logits Size per Beam: {logits_bytes_one_beam / 1024:.2f} KB")

    # --- SCALING TO 1000 BEAMS ---
    total_kv_gb = (kv_bytes_one_beam * NUM_BEAMS) / (1024**3)
    total_logits_gb = (logits_bytes_one_beam * NUM_BEAMS) / (1024**3)
    
    # 0.7 GB overhead for CUDA context/kernels
    overhead_gb = 0.7
    total_vram_gb = (model_bytes / (1024**3)) + total_kv_gb + total_logits_gb + overhead_gb
    
    print(f"\n--- FINAL VERDICT ({NUM_BEAMS} Beams) ---")
    print(f"1. Model Weights:  {model_bytes / (1024**3):.2f} GB")
    print(f"2. KV Cache:       {total_kv_gb:.2f} GB  (History for 20 tokens)")
    print(f"3. Logits Cache:   {total_logits_gb:.2f} GB  (Next-token probs)")
    print(f"4. CUDA Overhead:  {overhead_gb:.2f} GB")
    print("-" * 30)
    print(f"TOTAL VRAM:        {total_vram_gb:.2f} GB")
    
    if total_vram_gb < 24:
         print("✅ Fits easily on RTX 3090 / 4090 / 5090")
    else:
         print("❌ WARNING: Might OOM on 24GB cards")

if __name__ == "__main__":
    measure_exact_vram_with_logits()