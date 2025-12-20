import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List
from brainaudio.inference.decoder.neural_lm_fusion import HuggingFaceLMFusion
import time

# ==========================================
# 1. Using HuggingFaceLMFusion from decoder
# ==========================================

MODEL_NAME = "google/gemma-3-270m"
USE_4BIT = False  # Toggle this to switch between quantized and full precision

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device} (4-bit: {USE_4BIT})...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        ).to(device)


except OSError:
    print(f"\n[Error] Could not load {MODEL_NAME}.")
    print("Ensure you are logged in via `huggingface-cli login` and have accepted the Gemma license.")
    exit()

# Gemma doesn't have a PAD token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lm_fusion = HuggingFaceLMFusion(model, tokenizer, weight=1.0, device=None)



# ==========================================
# 3. The Tests
# ==========================================
contexts = [
    ""
]
candidates = [
    ["I'm originally from maine.", "I'm originally from man."]
]

print("\nRunning LM Fusion Scorer...")
start_time = time.perf_counter()
scores = lm_fusion.score_continuations(contexts, candidates)
elapsed = time.perf_counter() - start_time
print(f"Scoring took {elapsed:.4f} seconds")

print("-" * 65)
print(f"{'Context':<35} | {'Word':<10} | {'Score':<10}")
print("-" * 65)

for i, ctx in enumerate(contexts):
    word_scores = scores[i]
    words = candidates[i]
    
    likely_word, unlikely_word = words[0], words[1]
    likely_score, unlikely_score = word_scores[0], word_scores[1]

    # Print results
    for word, score in zip(words, word_scores):
        print(f"{ctx} | {word:<10} | {score:.4f}")
    
    # Logic Check
    print(f"Diff: {likely_score - unlikely_score:.4f}")
    if likely_score > unlikely_score:
        print(f"✅ SUCCESS")
    else:
        print(f"❌ FAILURE")
    print("-" * 65)