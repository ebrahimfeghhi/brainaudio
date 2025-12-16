import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from brainaudio.inference.decoder.neural_lm_fusion import HuggingFaceLMFusion
import time
from unsloth import FastModel

# ==========================================
# 1. Using HuggingFaceLMFusion from decoder
# ==========================================

# Replace with the quantized model
MODEL_NAME = "google/gemma-3-270m" 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")

try:
       
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32
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
    ["As such a high clay content", "Has such a high clay content"]
]

print("\nRunning LM Fusion Scorer...")
start_time = time.perf_counter()
scores, _ = lm_fusion.score_continuations(contexts, candidates)
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