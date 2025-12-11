import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

# ==========================================
# 1. The Scorer Class (Optimized for Gemma)
# ==========================================
class SimpleScorer:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer.truncation_side = "left"  # Keep the end of the sentence
        self.tokenizer.padding_side = "right"
        
    @torch.no_grad()
    def score_continuations(self, contexts: List[str], candidate_words: List[List[str]]) -> List[List[float]]:
        
        flat_texts = []
        flat_context_lens = []
        
        # 1. Flatten structure
        for context, candidates in zip(contexts, candidate_words):
            # Calculate context length roughly (approximation for indexing)
            # Note: add_special_tokens=False is safer if we want raw length
            ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
            ctx_len = len(ctx_ids)
            
            for word in candidates:
                # 2. Fix Spacing Logic
                if not context:
                    full_text = word
                elif context.endswith(" ") or word.startswith(" "):
                    full_text = f"{context}{word}"
                else:
                    full_text = f"{context} {word}"
                
                flat_texts.append(full_text)
                flat_context_lens.append(ctx_len) 

        if not flat_texts:
            return []
    
        # 3. Batch Tokenize
        # Note: 'padding=True' is essential if lengths differ
        inputs = self.tokenizer(
            flat_texts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        input_ids = inputs.input_ids 
        attention_mask = inputs.attention_mask # padded tokens have a value of 0 

        # Forward Pass
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits 

        # Shift Logits & Labels
        shift_logits = logits[..., :-1, :].contiguous() # B x (Seq Len - 1) x Vocab
        shift_labels = input_ids[..., 1:].contiguous() # B x (Seq Len - 1)
        
        
        # Log Softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather scores
        gathered_probs = torch.gather(
            log_probs, 
            dim=2, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1) # Beam Size x (Seq Len - 1)
        
        # Reconstruct and Sum
        final_scores = []
        flat_idx = 0
        
        for candidates in candidate_words:
            beam_scores = []
            
            for _ in candidates:
                
                start_idx = flat_context_lens[flat_idx] 
                
                # 4. Fix End Index Logic
                # We want to sum from the end of context to the end of the VALID sequence
                # -1 because gathered_probs is shifted
                total_valid_len = attention_mask[flat_idx].sum().item() - 1
                
                
                score = gathered_probs[flat_idx, start_idx:total_valid_len].sum().item()
                beam_scores.append(score)
                flat_idx += 1
                
            final_scores.append(beam_scores)

        return final_scores

# ==========================================
# 2. Test Setup (Gemma Specifics)
# ==========================================

# Replace with "google/gemma-3-270m" or your custom path if available.
# Using 2b as the closest public proxy.
MODEL_NAME = "google/gemma-3-270m" 

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load in bfloat16 for stability/speed on GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        dtype=torch.float32,
        device_map=device
    )
except OSError:
    print(f"\n[Error] Could not load {MODEL_NAME}.")
    print("Ensure you are logged in via `huggingface-cli login` and have accepted the Gemma license.")
    exit()

# Gemma doesn't have a PAD token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

scorer = SimpleScorer(model, tokenizer, device)

# ==========================================
# 3. The Tests
# ==========================================
contexts = [
    "it’s to valuable to just let people stay in"
]
candidates = [
    ["sorry", "sory"]
]

print("\nRunning Scorer...")
scores = scorer.score_continuations(contexts, candidates)

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