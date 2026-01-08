import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

# 1. Load Model and Tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# 2. Define Inputs
flat_contexts = [
    'you can', 'u can', 'yu can', 'you can', 'u can', 'yu can', 'you can', 'u can', 'yu can', 'you can', 'u can', 'yu can', 'you can', 'u can', 'yu can', 'u', 'you', 'yu', 'you can', 'u can', 'yu can', 'u', 'you', 'yu', 'you can', 'u can', 'yu can', 'u', 'you', 'yu', 'you can', 'u can', 'yu can', 'u', 'you', 'yu', 'u', 'you', 'yu', 'you can', 'u can', 'yu can', 'you can', 'u can', 'yu can', 'you can', 'u can', 'yu can', 'you can', 'u can', 'yu can', 'u', 'you', 'yu', 'u', 'you', 'yu', 'u', 'you', 'yu'
]

flat_candidates = [
    ['er', 'Er', 'Eure', 'eure', 'ur', 'Ur'], ['er', 'Er', 'Eure', 'eure', 'ur', 'Ur'], ['er', 'Er', 'Eure', 'eure', 'ur', 'Ur'], ['ee', 'Ee', 'e', 'E', 'E.', 'e.'], ['ee', 'Ee', 'e', 'E', 'E.', 'e.'], ['ee', 'Ee', 'e', 'E', 'E.', 'e.'], ['Au', 'au', 'Aux', 'aux', 'eau', 'Eau', 'eaux', 'Eaux', 'ohh', 'Ohh', 'oh', 'Oh', 'o', 'O', 'O.', 'o.', "o'", "O'", 'Owe', 'owe', 'Ow', 'ow'], ['Au', 'au', 'Aux', 'aux', 'eau', 'Eau', 'eaux', 'Eaux', 'ohh', 'Ohh', 'oh', 'Oh', 'o', 'O', 'O.', 'o.', "o'", "O'", 'Owe', 'owe', 'Ow', 'ow'], ['Au', 'au', 'Aux', 'aux', 'eau', 'Eau', 'eaux', 'Eaux', 'ohh', 'Ohh', 'oh', 'Oh', 'o', 'O', 'O.', 'o.', "o'", "O'", 'Owe', 'owe', 'Ow', 'ow'], ['a', 'A', 'uh', 'Uh', 'uhh', 'Uhh'], ['a', 'A', 'uh', 'Uh', 'uhh', 'Uhh'], ['a', 'A', 'uh', 'Uh', 'uhh', 'Uhh'], ['ooh', 'Ooh', 'oooh', 'Oooh', 'Ou', 'ou'], ['ooh', 'Ooh', 'oooh', 'Oooh', 'Ou', 'ou'], ['ooh', 'Ooh', 'oooh', 'Oooh', 'Ou', 'ou'], ['Caen', 'caen', 'cahn', 'Cahn', 'can', 'Can', 'cann', 'Cann', 'kan', 'Kan', 'kanne', 'Kanne', 'kann', 'Kann'], ['Caen', 'caen', 'cahn', 'Cahn', 'can', 'Can', 'cann', 'Cann', 'kan', 'Kan', 'kanne', 'Kanne', 'kann', 'Kann'], ['Caen', 'caen', 'cahn', 'Cahn', 'can', 'Can', 'cann', 'Cann', 'kan', 'Kan', 'kanne', 'Kanne', 'kann', 'Kann'], ['aah', 'Aah', 'ah', 'Ah', 'Ahh', 'ahh', 'Awe', 'awe'], ['aah', 'Aah', 'ah', 'Ah', 'Ahh', 'ahh', 'Awe', 'awe'], ['aah', 'Aah', 'ah', 'Ah', 'Ahh', 'ahh', 'Awe', 'awe'], ['canned', 'Canned'], ['canned', 'Canned'], ['canned', 'Canned'], ['ae', 'Ae', 'a.', 'A.', 'ay', 'Ay'], ['ae', 'Ae', 'a.', 'A.', 'ay', 'Ay'], ['ae', 'Ae', 'a.', 'A.', 'ay', 'Ay'], ["caen's", "Caen's", 'cannes', 'Cannes', "can's", "Can's", 'cans', 'Cans', 'kanz', 'Kanz'], ["caen's", "Caen's", 'cannes', 'Cannes', "can's", "Can's", 'cans', 'Cans', 'kanz', 'Kanz'], ["caen's", "Caen's", 'cannes', 'Cannes', "can's", "Can's", 'cans', 'Cans', 'kanz', 'Kanz'], ['Aue', 'aue'], ['Aue', 'aue'], ['Aue', 'aue'], ["Can't", "can't", 'Cant', 'cant', 'Kandt', 'kandt', 'Kant', 'kant'], ["Can't", "can't", 'Cant', 'cant', 'Kandt', 'kandt', 'Kant', 'kant'], ["Can't", "can't", 'Cant', 'cant', 'Kandt', 'kandt', 'Kant', 'kant'], ['Canner', 'canner', 'Kanner', 'kanner'], ['Canner', 'canner', 'Kanner', 'kanner'], ['Canner', 'canner', 'Kanner', 'kanner'], ['Eh', 'eh'], ['Eh', 'eh'], ['Eh', 'eh'], ['ai', 'Ai', 'Aye', 'aye', 'eye', 'Eye', 'i', 'I', 'i.', 'I.'], ['ai', 'Ai', 'Aye', 'aye', 'eye', 'Eye', 'i', 'I', 'i.', 'I.'], ['ai', 'Ai', 'Aye', 'aye', 'eye', 'Eye', 'i', 'I', 'i.', 'I.'], ['oie', 'Oie', 'oi', 'Oi', 'Oye', 'oye', 'oy', 'Oy'], ['oie', 'Oie', 'oi', 'Oi', 'Oye', 'oye', 'oy', 'Oy'], ['oie', 'Oie', 'oi', 'Oi', 'Oye', 'oye', 'oy', 'Oy'], ['aw', 'Aw'], ['aw', 'Aw'], ['aw', 'Aw'], ['Canney', 'canney', 'canny', 'Canny'], ['Canney', 'canney', 'canny', 'Canny'], ['Canney', 'canney', 'canny', 'Canny'], ['Kanno', 'kanno'], ['Kanno', 'kanno'], ['Kanno', 'kanno'], ['cana', 'Cana', 'Kana', 'kana', 'khanna', 'Khanna'], ['cana', 'Cana', 'Kana', 'kana', 'khanna', 'Khanna'], ['cana', 'Cana', 'Kana', 'kana', 'khanna', 'Khanna']
]

# 3. Pre-compute KV Cache for Unique Contexts
print("Pre-computing KV Cache...")
unique_contexts = np.unique(flat_contexts)
kv_cache_dict = {}

# We interpret the request as: "Use the cache to generate logits". 
# The logits for the *next* token come from processing the *last* token of the context.
# To save computation, we cache the state of context[:-1], and at runtime we process context[-1].

for ctx in unique_contexts:
    inputs = tokenizer(ctx, return_tensors="pt")
    input_ids = inputs.input_ids
    
    # If context is a single token, we can't really cache "previous" state, 
    # so we store None (or handle specifically).
    if input_ids.shape[1] > 1:
        # Run model on all but the last token to get cache
        with torch.no_grad():
            outputs = model(input_ids[:, :-1])
        
        # Store (past_key_values, last_token_id)
        # We need the last token ID to feed it back in during the "scoring" phase
        kv_cache_dict[ctx] = (outputs.past_key_values, input_ids[:, -1:])
    else:
        # Edge case: Context is 1 token. No cache possible for "previous" tokens.
        kv_cache_dict[ctx] = (None, input_ids)

print(f"Cached {len(kv_cache_dict)} unique contexts.")

# 4. Scoring Loop
print("\nScoring candidates...")
results = []

# Iterate through the original flat lists (to maintain order/mapping)
for i, (ctx, candidates) in enumerate(zip(flat_contexts, flat_candidates)):
    
    # A. Retrieve Cache
    past_key_values, last_token_input = kv_cache_dict[ctx]
    
    # B. Generate Logits using Cache
    # This is the critical step: we only process the *last token* of the context
    with torch.no_grad():
        if past_key_values is not None:
            outputs = model(last_token_input, past_key_values=past_key_values)
        else:
            outputs = model(last_token_input)
            
    # Logits for the NEXT token are the last item in the sequence output
    # Shape: [1, 1, Vocab_Size]
    next_token_logits = outputs.logits[:, -1, :]
    next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
    
    # C. Score Candidates
    # Convert candidates to token IDs (assuming single token as requested)
    cand_ids = []
    valid_candidates = []
    
    for cand in candidates:
        # Note: add_special_tokens=False is important to avoid adding BOS again
        # We prepend a space because in sentencepiece " word" is different from "word"
        # Adjust per your tokenizer's specific behavior if needed.
        tokens = tokenizer.encode(" " + cand, add_special_tokens=False)
        if len(tokens) == 1:
            cand_ids.append(tokens[0])
            valid_candidates.append(cand)
        else:
            # Fallback: take first token or skip (logging purely for visibility)
            cand_ids.append(tokens[0]) 
            valid_candidates.append(cand)

    cand_tensor = torch.tensor(cand_ids).unsqueeze(0) # Shape [1, Num_Candidates]
    
    # Gather scores
    # We want to select the log_probs at the indices of our candidates
    # next_token_log_probs is [1, Vocab], we assume index corresponds to token ID
    cand_scores = next_token_log_probs[0, cand_ids].tolist()
    
    results.append({
        "context": ctx,
        "scores": list(zip(valid_candidates, cand_scores))
    })

# 5. Print a few examples
for i in [0, 15, 40]: # Random indices to check
    r = results[i]
    print(f"\nIndex {i} | Context: '{r['context']}'")
    # Sort by score for display
    sorted_scores = sorted(r['scores'], key=lambda x: x[1], reverse=True)
    for word, score in sorted_scores:
        print(f"  {word}: {score:.4f}")