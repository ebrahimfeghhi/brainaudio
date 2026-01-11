import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load Model
model_id = "fla-hub/rwkv7-0.1B-g1"
print(f"Loading {model_id}...")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise RuntimeError("This RWKV7 implementation requires a GPU.")

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Helper to deep clone the specific RWKV7 state structure (List[Dict[str, Tensor]])
def clone_rwkv_state(state):
    new_state = []
    for layer in state:
        new_layer = {}
        for k, v in layer.items():
            if isinstance(v, torch.Tensor):
                new_layer[k] = v.clone() # Actual memory copy
            else:
                new_layer[k] = copy.deepcopy(v)
        new_state.append(new_layer)
    return new_state

# 2. Prepare Inputs
text = "Hello"
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
pad_id = torch.tensor([[tokenizer.pad_token_id]]).to(device)

print(f"\nProcessing '{text}'...")

# 3. Clean Run
with torch.no_grad():
    out_clean = model(input_ids, use_cache=True)
    # !!! CRITICAL FIX: Clone the state immediately !!!
    state_clean = clone_rwkv_state(out_clean.past_key_values)

# 4. Corrupted Run
print("Feeding PAD token...")
with torch.no_grad():
    # We pass the cloned state, but the model will produce a NEW state object usually
    # or update the one passed in. We compare against our safe 'state_clean' copy.
    out_pad = model(pad_id, past_key_values=out_clean.past_key_values, use_cache=True)
    state_corrupted = out_pad.past_key_values

# 5. Measure Corruption
print("\n--- RESULTS (With Cloning) ---")
is_corrupted = False

for i, (layer_clean, layer_dirty) in enumerate(zip(state_clean, state_corrupted)):
    layer_diff = 0.0
    for key in layer_clean.keys():
        t_clean = layer_clean[key]
        t_dirty = layer_dirty[key]
        
        if isinstance(t_clean, torch.Tensor):
            diff = torch.abs(t_clean - t_dirty).sum().item()
            layer_diff += diff
            
    if layer_diff > 1e-5:
        print(f"Layer {i}: State changed by {layer_diff:.4f} (CORRUPTED)")
        is_corrupted = True
    else:
        print(f"Layer {i}: State unchanged.")

if is_corrupted:
    print("\nCONCLUSION: The state IS corrupted (as expected).")
else:
    print("\nCONCLUSION: The state is genuinely safe (Model ignores padding).")