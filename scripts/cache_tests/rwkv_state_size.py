import os
import torch
import gc

# Force configuration to match your setup
os.environ["RWKV_V7_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # Use '0' since you don't have nvcc

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

# --- CONFIG ---
MODEL_FILE = "/home/ebrahim/brainaudio/llms/rwkv7-g1b-2.9b"
BEAM_SIZE = 500

print(f"Loading {MODEL_FILE}...")
# Use 'cuda fp16' to simulate real VRAM usage
model = RWKV(model=MODEL_FILE, strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

# 1. Get a Single State Object
print("\nRunning warm-up pass...")
logits, state_single = model.forward(pipeline.encode("The"), None)

# 2. Measure Size of One State
def get_state_size_bytes(state_obj):
    total_bytes = 0
    # RWKV v7 state is a list of tensors
    for tensor in state_obj:
        if isinstance(tensor, torch.Tensor):
            total_bytes += tensor.element_size() * tensor.nelement()
    return total_bytes

bytes_per_beam = get_state_size_bytes(state_single)
mb_per_beam = bytes_per_beam / (1024 * 1024)

print(f"\n--- MEMORY REPORT (RWKV-7 2.9B) ---")
print(f"State Tensors per Beam: {len(state_single)}")
print(f"VRAM per Beam:          {mb_per_beam:.2f} MB")

# 3. Simulate Total Cost
total_vram_gb = (bytes_per_beam * BEAM_SIZE) / (1024**3)

print(f"\n--- SCALING PREDICTION ---")
print(f"Beam Size:     {BEAM_SIZE}")
print(f"Total Cache:   {total_vram_gb:.2f} GB (Estimated)")

# 4. Actual Allocation Test (Optional)
# We try to allocate it to see if it fits
try:
    print(f"\nAllocating {BEAM_SIZE} states on GPU...")
    beam_cache = []
    for i in range(BEAM_SIZE):
        # Deep copy tensors to simulate distinct beams
        beam_cache.append([t.clone() for t in state_single])
    
    print("✅ SUCCESS: Allocated 500 beams on GPU.")
    
    # Check PyTorch reserved memory
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"Current Total VRAM Reserved: {reserved:.2f} GB")
    
except RuntimeError as e:
    print(f"❌ OOM ERROR: Could not allocate all beams.")
    print(f"Error details: {e}")