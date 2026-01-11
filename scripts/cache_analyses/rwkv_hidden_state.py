import torch
from transformers import AutoModelForCausalLM

def get_mem_mb():
    """Returns currently allocated CUDA memory in MB."""
    return torch.cuda.memory_allocated() / 1024 / 1024

def measure_rwkv_state_cost():
    # Configuration
    MODEL_ID = "fla-hub/rwkv7-0.1B-g1"
    TOTAL_BEAMS = 5000 # Your target beam count
    DTYPE = torch.float16  # Using FP16 (recommended)
    
    print(f"--- RWKV State Memory Test (Beams={TOTAL_BEAMS}) ---")
    
    # 1. Load Model
    print("1. Loading model...", end="")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(" [WARNING: No GPU found, measuring CPU RAM]")
    
    # Load in FP16 to be realistic
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        torch_dtype=DTYPE
    ).to(device)
    print(" Done.")
    
    baseline_mem = get_mem_mb()
    print(f"   Base Model VRAM: {baseline_mem:.2f} MB")

    # 2. Get a Single State (Batch=1)
    print("\n2. Generating initial state for 1 beam...")
    dummy_input = torch.tensor([[0]], device=device)
    with torch.no_grad():
        out = model(dummy_input, use_cache=True)
        single_state = out.past_key_values

    # 3. Calculate Size Per Beam (Math)
    size_bytes = 0
    param_count = 0
    
    # Iterate through List[Dict[str, Tensor]]
    for layer in single_state:
        for key, tensor in layer.items():
            if isinstance(tensor, torch.Tensor):
                size_bytes += tensor.element_size() * tensor.nelement()
                param_count += tensor.nelement()
    
    size_mb = size_bytes / 1024 / 1024
    print(f"   Single Beam State Size: {size_mb:.4f} MB")
    print(f"   State Parameters per Beam: {param_count:,}")

    # 4. Extrapolate to 2000 Beams
    total_theoretical = size_mb * TOTAL_BEAMS
    print(f"\n3. Theoretical Cost for {TOTAL_BEAMS} beams:")
    print(f"   {total_theoretical:.2f} MB (Just for the tensors)")

    # 5. Stress Test: Actually Allocate it
    print(f"\n4. Stress Test: Allocating {TOTAL_BEAMS} beams on GPU...")
    
    try:
        expanded_state = []
        for layer in single_state:
            new_layer = {}
            for key, tensor in layer.items():
                if isinstance(tensor, torch.Tensor):
                    # Expand: [1, Heads, Dim] -> [2000, Heads, Dim]
                    # We use .contiguous() to force physical memory allocation
                    new_tensor = tensor.expand(TOTAL_BEAMS, *tensor.shape[1:]).contiguous()
                    new_layer[key] = new_tensor
                else:
                    new_layer[key] = tensor
            expanded_state.append(new_layer)
        
        # Measure after allocation
        torch.cuda.synchronize()
        current_mem = get_mem_mb()
        actual_cost = current_mem - baseline_mem
        
        print(f"   SUCCESS!")
        print(f"   Actual VRAM Used: {actual_cost:.2f} MB")
        print(f"   Difference (Overhead): {actual_cost - total_theoretical:.2f} MB")
        
    except torch.cuda.OutOfMemoryError:
        print("   FAILED: CUDA Out Of Memory!")
        
    # 6. The "Gather" Spike Simulation
    print("\n5. Simulating the 'Gather' Spike (The crash cause)...")
    print("   Attempting to copy 50% of beams for scoring (Gather Step)...")
    
    try:
        # Simulate selecting top 1000 candidates to score
        subset_size = TOTAL_BEAMS // 2 
        subset_state = []
        
        # This simulates: gathered_state = state[indices]
        for layer in expanded_state:
            new_layer = {}
            for key, tensor in layer.items():
                if isinstance(tensor, torch.Tensor):
                    # Slicing and cloning happens here
                    new_layer[key] = tensor[:subset_size].clone() 
            subset_state.append(new_layer)
            
        torch.cuda.synchronize()
        spike_mem = get_mem_mb()
        spike_cost = spike_mem - current_mem
        
        print(f"   Spike Size: {spike_cost:.2f} MB")
        print(f"   Total Peak VRAM: {spike_mem:.2f} MB")
        
    except torch.cuda.OutOfMemoryError:
        print("   FAILED: OOM during Gather!")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    measure_rwkv_state_cost()