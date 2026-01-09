#!/usr/bin/env python3
"""
Test script to verify get_initial_rwkv_state function.

This script:
1. Loads an RWKV model
2. Gets initial state and logits after BOS token (expanded for beam search)
3. Verifies the state can be used for continued generation
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from brainaudio.inference.decoder.neural_lm_fusion_kv import get_initial_rwkv_state


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Configuration
    model_name = "fla-hub/rwkv7-0.1B-g1"
    batch_size = 1
    beam_size = 100
    num_homophones = 3
    total_beams = batch_size * beam_size * num_homophones

    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    model.eval()

    # Get initial state and logits (expanded for beam search)
    print(f"\nGetting initial RWKV state for {total_beams} beams "
          f"({batch_size} batch × {beam_size} beams × {num_homophones} homophones)...")
    state, logits = get_initial_rwkv_state(
        model, tokenizer,
        batch_size=batch_size,
        beam_size=beam_size,
        num_homophones=num_homophones,
        device=device
    )

    print(f"Logits shape: {logits.shape}")
    print(f"State type: {type(state)}")
    print(f"Number of layers: {len(state)}")

    if isinstance(state[0], dict):
        print(f"State keys: {list(state[0].keys())}")
        for key in state[0].keys():
            val = state[0][key]
            if val is not None:
                print(f"  {key}: shape {val.shape}, dtype {val.dtype}")
            else:
                print(f"  {key}: None")

    # Test that we can use the state for generation with a subset of beams
    print("\nTesting generation with a subset of beams...")

    # Take first 5 beams
    subset_size = 5
    subset_indices = list(range(subset_size))

    # Gather state for subset
    subset_state = []
    for layer in state:
        if isinstance(layer, dict):
            subset_layer = {}
            for key, tensor in layer.items():
                if tensor is not None:
                    subset_layer[key] = tensor[subset_indices]
                else:
                    subset_layer[key] = None
            subset_state.append(subset_layer)
        else:
            subset_state.append(layer[subset_indices] if layer is not None else None)

    subset_logits = logits[subset_indices]

    # Sample tokens from logits
    log_probs = F.log_softmax(subset_logits, dim=-1)
    next_tokens = torch.argmax(log_probs, dim=-1)  # Shape: [5]
    print(f"First tokens for 5 beams: {next_tokens.tolist()}")
    print(f"  Decoded: {[tokenizer.decode([t]) for t in next_tokens.tolist()]}")

    # Continue with a forward pass using the subset state
    with torch.no_grad():
        outputs = model(
            input_ids=next_tokens.unsqueeze(1),  # Shape: [5, 1]
            past_key_values=subset_state,
            use_cache=True
        )

    new_logits = outputs.logits[:, -1, :]
    print(f"New logits shape: {new_logits.shape}")

    # Sample another token
    log_probs2 = F.log_softmax(new_logits, dim=-1)
    next_tokens2 = torch.argmax(log_probs2, dim=-1)
    print(f"Second tokens: {next_tokens2.tolist()}")
    print(f"  Decoded: {[tokenizer.decode([t]) for t in next_tokens2.tolist()]}")

    print("\nState can be used for continued generation!")


if __name__ == "__main__":
    main()
