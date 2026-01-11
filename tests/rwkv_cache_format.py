"""Examine the RWKV cached state format returned by HuggingFace."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def examine_rwkv_cache():
    model_id = "fla-hub/rwkv7-0.1B-g1"
    print(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # Run a forward pass with caching
    text = "The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"\nInput text: '{text}'")
    print(f"Input IDs shape: {inputs.input_ids.shape}")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_key_values = outputs.past_key_values

    # Examine the structure
    print(f"\n{'='*60}")
    print("PAST_KEY_VALUES STRUCTURE")
    print(f"{'='*60}")

    print(f"\nType: {type(past_key_values)}")
    print(f"Length (num layers): {len(past_key_values)}")

    # Examine first layer
    print(f"\n--- Layer 0 ---")
    layer0 = past_key_values[0]
    print(f"Type: {type(layer0)}")

    if isinstance(layer0, dict):
        print("Keys:", list(layer0.keys()))
        for key, tensor in layer0.items():
            if tensor is not None:
                print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
            else:
                print(f"  {key}: None")
    elif isinstance(layer0, (tuple, list)):
        print(f"Length: {len(layer0)}")
        for i, tensor in enumerate(layer0):
            if tensor is not None:
                print(f"  [{i}]: shape={tensor.shape}, dtype={tensor.dtype}")
            else:
                print(f"  [{i}]: None")
    else:
        print(f"Shape: {layer0.shape}, dtype={layer0.dtype}")

    # Check if all layers have the same structure
    print(f"\n--- All Layers Summary ---")
    layer_types = [type(layer).__name__ for layer in past_key_values]
    unique_types = set(layer_types)
    print(f"Layer types: {unique_types}")

    if isinstance(past_key_values[0], dict):
        all_keys = [set(layer.keys()) for layer in past_key_values]
        if len(set(map(frozenset, all_keys))) == 1:
            print(f"All layers have same keys: {all_keys[0]}")
        else:
            print("Layers have different keys!")
            for i, keys in enumerate(all_keys):
                print(f"  Layer {i}: {keys}")

    # Test continuation with cache
    print(f"\n{'='*60}")
    print("TESTING CONTINUATION WITH CACHE")
    print(f"{'='*60}")

    # Add one more token
    next_token = tokenizer.encode(" jumps", add_special_tokens=False)
    next_token_tensor = torch.tensor([next_token], device=model.device)

    print(f"\nNext token(s): {next_token} -> '{tokenizer.decode(next_token)}'")
    print(f"Next token tensor shape: {next_token_tensor.shape}")

    with torch.no_grad():
        outputs2 = model(
            next_token_tensor,
            past_key_values=past_key_values,
            use_cache=True
        )

    new_past = outputs2.past_key_values
    print(f"\nNew past_key_values type: {type(new_past)}")
    print(f"New past_key_values length: {len(new_past)}")

    # Compare shapes
    if isinstance(new_past[0], dict):
        print("\nShape comparison (layer 0):")
        for key in new_past[0].keys():
            old_shape = past_key_values[0][key].shape if past_key_values[0][key] is not None else None
            new_shape = new_past[0][key].shape if new_past[0][key] is not None else None
            print(f"  {key}: {old_shape} -> {new_shape}")

    # Test that we can convert to tuple and back
    print(f"\n{'='*60}")
    print("TESTING TUPLE CONVERSION")
    print(f"{'='*60}")

    as_list = list(past_key_values)
    as_tuple = tuple(as_list)
    print(f"Converted to list: {type(as_list)}, len={len(as_list)}")
    print(f"Converted to tuple: {type(as_tuple)}, len={len(as_tuple)}")

    # Test if tuple works as input
    with torch.no_grad():
        outputs3 = model(
            next_token_tensor,
            past_key_values=as_tuple,
            use_cache=True
        )
    print("Tuple format works as past_key_values input!")

    # Test indexing behavior
    print(f"\n{'='*60}")
    print("TESTING BATCH INDEXING")
    print(f"{'='*60}")

    # Create a batch of 3 by repeating
    if isinstance(past_key_values[0], dict):
        expanded_state = []
        for layer in past_key_values:
            expanded_layer = {
                key: tensor.expand(3, *tensor.shape[1:]).contiguous() if tensor is not None else None
                for key, tensor in layer.items()
            }
            expanded_state.append(expanded_layer)

        print(f"Expanded to batch_size=3")
        print(f"Layer 0 shapes after expansion:")
        for key, tensor in expanded_state[0].items():
            if tensor is not None:
                print(f"  {key}: {tensor.shape}")

        # Index with [0, 2] to select first and third
        # Get device from actual tensor, not model.device (can differ for multi-GPU)
        state_device = expanded_state[0]['recurrent_state'].device
        indices = torch.tensor([0, 2], device=state_device)
        gathered_state = []
        for layer in expanded_state:
            gathered_layer = {
                key: tensor[indices] if tensor is not None else None
                for key, tensor in layer.items()
            }
            gathered_state.append(gathered_layer)

        print(f"\nGathered with indices [0, 2]:")
        print(f"Layer 0 shapes after gathering:")
        for key, tensor in gathered_state[0].items():
            if tensor is not None:
                print(f"  {key}: {tensor.shape}")

        # Convert to tuple and test
        gathered_state = tuple(gathered_state)

        # Run forward with gathered state
        batch_tokens = next_token_tensor.expand(2, -1)
        with torch.no_grad():
            outputs4 = model(
                batch_tokens,
                past_key_values=gathered_state,
                use_cache=True
            )
        print(f"\nForward pass with gathered state succeeded!")
        print(f"Output logits shape: {outputs4.logits.shape}")


if __name__ == "__main__":
    examine_rwkv_cache()
