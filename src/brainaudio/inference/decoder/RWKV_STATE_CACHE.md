# RWKVStateCache

A state management class for RWKV language models during batched beam search decoding.

## Overview

Unlike transformer models that use Key-Value (KV) caches, RWKV models maintain recurrent states that get updated with each token. `RWKVStateCache` manages these states efficiently for beam search, where we need to:

1. Track states for multiple beams simultaneously
2. Gather specific beam states for LM scoring
3. Update states after forward passes
4. Reorder states after beam pruning

## Memory Layout

States are stored in a **flattened layout** for efficient GPU operations:

```
flat_index = batch * (beam_size * num_homophones) + beam * num_homophones + homophone
```

### Example

With `batch_size=1`, `beam_size=400`, `num_homophones=3`:

| Flat Index | Batch | Beam | Homophone |
|------------|-------|------|-----------|
| 0          | 0     | 0    | 0         |
| 1          | 0     | 0    | 1         |
| 2          | 0     | 0    | 2         |
| 3          | 0     | 1    | 0         |
| 4          | 0     | 1    | 1         |
| ...        | ...   | ...  | ...       |
| 1199       | 0     | 399  | 2         |

This layout groups homophones together, which is efficient because homophones of the same beam often need to be accessed together.

## State Structure

RWKV models have multiple state types per layer. The cache handles two formats:

### Dict Mode (RWKV v7+)

Each layer returns a dictionary of states:
```python
{
    'recurrent_state': Tensor,  # Main RNN state
    'attn_state': Tensor,       # Attention state (may be None)
    'conv_state': Tensor,       # Convolution state (may be None)
    'ffn_state': Tensor,        # Feed-forward state (may be None)
}
```

Not all layers have all state types. The cache tracks which layers have which states via `valid_layer_indices`.

### Tensor Mode (Legacy)

Older RWKV versions return a single tensor per layer.

## Storage Shape

For each state key, the tensor shape is:
```
[num_valid_layers, total_beams, ...]
```

Where `...` depends on the state type (e.g., `[heads, dim, dim]` for recurrent state).

## Key Methods

### `__init__(model, tokenizer, batch_size, beam_size, num_homophones, device)`

Initializes the cache by:
1. Running a forward pass with BOS token
2. Extracting the state structure
3. Replicating seed states for all beam slots

```python
state_cache = RWKVStateCache(
    model=rwkv_model,
    tokenizer=tokenizer,
    batch_size=1,
    beam_size=400,
    num_homophones=3,
    device="cuda"
)
```

### `get_initial_logits() -> Tensor`

Returns logits from the BOS token, expanded for all beams. Use this to score the first word of each beam.

```python
# Shape: [total_beams, vocab_size]
logits = state_cache.get_initial_logits()
```

### `_flat_index(batch_idx, beam_idx, homophone_idx) -> int`

Converts semantic (batch, beam, homophone) coordinates to a flat index.

```python
idx = state_cache._flat_index(0, 5, 2)  # Batch 0, beam 5, homophone 2
```

### `_flat_indices_for_tuples(tuples) -> Tensor`

Batch conversion of coordinate tuples to flat indices.

```python
indices = state_cache._flat_indices_for_tuples([
    (0, 5, 0),   # Batch 0, beam 5, homophone 0
    (0, 10, 1),  # Batch 0, beam 10, homophone 1
])
```

### `gather_states(flat_indices) -> list`

Extracts states for specific beams, formatted for model input.

```python
indices = torch.tensor([0, 3, 6], device="cuda")
states = state_cache.gather_states(indices)
# Returns list of layer dicts ready for model(past_key_values=states)
```

### `update(active_indices, candidate_ids) -> Tensor`

Runs a forward pass for specific beams and updates their states in-place.

```python
active = torch.tensor([0, 1, 2], device="cuda")
tokens = torch.tensor([[101], [102], [103]], device="cuda")  # One token per beam

logits = state_cache.update(active, tokens)
# logits shape: [3, 1, vocab_size]
# States for beams 0, 1, 2 are now updated
```

### `reorder(survivor_indices)`

Reorders the cache after beam pruning. Position `i` will contain the state that was at `survivor_indices[i]`.

```python
# After pruning, beam 5 becomes beam 0, beam 10 becomes beam 1, etc.
survivors = torch.tensor([5, 10, 15, ...], device="cuda")
state_cache.reorder(survivors)
```

### `clone_state(src_flat_idx, dst_flat_idx)`

Copies state from one slot to another. Useful when a beam spawns multiple homophone variants.

```python
# Copy state from beam 0, homophone 0 to beam 0, homophone 1
state_cache.clone_state(0, 1)
```

## Handling None States

Some RWKV layers don't use all state types. For example, `attn_state` might be `None` for certain layers. The cache handles this by:

1. Tracking which layers have valid states: `self.valid_layer_indices[key]`
2. Only storing tensors for valid layers
3. Reconstructing the full structure (with Nones) when gathering for model input

```python
# Example: recurrent_state exists for all 12 layers
# But attn_state only exists for layers 0, 4, 8

self.valid_layer_indices = {
    'recurrent_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'attn_state': [0, 4, 8],
    'conv_state': None,  # All None
    'ffn_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}
```

## Usage in Beam Search

```python
# 1. Initialize cache
state_cache = RWKVStateCache(model, tokenizer, 1, beam_size, num_homophones)

# 2. Get initial logits for first word scoring
initial_logits = state_cache.get_initial_logits()

# 3. During decoding, when beams need LM scoring:
active_beams = torch.tensor([...], device="cuda")  # Beams that hit word boundary
next_tokens = torch.tensor([[tok] for tok in tokens], device="cuda")
logits = state_cache.update(active_beams, next_tokens)

# 4. After beam pruning:
survivor_indices = torch.tensor([...], device="cuda")
state_cache.reorder(survivor_indices)
```

## Memory Considerations

The cache pre-allocates memory for all possible beam slots:

```
Memory ≈ num_layers × total_beams × state_size_per_layer
```

For RWKV-7 0.1B with 400 beams × 3 homophones:
- ~12 layers × 1200 beams × ~1MB per state ≈ 14GB

Consider reducing `beam_size` or `num_homophones` if memory is constrained.

## Comparison with Transformer KV Cache

| Aspect | Transformer KV Cache | RWKV State Cache |
|--------|---------------------|------------------|
| Growth | Grows with sequence length | Fixed size |
| Content | Keys and Values | Recurrent states |
| Update | Append new K/V | Replace state |
| Memory | O(seq_len × beams) | O(beams) |

RWKV's fixed-size state is advantageous for long sequences, as memory doesn't grow with context length.
