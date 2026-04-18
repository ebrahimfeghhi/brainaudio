# TransformerModel Architecture

`src/brainaudio/models/transformer.py`

Adapted from [Francois Porcher's ViT implementation](https://github.com/FrancoisPorcher/vit-pytorch). Treats neural input as a sequence of patches and decodes it into per-timestep class logits (CTC-style output).

---

## Overview

```
neuralInput (B, T, F)
       │
       ▼
 pad_to_multiple        — pad T so it's divisible by samples_per_patch
       │
       ▼
  unsqueeze(1)          — (B, 1, T, F)  treat as single-channel 2D image
       │
       ▼
 patch_embedder         — (B, num_patches, dim)
       │
  [optional time masking during training]
       │
       ▼
 dropout_layer          — input dropout
       │
       ▼
 Transformer            — depth × (Attention + FFN) with residuals
       │
       ▼
 projection             — (B, num_patches, nClasses + 1)
```

The `+1` in the output accounts for the CTC blank token.

---

## Components

### `pad_to_multiple`

Pads the time dimension (dim=1) so that `T % samples_per_patch == 0`. This is necessary before patchification since the `Rearrange` operation requires an evenly divisible sequence length.

---

### `patch_embedder` (`nn.Sequential`)

Converts the raw neural signal into a sequence of patch embeddings.

```
Rearrange  →  LayerNorm  →  Linear  →  LayerNorm
```

1. **Rearrange** (`b c (h p1) (w p2) -> b (h w) (p1 p2 c)`): Splits the time axis into non-overlapping windows of size `samples_per_patch`, flattening each window + its features into a single vector. Output shape: `(B, num_patches, patch_dim)` where `patch_dim = samples_per_patch × num_features`.
2. **LayerNorm**: Normalizes each raw patch vector.
3. **Linear**: Projects `patch_dim → dim`.
4. **LayerNorm**: Normalizes the projected embedding.

During training with time masking enabled, the `Rearrange` step runs first so that masking operates on the patchified (pre-projection) representation, then the remaining layers are applied.

---

### `FFN`

Position-wise feed-forward network applied after each attention layer.

```
LayerNorm  →  Linear(dim → hidden_dim)  →  GELU  →  Dropout  →  Linear(hidden_dim → dim)  →  Dropout
```

`hidden_dim = mlp_dim_ratio × dim`. Uses pre-norm (LayerNorm before the linear layers).

---

### `Attention`

Multi-head self-attention with **T5-style relative position bias**.

**Scaled dot-product attention:**

```
dots = (Q @ K^T) / sqrt(dim_head)     shape: (B, heads, N, N)
dots = dots + rel_bias                 additive relative position bias
dots = masked_fill(temporal_mask==0, -inf)   causal masking (if unidirectional)
attn = softmax(dots)
out  = attn @ V
```

**Relative position bias**: An embedding table of size `(2 * max_rel_dist - 1, 1)` maps each pairwise relative distance `(i - j)` to a scalar bias added to the attention logits. Distances are clamped to `[-max_rel_dist+1, max_rel_dist-1]`. This lets the model learn position-dependent attention patterns without fixed sinusoidal encodings.

**Output projection**: `Linear(inner_dim → dim)` when `heads > 1` or `dim_head != dim`, otherwise identity.

---

### `create_temporal_mask`

Builds a causal (autoregressive) mask of shape `(1, 1, T, T)`. Entry `[i, j]` is `True` if position `j ≤ i`, meaning each timestep can only attend to itself and earlier timesteps. Used when `bidirectional=False`.

---

### `Transformer`

Stacks `depth` identical blocks, each containing one `Attention` layer and one `FFN` with residual connections:

```
x = Attention(x) + x
x = FFN(x) + x
```

A final `LayerNorm` is applied after all blocks.

---

### `TransformerModel` (top-level)

Inherits from `BaseTimeMaskedModel` which provides the `apply_time_masking` method for SpecAugment-style time masking during training.

**Constructor parameters:**

| Parameter | Description |
|---|---|
| `samples_per_patch` | Number of time samples per patch (patch width) |
| `num_features` | Number of neural features (patch height / input channels) |
| `dim` | Transformer embedding dimension |
| `depth` | Number of transformer blocks |
| `heads` | Number of attention heads |
| `dim_head` | Dimension per attention head |
| `mlp_dim_ratio` | FFN hidden dim multiplier (`hidden = ratio × dim`) |
| `dropout` | Dropout rate inside transformer |
| `input_dropout` | Dropout applied to patch embeddings before transformer |
| `nClasses` | Number of output classes (projection outputs `nClasses + 1` for CTC blank) |
| `max_mask_pct` | Max fraction of patches to mask during training (0 = disabled) |
| `num_masks` | Number of contiguous mask segments |
| `return_final_layer` | If `True`, returns `(logits, hidden_states)` instead of just `logits` |
| `bidirectional` | If `True`, no causal mask — all patches attend to all others |

**`compute_length(X_len)`**: Maps input time lengths to patch sequence lengths via `ceil(X_len / samples_per_patch)`. Used by CTC loss to compute valid output lengths.
