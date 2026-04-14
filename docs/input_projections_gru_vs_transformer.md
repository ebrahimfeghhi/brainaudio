# Input Projections: GRU vs Transformer

Comparison of the linear transformations applied to the raw neural input before the main encoder body in `GRU_24` and `TransformerModel`.

---

## Pipelines

**GRU:**
```
(B, T, 256)
    │
    ▼
day-specific Linear (256 × 256) + bias, init to identity   — learned per session
    │
    ▼
Softsign
    │
    ▼
Unfold (kernelLen=32, strideLen) → (B, T', 8192)           — not learned, reshape only
    │
    ▼
weight_ih (8192 → hidden_dim), implicit inside GRU cell    — fused across 3 gates
    │
    ▼
GRU hidden states (B, T', hidden_dim)
```

**Transformer:**
```
(B, T, 256)
    │
    ▼
Rearrange (samples_per_patch=5) → (B, num_patches, 1280)   — not learned, reshape only
    │
    ▼
LayerNorm(1280)
    │
    ▼
Linear (1280 → dim)                                        — explicit learned projection
    │
    ▼
LayerNorm(dim)
    │
    ▼
Transformer blocks (B, num_patches, dim)
```

---

## Key Differences

**Day-specific recalibration**
The GRU applies a learned `256 × 256` linear transform (+ bias) per recording session before any reshaping, initialized to the identity so it starts as a no-op. This compensates for non-stationarity across sessions (e.g. electrode drift). The Transformer has no equivalent — raw input goes directly into the patch reshaping.

**Normalization vs. nonlinearity**
The Transformer wraps its projection with LayerNorm on both sides (pre- and post-projection), stabilizing the scale of the patch vectors before and after the linear map. The GRU uses Softsign after the day transform — a squashing nonlinearity that bounds outputs to `(-1, 1)` — but applies no normalization before the GRU cell.

**Explicit vs. implicit projection**
The Transformer has a standalone `Linear(1280, dim)` that explicitly projects the patch to the encoder dimension. The GRU has no such layer — the equivalent projection is `weight_ih` inside the GRU cell, which maps the 8192-dim input to `hidden_dim` implicitly as part of the gate computations. It is fused into a single matrix of shape `(3 × hidden_dim, 8192)` — one block per gate (update, reset, candidate).

**Temporal resolution**
The GRU groups 32 timesteps per window (8192-dim input), while the Transformer groups 5 timesteps per patch (1280-dim input). The Transformer therefore operates at finer temporal resolution entering the encoder.
