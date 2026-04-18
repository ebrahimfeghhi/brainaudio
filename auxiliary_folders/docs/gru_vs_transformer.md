# GRU vs Transformer: Architectural Differences

Comparison of `GRU_24` (`gru_b2t_24.py`) and `TransformerModel` (`transformer.py`) as used in this codebase.

---

## High-Level Summary

| | GRU | Transformer |
|---|---|---|
| Core operation | Recurrent hidden state update | Multi-head self-attention |
| Sequence processing | Sequential (left to right, or both directions) | Parallel over all timesteps |
| Positional information | Implicit via recurrence | Explicit via relative position bias |
| Input preprocessing | Day-specific affine transform + Softsign | None |
| Temporal context | Full left context (or full bidirectional) | Causal mask, chunked, or full bidirectional |
| Inter-layer connections | Direct stacking, no norm/residual between layers | Residual + LayerNorm after every Attention and FFN |
| Down-sampling | Temporal unfolding (stride + kernel) | Patchification (non-overlapping windows) |

---

## Input Preprocessing

**GRU** applies a per-session affine transform before the recurrent layers:

```
x → dayWeights[dayIdx] @ x + dayBias[dayIdx] → Softsign
```

Each recording day has its own learned `(C, C)` weight matrix and bias, initialized to identity/zero so the transform starts as a no-op. This compensates for non-stationarity across sessions (electrode drift, impedance changes, etc.).

**Transformer** has no such transform — the raw neural input goes directly into the patch embedder. It assumes the input distribution is stable enough across sessions to not require per-session correction.

---

## Temporal Down-sampling

Both models reduce the time resolution before their core encoder, but via different mechanisms.

**GRU — Temporal Unfolding:**
`nn.Unfold` slides a window of `kernelLen` samples with stride `strideLen`, concatenating each window into a single vector:
```
T → T' = (T - kernelLen) / strideLen
input_size = C × kernelLen
```
Each GRU step sees a local context of `kernelLen` raw samples.

**Transformer — Patchification:**
Non-overlapping windows of `samples_per_patch` samples are flattened and projected to `dim`:
```
T → T' = ceil(T / samples_per_patch)
patch_dim = samples_per_patch × num_features
```
Patches are non-overlapping (no stride < patch size), so there is no local context overlap between adjacent patches.

---

## Core Encoding

**GRU — Recurrent update:**

At each timestep the hidden state is updated via reset and update gates:
```
z_t = σ(W_z x_t + U_z h_{t-1})      # update gate
r_t = σ(W_r x_t + U_r h_{t-1})      # reset gate
ñ_t = tanh(W_n x_t + r_t ⊙ U_n h_{t-1})
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ñ_t
```

Information from earlier timesteps must propagate through the hidden state step by step. Long-range dependencies therefore depend on the gating mechanism to preserve relevant history across many steps.

**Transformer — Self-attention:**

Every patch attends directly to every other (allowed) patch in a single operation:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d) + rel_bias + causal_mask) V
```

Long-range dependencies are resolved in O(1) steps regardless of distance. The relative position bias adds a learned scalar per pairwise distance, giving the model positional awareness without sinusoidal encodings.

---

## Layer Structure

**GRU:**
```
Layer 1 GRU → Dropout → Layer 2 GRU → ... → Layer N GRU
```
Layers are stacked directly inside `nn.GRU`. There is no FFN, LayerNorm, or residual connection between layers.

**Transformer:**
```
for each layer:
    x = Attention(x) + x       # residual
    x = FFN(x) + x             # residual
LayerNorm(x)
```
Each layer has two sub-components (Attention + FFN), each wrapped with a residual connection. A final LayerNorm is applied after all layers. The FFN expands to `mlp_dim_ratio × dim` internally, allowing each position to be transformed non-linearly after attention mixing.

---

## Temporal Masking / Causality

**GRU:** causality is inherent in the forward direction — each step only sees past inputs. Bidirectionality is toggled via `bidirectional=True`, which runs a second GRU in reverse and concatenates outputs.

**Transformer:** causality is enforced externally via an attention mask. Three modes are supported:
- **Causal** (`bidirectional=False`): each patch attends only to itself and earlier patches.
- **Bidirectional** (`bidirectional=True`): all patches attend to all others (no mask).
- **Chunked** (`transformer_chunking.py` / `transformer_demichunking.py`): each patch attends to its chunk plus a bounded left context — enables streaming inference while training with variable chunk sizes.

---

## Inductive Biases

| Bias | GRU | Transformer |
|---|---|---|
| Locality | Strong — hidden state is updated one step at a time | Weak — all positions attend globally by default |
| Order sensitivity | Strong — recurrence encodes position implicitly | Weak — requires explicit position encoding (relative bias here) |
| Long-range dependencies | Harder — must survive many gating steps | Easier — direct attention between any two positions |
| Parameter sharing | Same weights reused at every timestep | Same weights reused at every layer position |
