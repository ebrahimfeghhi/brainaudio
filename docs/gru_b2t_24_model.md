# GRU_24 Architecture

`src/brainaudio/models/gru_b2t_24.py`

GRU-based neural speech encoder. Inherits from `BaseTimeMaskedModel` for SpecAugment-style time masking. Outputs per-timestep class logits for CTC decoding.

---

## Overview

```
neuralInput (B, T, C)
       │
       ▼
 [optional time masking] — SpecAugment during training
       │
       ▼
 inputDropoutLayer      — input dropout
       │
       ▼
 day-specific affine    — per-session linear transform + Softsign
       │
       ▼
 unfolder               — temporal striding via nn.Unfold → (B, T', C×kernelLen)
       │
       ▼
 gru_decoder            — stacked GRU → (B, T', hidden_dim[×2])
       │
       ▼
 post_rnn_block         — Identity (placeholder for future refinement)
       │
       ▼
 fc_decoder_out         — (B, T', nClasses + 1)
```

The `+1` in the output accounts for the CTC blank token.

---

## Components

### Gaussian Smoothing

Currently commented out. When enabled, `GaussianSmoothing(neural_dim, kernel_size=20, sigma=2.0, dim=1)` would low-pass filter the neural signal along the time axis before any other processing.

---

### Time Masking

Inherited from `BaseTimeMaskedModel`. During training, when `max_mask_pct > 0`, `num_masks` contiguous segments of the time axis are zeroed out (SpecAugment-style). Applied after smoothing but before the day-specific transform.

---

### Day-specific Affine Transform

Each recording session (day) has its own learnable weight matrix and bias:

```
dayWeights : (nDays, C, C)   — initialized to identity per day
dayBias    : (nDays, 1, C)   — initialized to zero
```

For a batch where each sample has a `dayIdx`, the transform is:

```
transformedNeural = einsum("btd,bdk->btk", neuralInput, dayWeights[dayIdx]) + dayBias[dayIdx]
transformedNeural = Softsign(transformedNeural)
```

This lets the model account for non-stationarity across recording sessions without changing the network weights. Weights are initialized to identity so the transform starts as a no-op.

Additionally, per-day `nn.Linear` layers (`inpLayer0`, `inpLayer1`, ...) are registered but not used in the forward pass — they are legacy parameters.

---

### Temporal Unfolding

`nn.Unfold((kernelLen, 1), stride=strideLen)` slides a window of length `kernelLen` across the time axis with step `strideLen`, concatenating each window into a single vector:

```
(B, C, T) → unfold → (B, C×kernelLen, T') → permute → (B, T', C×kernelLen)
```

where `T' = (T - kernelLen) / strideLen`. This simultaneously down-samples the sequence and widens the feature dimension, giving each GRU step a local temporal context of `kernelLen` samples.

---

### GRU

```python
nn.GRU(input_size = neural_dim × kernelLen,
        hidden_size = hidden_dim,
        num_layers  = layer_dim,
        batch_first = True,
        dropout     = dropout,       # only applied between layers (layer_dim > 1)
        bidirectional = bidirectional)
```

At each timestep `t`, the GRU computes:

```
z_t = σ(W_z x_t + U_z h_{t-1} + b_z)          # update gate
r_t = σ(W_r x_t + U_r h_{t-1} + b_r)          # reset gate
ñ_t = tanh(W_n x_t + r_t ⊙ (U_n h_{t-1}) + b_n)   # candidate hidden state
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ñ_t        # new hidden state
```

where `x_t` is the input at timestep `t` (shape `C×kernelLen`), `h_{t-1}` is the previous hidden state, `σ` is sigmoid, and `⊙` is element-wise multiplication.

**Gates:**
- **Update gate** `z_t`: controls how much of the previous hidden state to carry forward vs. replace with new information. Values near 1 preserve the past; values near 0 let new input dominate.
- **Reset gate** `r_t`: controls how much of the previous hidden state is used when computing the candidate. Values near 0 effectively reset memory, allowing the model to ignore irrelevant history.
- **Candidate** `ñ_t`: a proposed new hidden state computed from the current input and a reset-gated version of the past hidden state.

**Multi-layer:** with `layer_dim > 1`, the output `h_t` of layer `l` becomes the input `x_t` to layer `l+1`. There is no FFN, LayerNorm, or residual connection between layers — stacking is handled entirely inside `nn.GRU`. This differs from the Transformer, where each layer is an Attention + FFN pair with residuals and normalization. Dropout is applied between GRU layers (not on the final output).

**Bidirectional:** if `bidirectional=True`, a second GRU processes the sequence in reverse. Its hidden states are concatenated with the forward pass at each timestep, doubling the output dimension to `hidden_dim × 2`.

**Initialization:**
- `weight_hh` (U matrices): **orthogonal** — preserves gradient norms through time, reducing vanishing/exploding gradients.
- `weight_ih` (W matrices): **Xavier uniform** — scales variance relative to input/output size.
- Hidden state `h0`: zeros, detached so no gradient flows back through the initial state.

---

### `post_rnn_block`

Currently `nn.Identity()` — a placeholder for any post-RNN processing (e.g. a transformer refinement layer) without changing the forward pass signature.

---

### Output Projection

`nn.Linear(rnn_out_dim, n_classes + 1)` maps each GRU timestep to class logits. The extra class is the CTC blank token.

---

### `compute_length(X_len)`

Maps input time lengths to output sequence lengths:

```
T' = (X_len - kernelLen) / strideLen
```

Used by CTC loss to compute valid output lengths after temporal down-sampling.

---

## Constructor Parameters

| Parameter | Description |
|---|---|
| `neural_dim` | Number of input neural channels |
| `n_classes` | Number of output classes (projection outputs `n_classes + 1` for CTC blank) |
| `hidden_dim` | GRU hidden state size |
| `layer_dim` | Number of stacked GRU layers |
| `nDays` | Number of recording sessions (for day-specific affine transforms) |
| `dropout` | Dropout between GRU layers (only active when `layer_dim > 1`) |
| `input_dropout` | Dropout applied to inputs before the day-specific transform |
| `strideLen` | Stride of the unfolding window (controls temporal down-sampling rate) |
| `kernelLen` | Width of the unfolding window (local context per GRU step) |
| `bidirectional` | If `True`, GRU reads the sequence in both directions |
| `max_mask_pct` | Max fraction of timesteps to mask during training (0 = disabled) |
| `num_masks` | Number of contiguous mask segments per sample |
