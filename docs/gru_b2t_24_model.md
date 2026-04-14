# GRU_24 Architecture

`src/brainaudio/models/gru_b2t_24.py`

GRU-based neural speech encoder. Inherits from `BaseTimeMaskedModel` for SpecAugment-style time masking. Outputs per-timestep class logits for CTC decoding.

---

## Overview

```
neuralInput (B, T, C)
       │
       ▼
 GaussianSmoothing      — smooth along time axis
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

`GaussianSmoothing(neural_dim, kernel_size=20, sigma=2.0, dim=1)` is applied along the time axis before anything else. It low-pass filters the neural signal to reduce high-frequency noise prior to encoding.

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

- Hidden-to-hidden weights (`weight_hh`) initialized with **orthogonal initialization** — helps preserve gradient norms over long sequences.
- Input-to-hidden weights (`weight_ih`) initialized with **Xavier uniform**.
- Hidden state `h0` is zeros and detached (no gradient flows through initial state).
- Output shape: `(B, T', hidden_dim × 2)` if bidirectional, else `(B, T', hidden_dim)`.

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
