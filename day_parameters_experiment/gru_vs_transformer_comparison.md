# GRU vs Transformer: Model and Training Differences (B2T '24 and '25)

This document summarizes the architectural and training differences between the GRU and Transformer models for the Brain-to-Text 2024 and 2025 datasets.

---

## 1. Model Architecture

### 1.1 Input Processing

| Aspect | GRU | Transformer |
|---|---|---|
| Day-specific transform | Per-day weight matrix + bias (initialized to identity/zero), always applied | Not used (`nDays=None`) |
| Day transform activation | Softsign (always applied after day transform) | Not used (`day_softsign=False`) |
| Input tokenization | Temporal unfolding: `unfold(kernelLen, strideLen)` concatenates a sliding window of raw timesteps into a single vector | Patch embedding: reshapes `(B, T, F)` into non-overlapping patches of size `samples_per_patch × features`, then applies `LayerNorm → Linear → LayerNorm` |
| Input dropout | Applied after day transform | Applied after patch embedding |

### 1.2 Temporal Modeling

| Aspect | GRU | Transformer |
|---|---|---|
| Core module | Multi-layer bidirectional (B2T '24) or unidirectional (B2T '25) GRU | Multi-layer self-attention (Transformer encoder) |
| Positional encoding | Implicit via recurrent processing | T5-style relative position bias (learned embedding over pairwise distance, clamped to `max_rel_dist=200`) |
| Attention mask | N/A | Dynamic chunked attention mask (see §1.3) |
| Directionality | B2T '24: bidirectional; B2T '25: unidirectional | Always unidirectional (causal), enforced via chunked attention |

### 1.3 Chunked Attention (Transformer only)

The Transformer uses dynamic chunked left-context attention for streaming inference. A `ChunkConfig` specifies a chunk size and number of left-context chunks to attend to. At training time, a `ChunkConfigSampler` randomly draws a config each forward pass; at eval time, a fixed eval config is used.

| Parameter | B2T '24 | B2T '25 |
|---|---|---|
| `chunkwise_prob` | 1.0 (always chunked) | 1.0 (always chunked) |
| `chunk_size_min/max` | 1 / 1 (chunk size fixed to 1 patch) | 1 / 1 (chunk size fixed to 1 patch) |
| `left_constrain_prob` | 1.0 | 1.0 |
| `context_sec_min/max` | 5 – 20 s | 5 – 20 s |
| `timestep_duration_sec` | 0.1 s/patch (5 samples/patch) | 0.08 s/patch (4 samples/patch) |
| Eval chunk size | 1 | 1 |
| Eval context | 12.5 s | 20 s |

### 1.4 Weight Initialization

| Aspect | GRU | Transformer |
|---|---|---|
| Recurrent weights | Orthogonal (`weight_hh`), Xavier uniform (`weight_ih`) | N/A |
| Initial hidden state | Learnable parameter `h0` (Xavier uniform init, expanded per batch) | N/A |
| Day weights | Identity matrix per day | Identity matrix per day (when `nDays` is set) |
| Output projection | Xavier uniform | Default PyTorch init |

### 1.5 Output

Both models project to `nClasses + 1` logits (the extra class is the CTC blank token) via a final linear layer.

---

## 2. Model Hyperparameters (from configs)

### 2.1 B2T '24

| Hyperparameter | GRU (`gru_b2t_24_baseline`) | Transformer (`neurips_b2t_24_chunked_transformer`) |
|---|---|---|
| Input features | 256 | 256 |
| Hidden/model dim | 1024 units | 384 (`6 heads × 64 dim_head`) |
| Layers / depth | 5 GRU layers | 5 Transformer layers |
| Directionality | Bidirectional | Unidirectional (causal chunked) |
| Kernel / patch size | `kernelLen=32`, `strideLen=4` | `samples_per_patch=5` |
| Day-specific transform | 24 days | None |
| `nDays` | 24 | null |
| Dropout | 0.40 | 0.35 |
| Input dropout | 0.0 | 0.2 |

### 2.2 B2T '25

| Hyperparameter | GRU (`gru_b2t_25_baseline`) | Transformer (`neurips_b2t_25_chunked_transformer`) |
|---|---|---|
| Input features | 512 | 512 |
| Hidden/model dim | 768 units | 384 (`6 heads × 64 dim_head`) |
| Layers / depth | 5 GRU layers | 5 Transformer layers |
| Directionality | Unidirectional | Unidirectional (causal chunked) |
| Kernel / patch size | `kernelLen=14`, `strideLen=4` | `samples_per_patch=4` |
| Day-specific transform | 45 days | None |
| `nDays` | 45 | null |
| Dropout | 0.4 | 0.35 |
| Input dropout | 0.2 | 0.2 |

---

## 3. Sequence Length Computation

| Model | Formula |
|---|---|
| GRU | `floor((X_len - kernelLen) / strideLen + 1)` |
| Transformer | `ceil(X_len / samples_per_patch)` |

The GRU formula can shorten the sequence significantly (especially with `kernelLen=32`), while the Transformer ceiling formula is more conservative. The input is also zero-padded to be divisible by `samples_per_patch` before the Transformer processes it.

---

## 4. Training Configuration

### 4.1 B2T '24

| Setting | GRU | Transformer |
|---|---|---|
| Optimizer | Adam | AdamW |
| Learning rate | 0.02 | 0.001 |
| `eps` | 0.1 | 1e-8 |
| L2 decay | 1e-5 | 1e-5 |
| LR scheduler | None | Multistep (decay ×0.1 at epoch 150, `lr_scaling_factor=0.1`) |
| Epochs | 73 | 250 |
| Grad norm clip | Disabled (−1) | Disabled (−1) |
| Normalize CTC len | True | True |
| `normalize_ctc_len` warmup steps | 0 | 0 |

### 4.2 B2T '25

| Setting | GRU | Transformer |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | 0.005 | 0.002 |
| `eps` | 0.1 | 1e-8 |
| L2 decay | 0.001 | 1e-5 |
| LR scheduler | Cosine (`lr_scaling_factor=0.02`, decay over 120,000 steps, warmup 1,000 steps) | Multistep (decay ×0.1 at epoch 300, `lr_scaling_factor=0.1`, warmup 10 steps) |
| Epochs | 950 | 500 |
| Grad norm clip | 10 | 100 |
| Normalize CTC len | False | True |

---

## 5. Data Augmentation

### 5.1 Noise augmentation (both models, applied in trainer)

| Augmentation | B2T '24 GRU | B2T '24 Transformer | B2T '25 GRU | B2T '25 Transformer |
|---|---|---|---|---|
| White noise SD | 0.8 | 0.2 | 1.0 | 0.2 |
| Constant offset SD | 0.2 | 0.05 | 0.2 | 0.05 |
| Gaussian smooth width | 2.0 | 2.0 | 2.0 | 2.0 |
| Smooth kernel size | 20 | 20 | 100 | 20 |
| `random_cut` | 0 | 0 | 3 (randomly trim 0–2 steps from start) | 0 |

### 5.2 Time masking (SpecAugment-style, applied inside model)

| Parameter | B2T '24 GRU | B2T '24 Transformer | B2T '25 GRU | B2T '25 Transformer |
|---|---|---|---|---|
| `num_masks` | 0 (disabled) | 20 | 0 (disabled) | 20 |
| `max_mask_pct` | 0.0 | 0.075 | 0.0 | 0.05 |

**Note on masking location:** For the GRU, time masking is applied to the raw neural input before the day transform. For the Transformer, it is applied after the initial patch rearrangement but before the linear projection and second LayerNorm.

---

## 6. Multi-Participant Support

| Aspect | GRU | Transformer |
|---|---|---|
| Multiple participants | Not supported natively; `shared_input=True` can disable per-day weights | Supported via `num_participants` and a separate `patch_embedder` per participant |
| Forward pass | `model.forward(X, X_len, day_idx)` | `model.forward(X, X_len, participant_idx, day_idx)` |
| Batch constraint | Mixed days in a batch is fine | All samples in a batch must belong to the same participant |

---

## 7. Summary of Key Differences

| Dimension | GRU | Transformer |
|---|---|---|
| Inductive bias | Sequential, local recurrence | Global (within chunk) self-attention |
| Causality | B2T '24 bidirectional; B2T '25 unidirectional | Always causal (dynamic chunked attention) |
| Input representation | Sliding window concatenation (unfolding) | Non-overlapping patch embedding with normalization |
| Positional information | Implicit via recurrence | Learned T5-style relative position bias |
| Day transforms | Always applied + softsign | Not used (`nDays=null`, `day_softsign=False`) |
| Optimizer (B2T '24) | Adam, LR 0.02, no scheduler | AdamW, LR 0.001, multistep scheduler |
| Optimizer (B2T '25) | AdamW, LR 0.005, cosine scheduler | AdamW, LR 0.002, multistep scheduler |
| Training epochs | Short (73) for '24, long (950) for '25 | Medium (250 for '24, 500 for '25) |
| Noise augmentation | Higher white noise SD | Lower white noise SD |
| Time masking | Disabled | Enabled (20 masks) |
| Multi-participant | Single participant per model | Multiple participants, one patch embedder each |
