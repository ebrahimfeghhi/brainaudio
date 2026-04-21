# Models

This package contains the neural encoder models used for Brain-to-Text decoding. All models inherit from `BaseTimeMaskedModel` and output per-timestep phoneme logits (including a CTC blank token) that are consumed by the CTC beam-search decoder.

---

## `base_model.py` — `BaseTimeMaskedModel`

Abstract base class for all encoders. Provides a shared **SpecAugment-style time masking** implementation used during training.

**Constructor parameters**

| Parameter | Description |
|---|---|
| `max_mask_pct` | Maximum fraction of the sequence to mask per mask region |
| `num_masks` | Number of independent mask regions applied per sample |
| `samples_per_patch` | Temporal resolution of one patch (used to convert sample lengths to patch lengths) |

**Key method: `apply_time_masking(X, X_len, mask_value)`**

Applies fully-vectorized random time masking to a patched input tensor `(B, P, D)`. Each sample gets `num_masks` independent contiguous regions zeroed out (or filled with `mask_value`), with each region length drawn uniformly from `[0, max_mask_pct * valid_len]`. Subclasses must implement `compute_length(X_len)` to convert raw sample lengths to output lengths.

---

## `gru.py` — `GRU`

Unified GRU-based encoder for both B2T '24 and B2T '25.

Neural activity passes through a **per-day affine transform** (separate weight matrix and bias per recording session, stored as a `ParameterList`), followed by a Softsign nonlinearity. The transformed signal is temporally unfolded into overlapping patches and fed into a multi-layer GRU. The initial hidden state `h0` is a learnable parameter.

**Constructor parameters**

| Parameter | Description |
|---|---|
| `neural_dim` | Number of input neural channels |
| `n_classes` | Number of phoneme classes (excluding CTC blank) |
| `hidden_dim` | GRU hidden state size |
| `layer_dim` | Number of stacked GRU layers |
| `nDays` | Number of recording sessions (one affine transform per day) |
| `dropout` | Dropout between GRU layers |
| `input_dropout` | Dropout applied after the day-specific transform |
| `strideLen` | Stride of the temporal unfolding window |
| `kernelLen` | Kernel size of the temporal unfolding window |
| `bidirectional` | If `True`, use a bidirectional GRU |
| `max_mask_pct` | Time masking fraction (inherited) |
| `num_masks` | Number of time masks (inherited) |
| `samples_per_patch` | Temporal resolution of one patch (default 1) |
| `shared_input` | If `True`, all days share a single input transform |

**Forward:** `(x, x_len, day_idx) → (B, T', n_classes+1)`

Output length: `T' = (T - kernelLen) / strideLen + 1`

---

## `transformer_chunking.py` — `TransformerModel` (chunked)

ViT-style causal Transformer encoder with **dynamic chunked left-context attention** for streaming / low-latency inference. During training, the chunk size and left-context window are sampled from configurable ranges each forward pass, simulating a range of latency budgets. At evaluation, a fixed chunk config can be used.

Also supports an optional **day-specific affine transform** (controlled by `nDays` and `day_softsign`) and **multi-participant** training via per-participant patch embedders.

**Constructor parameters**

| Parameter | Description |
|---|---|
| `features_list` | List of input channel counts, one per participant |
| `samples_per_patch` | Number of time steps per patch |
| `dim` | Transformer embedding dimension |
| `depth` | Number of Transformer layers |
| `heads` | Number of attention heads |
| `dim_head` | Per-head dimension |
| `mlp_dim_ratio` | FFN hidden size as a multiple of `dim` |
| `dropout` | Attention and FFN dropout |
| `input_dropout` | Dropout applied after patch embedding |
| `nClasses` | Number of phoneme classes (excluding CTC blank) |
| `max_mask_pct` | Time masking fraction (inherited) |
| `num_masks` | Number of time masks (inherited) |
| `num_participants` | Number of participants; one patch embedder per participant |
| `return_final_layer` | If `True`, also return the final Transformer hidden states |
| `nDays` | If set, adds a per-day affine transform before patch embedding |
| `day_softsign` | If `True`, applies Softsign after the day-specific transform |
| `chunked_attention` | Dict configuring the chunk sampler and eval chunk config (see below) |

**`chunked_attention` config dict keys**

| Key | Description |
|---|---|
| `chunk_size_min` / `chunk_size_max` | Range of chunk sizes (in patches) sampled during training |
| `context_sec_min` / `context_sec_max` | Range of left-context window sizes (in seconds) sampled during training |
| `timestep_duration_sec` | Duration of one time step in seconds |
| `left_constrain_prob` | Probability of applying a hard left-context constraint during training |
| `chunkwise_prob` | Probability of using strict chunkwise attention during training |
| `eval.chunk_size` | Fixed chunk size used at evaluation |
| `eval.context_sec` | Fixed left-context window (seconds) at evaluation; `null` = full context |

**Forward:** `(neuralInput, X_len, participant_idx=None, day_idx=None) → (B, P, n_classes+1)`

Output length: `P = ceil(T / samples_per_patch)`
