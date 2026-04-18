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

## `gru_b2t_24.py` — `GRU_24`

GRU-based encoder introduced in Willett et al. 2023 for the B2T '24 benchmark.

### `GRU_24` — Day-specific input layer

The standard encoder. Neural activity first passes through a **per-day affine transform** (separate weight matrix and bias for each recording session), followed by a Softsign nonlinearity. The transformed signal is then temporally unfolded into overlapping patches and fed into a multi-layer GRU.

**Constructor parameters**

| Parameter | Description |
|---|---|
| `neural_dim` | Number of input neural channels (256 for B2T '24) |
| `n_classes` | Number of phoneme classes (excluding CTC blank) |
| `hidden_dim` | GRU hidden state size |
| `layer_dim` | Number of stacked GRU layers |
| `nDays` | Number of recording sessions (one affine transform per day) |
| `dropout` | Dropout between GRU layers |
| `input_dropout` | Dropout applied to inputs after the day-specific transform |
| `strideLen` | Stride of the temporal unfolding window |
| `kernelLen` | Kernel size of the temporal unfolding window |
| `bidirectional` | If `True`, use a bidirectional GRU |
| `max_mask_pct` | Time masking fraction (inherited) |
| `num_masks` | Number of time masks (inherited) |

**Forward:** `(neuralInput, X_len, dayIdx) → (B, T', n_classes+1)`

Output length: `T' = (T - kernelLen) / strideLen`

---

## `gru_b2t_25.py` — `GRU_25`

GRU-based encoder for the B2T '25 benchmark. The architecture follows `GRU_24` but with two differences: day weights and biases are stored as a `ParameterList` (one matrix per day) rather than a single stacked tensor, and the initial hidden state `h0` is a learnable parameter rather than a fixed zero initialization.

**Key differences from `GRU_24`**

| | `GRU_24` | `GRU_25` |
|---|---|---|
| Day weights storage | Stacked `nn.Parameter` tensor | `nn.ParameterList` (one per day) |
| Initial hidden state | Zero tensor | Learnable `h0` parameter |
| `samples_per_patch` | Hardcoded to 1 | Configurable (default 1) |

**Forward:** `(x, x_len, day_idx) → (B, T', n_classes+1)`

Output length: `T' = (T - kernelLen) / strideLen + 1`

---

## `transformer.py` — `TransformerModel`

ViT-style causal Transformer encoder. Neural activity is divided into non-overlapping 1D patches, embedded via a patch embedding module (LayerNorm → Linear → LayerNorm), and processed by a stack of self-attention + FFN layers with T5-style relative position bias.

**Constructor parameters**

| Parameter | Description |
|---|---|
| `samples_per_patch` | Number of time steps per patch |
| `num_features` | Number of input channels |
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
| `return_final_layer` | If `True`, also return the final Transformer hidden states |
| `bidirectional` | If `False`, a causal attention mask is applied (each patch attends only to itself and earlier patches) |

**Forward:** `(neuralInput, X_len, day_idx=None) → (B, P, n_classes+1)`

Output length: `P = ceil(T / samples_per_patch)`

---

## `transformer_chunking.py` — `TransformerModel` (chunked)

Extension of the Transformer encoder that adds **dynamic chunked left-context attention** for streaming / low-latency inference. During training, the chunk size and left-context window are sampled from configurable ranges each forward pass, simulating a range of latency budgets. At evaluation, a fixed chunk config can be used.

Also supports an optional **day-specific affine transform** (controlled by `nDays` and `day_softsign`) and **multi-participant** training via per-participant patch embedders.

**Additional constructor parameters** (on top of the base Transformer)

| Parameter | Description |
|---|---|
| `num_participants` | Number of participants; one patch embedder is created per participant |
| `features_list` | List of input channel counts, one per participant |
| `nDays` | If set, adds a per-day affine transform before patch embedding |
| `day_softsign` | If `True`, applies Softsign after the day-specific transform |
| `chunked_attention` | Dict configuring the chunk sampler and eval chunk config (see below) |

**`chunked_attention` config dict keys**

| Key | Description |
|---|---|
| `chunk_size_min` / `chunk_size_max` | Range of chunk sizes (in patches) sampled during training |
| `context_sec_min` / `context_sec_max` | Range of left-context window sizes (in seconds) sampled during training |
| `timestep_duration_sec` | Duration of one time step in seconds (used to convert context seconds to patches) |
| `left_constrain_prob` | Probability of applying a hard left-context constraint during training |
| `chunkwise_prob` | Probability of using strict chunkwise (no left context) attention during training |
| `eval.chunk_size` | Fixed chunk size used at evaluation |
| `eval.context_sec` | Fixed left-context window size (seconds) used at evaluation; `null` = full context |

**Forward:** `(neuralInput, X_len, participant_idx=None, day_idx=None) → (B, P, n_classes+1)`
