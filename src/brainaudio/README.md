# brainaudio package

This is the core Python package. It is installed in editable mode via `uv sync` and imported as `brainaudio`.

---

## Package Structure

```
src/brainaudio/
├── models/          # CTC encoder architectures (GRU, Transformer)
├── training/        # Trainer, loss, augmentations, learning schedulers
├── inference/       # Logit generation and CTC beam search decoder
└── datasets/        # Dataset loading and format utilities
```

---

## `models/` — Encoder Architectures

Neural encoder models that map neural activity to per-timestep phoneme logits. All models inherit from `BaseTimeMaskedModel` and output tensors of shape `(B, T', n_classes+1)` consumed by the CTC decoder.

| File | Class | Used for |
|---|---|---|
| `base_model.py` | `BaseTimeMaskedModel` | Abstract base; provides SpecAugment-style time masking |
| `gru_b2t_24.py` | `GRU_24` | GRU encoder for B2T '24 |
| `gru_b2t_25.py` | `GRU_25` | GRU encoder for B2T '25 |
| `transformer_chunking.py` | `TransformerModel` | Chunked causal Transformer with day-specific layers (used in paper) |
| `transformer.py` | `TransformerModel` | Base Transformer without chunking |

See [`models/README.md`](models/README.md) for full constructor parameters and architecture details.

---

## `training/` — Training Infrastructure

| File | Description |
|---|---|
| `trainer.py` | Main training loop (`trainModel`): CTC loss, gradient clipping, checkpointing, W&B logging |
| `utils/loss.py` | CTC loss with optional length normalization |
| `utils/augmentations.py` | Gaussian smoothing and other input augmentations |
| `utils/learning_scheduler.py` | Learning rate schedulers (cosine, multistep, none) |
| `utils/custom_configs/` | YAML training configs for all paper models |

---

## `inference/` — Logit Generation & Decoding

| File | Description |
|---|---|
| `load_model_generate_logits.py` | Load a trained encoder from disk and run forward passes to produce logit `.npz` files |
| `eval_metrics.py` | PER / WER computation |
| `lm_funcs.py` | N-gram LM utilities used during decoding |
| `forced_alignments.py` | Forced alignment utilities |
| `decoder/` | Lightbeam CTC beam search decoder with n-gram and LLM scoring |

See [`inference/decoder/README.md`](inference/decoder/README.md) for a walkthrough of the beam search algorithm.

---

## `datasets/` — Dataset Utilities

Utilities for loading the trial-level pickle format produced by `scripts/dataset/`. See [`datasets/README.md`](datasets/README.md) for the expected data format.
