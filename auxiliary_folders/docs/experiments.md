# Experiments

## Day-Specific Linear Layer in Transformer

**Question:** Does adding a per-session affine transform (analogous to the GRU's day-specific transform) improve Transformer performance?

**Where it's applied:** On the raw neural input `(B, T, 256)` before `pad_to_multiple` and patchification. The transform operates in channel space (`256 × 256`) and is initialized to identity, so it starts as a no-op. Optionally followed by Softsign.

**Config base:** `neurips_b2t_24_chunked_transformer.yaml`
- `modelType`: transformer
- `chunked_attention`: unidirectional, chunk_size=1, context_sec 5–20s
- `d_model`: 384, `depth`: 5, `n_heads`: 6, `dim_head`: 64
- `samples_per_patch`: 5, `num_features`: 256
- Seeds: 0–9 (10 seeds each)

### Runs

| Run | `nDays` | `day_softsign` | `modelName` | Status |
|---|---|---|---|---|
| Transformer (day linear, no softsign) | 24 | false | `neurips_b2t_24_chunked_unidirectional_day_specific_transformer` | running (GPU 2) |
| Transformer (day linear + softsign) | 24 | true | `neurips_b2t_24_chunked_unidirectional_day_specific_softsign_transformer` | running (GPU 3) |
