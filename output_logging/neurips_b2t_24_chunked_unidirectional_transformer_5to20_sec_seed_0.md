# Args: neurips_b2t_24_chunked_unidirectional_transformer_5to20_sec_seed_0

## Training

| Parameter | Value |
|---|---|
| `modelType` | transformer |
| `optimizer` | AdamW |
| `learning_rate` | 0.001 |
| `learning_rate_min` | 0.0001 |
| `beta1` | 0.9 |
| `beta2` | 0.999 |
| `eps` | 1e-08 |
| `learning_scheduler` | multistep |
| `milestones` | [150] |
| `gamma` | 0.1 |
| `lr_scaling_factor` | 0.1 |
| `learning_rate_decay_steps` | 951 |
| `learning_rate_warmup_steps` | 0 |
| `n_epochs` | 250 |
| `batchSize` | 64 |
| `nClasses` | 40 |
| `grad_norm_clip_value` | -1 |
| `l2_decay` | 1e-05 |
| `use_amp` | true |
| `seed` | 0 |

## Augmentation

| Parameter | Value |
|---|---|
| `num_masks` | 20 |
| `max_mask_pct` | 0.075 |
| `whiteNoiseSD` | 0.2 |
| `constantOffsetSD` | 0.05 |
| `gaussianSmoothWidth` | 2.0 |
| `smooth_kernel_size` | 20 |
| `random_cut` | 0 |
| `input_dropout` | 0.2 |
| `dropout` | 0.35 |

## Transformer Architecture

| Parameter | Value |
|---|---|
| `d_model` | 384 |
| `depth` | 5 |
| `n_heads` | 6 |
| `dim_head` | 64 |
| `mlp_dim_ratio` | 4 |
| `samples_per_patch` | 5 |
| `features_list` | [256] |

### Chunked Attention

| Parameter | Value |
|---|---|
| `chunkwise_prob` | 1.0 |
| `chunk_size_min` | 1 |
| `chunk_size_max` | 1 |
| `left_constrain_prob` | 1.0 |
| `context_sec_min` | 5 |
| `context_sec_max` | 20 |
| `timestep_duration_sec` | 0.1 |
| `eval.chunk_size` | 1 |
| `eval.context_sec` | null |

## GRU Architecture (reference, not used)

| Parameter | Value |
|---|---|
| `nInputFeatures` | 256 |
| `nUnits` | 1024 |
| `nLayers` | 5 |
| `bidirectional` | true |
| `strideLen` | 4 |
| `kernelLen` | 4 |
| `nDays` | 24 |

## Decoding

| Parameter | Value |
|---|---|
| `acoustic_scale` | 0.6 |
| `lm_weight` | 2.0 |
| `word_score` | 0.1 |
| `beam_size` | 30 |

### InterCTC

| Parameter | Value |
|---|---|
| `enable_interctc` | false |
| `alpha` | 0.25 |
| `inter_ctc_per_layers` | 3 |

## Early Stopping

| Parameter | Value |
|---|---|
| `early_stopping_enabled` | false |
| `early_stopping_checkpoint` | 951 |
| `early_stopping_wer_threshold` | 1 |
| `early_stopping_per_threshold` | 1 |
| `early_stopping_no_improvement` | 95 |
