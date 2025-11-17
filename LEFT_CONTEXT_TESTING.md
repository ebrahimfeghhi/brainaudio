# Left Context Restriction Verification Guide

This document explains the test files created to verify left context restrictions in `transformer_chunking_lc_time.py` and how to use them to debug training issues.

## Overview

The transformer model in `transformer_chunking_lc_time.py` implements chunked attention with optional left context restrictions. This means:

- **Normal mode**: Attention can see all past tokens (full causal attention)
- **Chunked mode**: Tokens are grouped into chunks, and each query chunk can only attend to a limited number of past chunks
- **Left context limit**: Determines how many past chunks can be seen (e.g., 2 chunks = ~1.6 seconds if chunk_size=4 and timestep=0.08s)

## Configuration from custom_config.yaml

```yaml
model:
  transformer:
    chunked_attention:
      chunkwise_prob: 1.0          # 1.0 = always use chunking
      chunk_size_min: 1            # Minimum chunk size (in patches)
      chunk_size_max: 20           # Maximum chunk size
      left_constrain_prob: 1.0     # 1.0 = always constrain left context
      context_sec_min: 3           # Minimum context window (in seconds)
      context_sec_max: 20          # Maximum context window (in seconds)
      timestep_duration_sec: 0.08  # Each patch is 80ms
      eval:
        chunk_size: 4              # Fixed chunk size during eval
        context_sec: 3.2           # Fixed context window during eval (~40 patches)
```

## Test Files

### 1. `scripts/verify_left_context.py` - Comprehensive Diagnostic

**Purpose**: Verify that the core left context logic is working correctly.

**Run it**:
```bash
cd /home/ebrahim/brainaudio
python scripts/verify_left_context.py
```

**What it tests**:
1. **ChunkConfigSampler**: Verifies sampling works correctly
   - Respects `chunkwise_prob` (0.0 always full context, 1.0 always chunks)
   - Respects `left_constrain_prob` (0.0 unlimited context, 1.0 limited)
   - Correctly calculates `context_chunks` from time-based `context_sec` setting

2. **Left Context Restriction Mask**: Verifies attention mask is created correctly
   - Tests that queries can only attend to allowed chunks
   - Verifies causality (no future attention)
   - Tests boundary cases (first chunk, edge cases)

3. **Eval Config Calculation**: Verifies evaluation config is built correctly
   - Takes `context_sec` and `chunk_size` from config
   - Calculates `context_chunks = ceil(context_sec / timestep_duration_sec / chunk_size)`

4. **Full Context Mode**: Tests disabling chunking
   - With `chunkwise_prob=0`, should always use full attention

5. **Unlimited Left Context**: Tests disabling left constraints
   - With `left_constrain_prob=0`, queries should see all past chunks (full causal)

6. **Seed Reproducibility**: Verifies deterministic sampling

**Expected output if everything is correct**:
```
✓ PASS: ChunkConfigSampler
✓ PASS: Left Context Restriction
✓ PASS: Eval Config Calculation
✓ PASS: Full Context Mode
✓ PASS: Unlimited Left Context
✓ PASS: Seed Reproducibility

✓ ALL TESTS PASSED - Left context implementation appears correct!
```

### 2. `scripts/debug_chunk_config.py` - Model Integration Test

**Purpose**: Verify the model correctly initializes and uses chunk configs.

**Run it**:
```bash
cd /home/ebrahim/brainaudio/scripts
python debug_chunk_config.py
```

**What it checks**:
1. Model instantiation with chunked attention config
2. Training sampler is initialized with correct parameters
3. Evaluation config is built correctly
4. Forward pass during training uses a sampled config
5. Forward pass during evaluation uses the fixed eval config
6. Actual chunk configs used are recorded in `model.last_chunk_config`

**Expected output if working**:
```
✓ Model created successfully
✓ Train sampler initialized
  Chunk size range: (1, 20)
  Context sec range: (3, 20)
  Timestep duration: 0.08
  Chunkwise prob: 1.0
  Left constrain prob: 1.0
✓ Eval config initialized
  Chunk size: 4
  Context chunks: 10

TRAINING MODE FORWARD PASS
✓ Forward pass successful
Chunk config used (TRAINING):
  Chunk size: 12
  Context chunks: 3

EVALUATION MODE FORWARD PASS
✓ Forward pass successful
Chunk config used (EVALUATION):
  Chunk size: 4
  Context chunks: 10
```

### 3. `tests/test_left_context.py` - Full Unit Test Suite

**Purpose**: Comprehensive unit tests using pytest (optional, more detailed)

**Run it**:
```bash
cd /home/ebrahim/brainaudio
python -m pytest tests/test_left_context.py -v -s
```

**Contains**: ~15+ test cases covering all aspects of chunking and left context logic

## Troubleshooting Guide

### If verify_left_context.py fails:

**Failure: "Left Context Restriction" test**
- Check: `create_dynamic_chunk_mask()` function in transformer_chunking_lc_time.py
- Issue: Mask calculation not respecting chunk boundaries
- Look for: Bug in chunk ID assignment or bounds calculation (lines 200-260)

**Failure: "Eval Config Calculation" test**
- Check: `_build_chunk_config()` method
- Issue: Context chunks not calculated correctly from context_sec
- Fix: Verify formula: `context_chunks = ceil(context_sec / timestep_duration_sec / chunk_size)`

**Failure: "ChunkConfigSampler" test**
- Check: `ChunkConfigSampler.sample()` method
- Issue: Probability checks not working or context calculation wrong
- Look for: Bounds checking and probability sampling logic (lines 120-165)

### If debug_chunk_config.py fails:

**Error: "Train sampler NOT initialized!"**
- Check: `_setup_chunked_attention()` method
- Issue: Sampler creation failed or not assigned to `self._train_sampler`
- Fix: Verify config is being passed correctly and no exceptions during sampler creation

**Error: "Forward pass successful" but no chunk config recorded**
- Check: `_sample_chunk_config()` method
- Issue: Not calling sampler or not storing config in `self._last_chunk_config`
- Fix: Verify `_sample_chunk_config()` is called in `forward()` and result is stored

**Output shows "Full context" during training when chunking should be enabled**
- Issue: `chunkwise_prob` might not be 1.0 or sampler has issue
- Check: Verify custom_config.yaml has `chunkwise_prob: 1.0`

### If model trains but doesn't seem to use chunking:

Run `debug_chunk_config.py` to check:
1. What chunk_size and context_chunks are being used?
2. Are they changing between batches (during training)?
3. Are they different from training during evaluation?

If everything is correct but training doesn't work:
- The issue is likely in the **attention module** itself, not the mask
- Check: Does `Attention.forward()` actually use the `temporal_mask` parameter?
- Look at line 319 in transformer_chunking_lc_time.py: `if temporal_mask is not None:`
- Verify: Mask is applied to `dots` (attention logits) before softmax

## Key Code Locations

**File**: `src/brainaudio/models/transformer_chunking_lc_time.py`

- **ChunkConfigSampler**: Lines 65-165 (sampling logic)
- **create_dynamic_chunk_mask()**: Lines 175-280 (mask creation)
- **Attention class**: Lines 286-340 (apply mask to attention)
- **TransformerModel._setup_chunked_attention()**: Lines 406-445 (initialize sampler)
- **TransformerModel.forward()**: Lines 482-523 (use mask in transformer)

## Key Formulas

**Context chunks calculation** (applies both in sampler and eval config):
```
total_context_timesteps = context_sec / timestep_duration_sec
context_chunks = ceil(total_context_timesteps / chunk_size)
```

Example with config values:
- `context_sec = 3.2` seconds
- `timestep_duration_sec = 0.08` seconds (80ms)
- `chunk_size = 4` patches
- Result: `context_chunks = ceil((3.2 / 0.08) / 4) = ceil(10) = 10`
- This means each query can see 10 past chunks + its own chunk = 11 chunks = 44 patches = 3.52 seconds

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Model trains but output is noise | Mask is full zeros (disabling all attention) | Check mask creation, verify causality constraint |
| Training diverges or NaNs early | Infinite masking or attention weights | Verify masked_fill uses correct value (should be -inf) |
| No speedup from chunking | Mask not created or not applied | Verify temporal_mask is passed and used in attention |
| Different results with same seed | Sampler seeding not working | Check ChunkConfigSampler.__init__ sets self._rng correctly |
| Eval config ignored | Eval config not initialized | Check _build_chunk_config returns proper ChunkConfig object |

## Next Steps After Verification

Once these tests pass:

1. **Check trainer integration**: Does trainer correctly pass data through model?
   - Run training with one batch and check gradients flow

2. **Monitor during training**: Add logging to see chunk configs used
   - Print `model.last_chunk_config` every N steps

3. **Profile attention**: Is chunking actually reducing computation?
   - Compare attention matrix sparsity with/without chunking

4. **Validate on test set**: Run inference with eval config and check WER

## Contact & Questions

If tests fail in unexpected ways:
1. Run `verify_left_context.py` first (tests core logic)
2. Run `debug_chunk_config.py` second (tests integration)
3. Check error messages against "Troubleshooting Guide" above
4. Look at specific code locations listed in "Key Code Locations"
