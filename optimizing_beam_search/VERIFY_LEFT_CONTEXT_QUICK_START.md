# Left Context Verification - Quick Start Guide

## Problem Statement

Your model with chunked attention and left context restrictions is not training. You need to verify that the implementation in `transformer_chunking_lc_time.py` is correctly:

1. Reading the chunked attention config from `custom_config.yaml`
2. Calculating context windows from time-based settings
3. Creating proper attention masks
4. Restricting attention according to left context limits

## Quick Verification (5 minutes)

### Step 1: Run the verification script

```bash
cd /home/ebrahim/brainaudio
bash verify_all.sh
```

This will:
- ✓ Test core left context logic
- ✓ Test model integration
- ✓ Report any issues found

### Step 2: Interpret Results

**If all tests pass** ✓
- The left context implementation is correct
- The training issue is elsewhere (trainer loop, data pipeline, learning rate, etc.)
- See "If Tests Pass But Training Fails" below

**If tests fail** ✗
- There's a bug in the chunking implementation
- See specific test failure output for which component is broken
- See `LEFT_CONTEXT_TESTING.md` for detailed troubleshooting

## Individual Test Scripts

If you want to run individual diagnostics:

### Test 1: Core Logic Only
```bash
python scripts/verify_left_context.py
```

Tests:
- ChunkConfigSampler initialization
- Left context restriction in attention masks
- Context calculation from time-based settings
- Full context mode (no chunking)
- Unlimited left context mode
- Seed reproducibility

### Test 2: Model Integration
```bash
cd scripts
python debug_chunk_config.py
```

Tests:
- Model loads config correctly
- Sampler initializes from config
- Forward pass samples chunk configs
- Training vs evaluation configs differ
- Chunk configs are recorded correctly

## What These Tests Verify

### ChunkConfigSampler Logic

**Config from `custom_config.yaml`:**
```yaml
chunk_size_min: 1
chunk_size_max: 20
context_sec_min: 3
context_sec_max: 20
timestep_duration_sec: 0.08
chunkwise_prob: 1.0          # Always chunk
left_constrain_prob: 1.0     # Always constrain context
```

**What sampler does:**
1. Samples a random chunk size between 1-20
2. Samples a random context time between 3-20 seconds
3. Converts context time to context chunks: `ceil(context_sec / timestep_duration_sec / chunk_size)`
4. Returns ChunkConfig with chunk_size and context_chunks

**Example:**
- Samples: chunk_size=5, context_sec=4.0
- Calculation: `ceil(4.0 / 0.08 / 5) = ceil(10) = 10`
- Result: Each query can see 10 past chunks + current chunk = 11 chunks = 4.4 seconds

### Attention Mask Calculation

**What the mask does:**
- For each query position, determines which key positions it can attend to
- Respects causality (no future attention)
- Respects left context constraint (limited past context)

**Example with chunk_size=5, context_chunks=2:**
```
Query in chunk 4 (positions 20-24) can attend to:
  - Chunk 2 (10-14)   ✓
  - Chunk 3 (15-19)   ✓
  - Chunk 4 (20-24)   ✓
  - Chunk 5 (25-29)   ✗ (future - causality)
  - Earlier chunks    ✗ (beyond context)
```

### Evaluation Mode

**Eval config from `custom_config.yaml`:**
```yaml
eval:
  chunk_size: 4
  context_sec: 3.2
```

**During evaluation:**
- Fixed chunk_size: 4
- Calculated context_chunks: `ceil(3.2 / 0.08 / 4) = 10`
- Every query sees exactly 10 past chunks + current = 44 patches = 3.52 seconds

## Key Implementation Details

**File**: `src/brainaudio/models/transformer_chunking_lc_time.py`

### ChunkConfigSampler (lines 65-165)
- Takes time-based ranges (seconds), not patch counts
- Converts context_sec to context_chunks internally
- Respects probability settings (chunkwise_prob, left_constrain_prob)

### create_dynamic_chunk_mask() (lines 175-280)
- Creates (1, 1, T, T) boolean mask
- True = attend, False = mask
- Checks: (key_chunk >= lower_bound) AND (key_chunk <= upper_bound)

### Attention.forward() (lines 319-322)
```python
if temporal_mask is not None:
    dots = dots.masked_fill(temporal_mask == 0, float('-inf'))
```
- Applies mask before softmax
- Masked positions become -inf, softmax = 0

### TransformerModel (lines 350-523)
- Initializes sampler in `_setup_chunked_attention()`
- Samples config in `forward()` before calling transformer
- Stores config for debugging in `self._last_chunk_config`

## If Tests Pass But Training Fails

The left context implementation is correct. Debug these instead:

### 1. Check trainer loop
```bash
# Look for:
# - Is model being called with right input shapes?
# - Are losses being computed?
# - Are gradients flowing?

grep -n "forward(" scripts/hpo_trainer.py
```

### 2. Check if gradients are actually computed
```python
# Add to your trainer:
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.6f}")
    else:
        print(f"{name}: NO GRADIENT")
```

### 3. Check learning rate and optimization
```yaml
# In custom_config.yaml:
learning_rate: 0.001      # Is this reasonable?
learning_rate_warmup_steps: 1000
learning_rate_decay_steps: 120000
```

### 4. Check data loading
```bash
# Verify data is being loaded and preprocessed correctly
python notebooks/check_dataset.ipynb
```

### 5. Check for NaNs early in training
- Monitor loss, gradient norms, activation values
- If NaN appears: might be attention mask issue even if tests pass

## If Tests Fail

### Most Common Issues

**Issue 1: "Left Context Restriction test FAILED"**
- Mask not respecting chunk boundaries
- Check: `create_dynamic_chunk_mask()` around line 250
- Look for: Incorrect lower_bound/upper_bound calculation

**Issue 2: "ChunkConfigSampler test FAILED"**  
- Probability settings not working
- Check: `sample()` method around line 140
- Look for: Incorrect probability checks or sampling logic

**Issue 3: "Eval Config Calculation FAILED"**
- Time-to-chunks conversion wrong
- Check: `_build_chunk_config()` around line 400
- Look for: Incorrect formula or missing timestep_duration_sec

**Issue 4: Model won't even instantiate**
- Config dict structure might be wrong
- Check: Config keys match what `_setup_chunked_attention()` expects
- Look at: Line 410 for required keys

## File Locations

```
/home/ebrahim/brainaudio/
├── scripts/
│   ├── verify_left_context.py          # Run this first
│   ├── debug_chunk_config.py           # Run this second
│   ├── custom_config.yaml              # Chunking config
│   └── hpo_trainer.py                  # Where model is trained
├── src/brainaudio/models/
│   └── transformer_chunking_lc_time.py # The implementation
├── LEFT_CONTEXT_TESTING.md             # Detailed guide
└── verify_all.sh                       # One-command verification
```

## Next Steps

1. **Run verification**: `bash verify_all.sh`
2. **Interpret results**: Check which tests pass/fail
3. **If pass**: Debug trainer loop instead
4. **If fail**: Check specific code section from error message
5. **Reference**: `LEFT_CONTEXT_TESTING.md` for detailed troubleshooting

## Support

If you get stuck:

1. Check output of `verify_left_context.py` for exact failure point
2. Look up that failure in `LEFT_CONTEXT_TESTING.md` troubleshooting section
3. The markdown points to exact line numbers in the code
4. Tests are designed to isolate issues to specific components

Good luck! 🚀
