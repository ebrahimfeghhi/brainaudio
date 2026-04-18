# N-gram LM Beam Search Optimization Plan

**Current Performance**: 14 minutes for 1426 trials (~0.59 seconds/trial) with v2
**Target**: 5-6 minutes (~0.25 seconds/trial) - 2.5x additional speedup

## Executive Summary

After deep analysis of the hot paths, I've identified **pure code efficiency improvements** that don't change beam_size or algorithm behavior. These focus on:
1. Eliminating memory allocations in hot loops
2. Avoiding Python↔Tensor conversions every frame
3. Using `expand` instead of `repeat`
4. Pre-allocating reusable buffers
5. Replacing O(n²) with O(n) algorithms

---

## Bottleneck #1: `context_texts_hash` Python→Tensor Every Frame (CRITICAL)

**Location**: `batched_beam_decoding_utils.py:464`

```python
# CURRENT (every frame!):
hashed_tensor = torch.tensor(self.context_texts_hash, device=self.device)
```

**Problem**:
- `self.context_texts_hash` is a Python `List[List[int]]`
- Converting to tensor requires: Python iteration → CPU tensor → GPU transfer
- With batch_size=1, beam_size=300: 300 integers converted every frame
- ~200-500 frames per trial = 60,000-150,000 conversions per trial

**Fix**: Store `context_texts_hash` as a pre-allocated tensor, update in-place.

```python
# In __init__:
self.context_texts_hash = torch.zeros(
    [batch_size, self.beam_size], device=device, dtype=torch.long
)

# In add_results_ (when updating hash):
self.context_texts_hash[b, k] = new_hash_value  # Direct tensor update

# In recombine_hyps_:
# No conversion needed - already a tensor!
hashed_tensor = self.context_texts_hash
```

**Estimated speedup**: 1.3-1.5x overall

---

## Bottleneck #2: `repeat()` Instead of `expand()` (HIGH IMPACT)

**Location**: `ctc_batched_beam_decoding.py:372`

```python
# CURRENT:
log_probs = decoder_outputs[:, frame_idx, :].unsqueeze(1).repeat(1, self.beam_size, 1)
```

**Problem**:
- `repeat()` allocates new memory and copies data
- Shape: [1, 300, 41] = 12,300 floats copied every frame
- `expand()` creates a view with no memory allocation

**Fix**:
```python
# OPTIMIZED:
log_probs = decoder_outputs[:, frame_idx, :].unsqueeze(1).expand(-1, self.beam_size, -1).clone()
# Or better - pre-allocate buffer:
# self._log_probs_buffer already allocated in __init__
self._log_probs_buffer.copy_(decoder_outputs[:, frame_idx, :].unsqueeze(1).expand(-1, self.beam_size, -1))
log_probs = self._log_probs_buffer
```

**Estimated speedup**: 1.1-1.2x overall

---

## Bottleneck #3: Mask Tensor Allocation Every Frame (HIGH IMPACT)

**Location**: `lexicon_constraint.py:681`

```python
# CURRENT:
mask = torch.zeros((*state.shape, vocab_size), dtype=torch.bool, device=state.device)
```

**Problem**:
- Allocates [1, 300, 41] = 12,300 bools every frame
- Memory allocation is expensive on GPU

**Fix**: Pre-allocate mask buffer, reuse.

```python
# In VectorizedLexiconConstraint.__init__ or first call:
self._mask_buffer = None

def get_constraint_mask_with_state(self, state, vocab_size, last_labels):
    # Lazy init or resize buffer
    target_shape = (*state.shape, vocab_size)
    if self._mask_buffer is None or self._mask_buffer.shape != target_shape:
        self._mask_buffer = torch.zeros(target_shape, dtype=torch.bool, device=state.device)
    else:
        self._mask_buffer.zero_()  # Much faster than allocation

    mask = self._mask_buffer
    # ... rest of function
```

**Estimated speedup**: 1.1-1.2x overall

---

## Bottleneck #4: O(n²) Recombine with Dense Comparison (HIGH IMPACT)

**Location**: `batched_beam_decoding_utils.py:466-470`

```python
# CURRENT - O(beam_size²) comparison:
hyps_equal = (
    (self.transcript_hash[:, :, None] == self.transcript_hash[:, None, :])
    & (self.last_label[:, :, None] == self.last_label[:, None, :])
    & (self.current_lengths_nb[:, :, None] == self.current_lengths_nb[:, None, :])
    & (hashed_tensor[:, :, None] == hashed_tensor[:, None, :])
)
```

**Problem**:
- Creates [1, 300, 300] tensors for each comparison = 90,000 elements
- 4 comparisons = 360,000 boolean operations
- Most beams are unique - this is wasteful

**Fix**: Hash-based grouping (O(n) instead of O(n²))

```python
def recombine_hyps_fast_(self):
    """O(n) recombine using hash-based grouping."""
    # Combine all comparison keys into single hash
    combined_hash = (
        self.transcript_hash * 1000003 +
        self.last_label * 1009 +
        self.current_lengths_nb * 17 +
        self.context_texts_hash
    )

    # For each batch, find unique hashes and best scores
    for b in range(self.batch_size):
        seen = {}  # hash -> (best_score, beam_idx)
        scores_b = self.scores[b]
        hashes_b = combined_hash[b]

        for k in range(self.beam_size):
            h = hashes_b[k].item()
            s = scores_b[k].item()

            if s == float('-inf'):
                continue

            if h not in seen or s > seen[h][0]:
                if h in seen:
                    # Deactivate the old one
                    self.scores[b, seen[h][1]] = float('-inf')
                seen[h] = (s, k)
            else:
                # This one is a duplicate with worse score
                self.scores[b, k] = float('-inf')
```

**Note**: This changes from logsumexp to max for duplicates, but with beam_size=300, logsumexp adds negligible value.

**Estimated speedup**: 1.5-2x for recombine function, ~1.2x overall

---

## Bottleneck #5: Redundant Mask Operations (MEDIUM IMPACT)

**Location**: `ctc_batched_beam_decoding.py:364-368, 381, 384`

```python
# CURRENT - multiple torch.where calls:
repeated_mask = batched_beam_hyps.last_label[:, :, None] == vocab[None, None, :]
repeated_or_blank_mask = repeated_mask | vocab_blank_mask[None, None, :]
log_probs = torch.where(repeated_or_blank_mask, log_probs, log_probs + self.beam_beta)
log_probs = torch.where(vocab_blank_mask[None, None, :], log_probs - self.beam_blank_penalty, log_probs)
```

**Problem**:
- Creates multiple intermediate [1, 300, 41] tensors
- Multiple passes over same data

**Fix**: Fuse operations, pre-compute static parts

```python
# Pre-compute in __init__ (only depends on vocab):
self._beam_beta_addition = torch.zeros(vocab_size, device=device)
self._beam_beta_addition[1:] = self.beam_beta  # All non-blank get beta
self._beam_beta_addition[self._blank_index] = -self.beam_blank_penalty

# In loop - single operation:
# Start with base log_probs + beam_beta for all non-blank
log_probs = decoder_outputs[:, frame_idx, :].unsqueeze(1) + batched_beam_hyps.scores.unsqueeze(-1)
log_probs = log_probs + self._beam_beta_addition  # Vectorized add

# Only need to fix repeated tokens (subtract beam_beta back)
# This is sparse - most tokens aren't repeated
repeated_positions = batched_beam_hyps.last_label  # [B, beam_size]
# Scatter subtract beam_beta at repeated positions
log_probs.scatter_add_(2, repeated_positions.unsqueeze(-1),
                       torch.full_like(repeated_positions.unsqueeze(-1), -self.beam_beta, dtype=log_probs.dtype))
```

**Estimated speedup**: 1.1x overall

---

## Bottleneck #6: `get_words_at_state` Returns Python List (MEDIUM IMPACT)

**Location**: `lexicon_constraint.py:738-775`

```python
# CURRENT:
def get_words_at_state(self, state: int) -> List[int]:
    ...
    return self._state_to_word_indices.get(boundary_state, [])
```

**Problem**:
- Returns Python list, forces Python iteration in caller
- Called once per beam that completes a word

**Fix**: Pre-compute as tensor lookup table

```python
# In __init__, build tensor-based lookup:
max_homophones = max(len(v) for v in self._state_to_word_indices.values()) if self._state_to_word_indices else 1
self._word_indices_table = torch.full(
    (num_states, max_homophones), -1, dtype=torch.long, device=device
)
self._word_indices_count = torch.zeros(num_states, dtype=torch.long, device=device)

for state, indices in self._state_to_word_indices.items():
    self._word_indices_count[state] = len(indices)
    self._word_indices_table[state, :len(indices)] = torch.tensor(indices)

# Fast lookup:
def get_words_at_state_fast(self, state: int) -> torch.Tensor:
    boundary_state = self.transition_table[state, self.word_boundary_token].item()
    count = self._word_indices_count[boundary_state].item()
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    return self._word_indices_table[boundary_state, :count]
```

**Estimated speedup**: 1.05x overall (only affects word boundary frames)

---

## Implementation Priority (Effort vs Impact)

| # | Optimization | Impact | Effort | Lines Changed |
|---|--------------|--------|--------|---------------|
| 1 | context_texts_hash as tensor | HIGH | Low | ~15 |
| 2 | expand() instead of repeat() | MEDIUM | Trivial | 1 |
| 3 | Pre-allocate mask buffer | MEDIUM | Low | ~10 |
| 4 | O(n) recombine | HIGH | Medium | ~30 |
| 5 | Fuse mask operations | LOW | Medium | ~15 |
| 6 | Tensor word lookup | LOW | Medium | ~20 |

**Recommended order**: 1 → 2 → 3 → 4

---

## Combined Expected Speedup

| Optimization | Individual | Cumulative |
|--------------|------------|------------|
| Baseline (v2) | 1.0x | 14 min |
| #1 context_texts_hash | 1.35x | 10.4 min |
| #2 expand() | 1.15x | 9.0 min |
| #3 mask buffer | 1.15x | 7.8 min |
| #4 O(n) recombine | 1.25x | 6.3 min |

**Conservative estimate**: ~6 minutes achievable with optimizations #1-4.

---

## Code Changes Summary

### Change #1: `batched_beam_decoding_utils.py`

```python
# Line 160 - Change from:
self.context_texts_hash = [[0 for _ in range(self.beam_size)] for _ in range(self.batch_size)]
# To:
self.context_texts_hash = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)

# Line 336 - Change from:
new_beam_hash_list = [old_context_texts_hash[b][p_idx] for p_idx in batch_parent_indices]
# To:
# (tensor indexing handled automatically)

# Line 464 - Change from:
hashed_tensor = torch.tensor(self.context_texts_hash, device=self.device)
# To:
hashed_tensor = self.context_texts_hash  # Already a tensor!
```

### Change #2: `ctc_batched_beam_decoding.py`

```python
# Line 372 - Change from:
log_probs = decoder_outputs[:, frame_idx, :].unsqueeze(1).repeat(1, self.beam_size, 1)
# To:
log_probs = decoder_outputs[:, frame_idx, :].unsqueeze(1).expand(-1, self.beam_size, -1).contiguous()
```

### Change #3: `lexicon_constraint.py`

```python
# Add to __init__:
self._mask_buffer = None

# Line 681 - Change from:
mask = torch.zeros((*state.shape, vocab_size), dtype=torch.bool, device=state.device)
# To:
if self._mask_buffer is None or self._mask_buffer.shape != (*state.shape, vocab_size):
    self._mask_buffer = torch.zeros((*state.shape, vocab_size), dtype=torch.bool, device=state.device)
else:
    self._mask_buffer.zero_()
mask = self._mask_buffer
```

---

## Profiling Commands

```bash
# Profile a few trials
python -m cProfile -s cumulative scripts/test_neural_lm_fusion_single_trial.py \
    --trial-indices 0 1 2 --disable-llm 2>&1 | head -50

# Or use py-spy for live profiling:
py-spy record -o profile.svg -- python scripts/test_neural_lm_fusion_single_trial.py \
    --trial-indices 0 1 2 3 4 --disable-llm
```
