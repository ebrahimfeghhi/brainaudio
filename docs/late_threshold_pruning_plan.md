# Late Threshold Pruning Plan

## Problem
Currently, threshold pruning happens BEFORE word N-gram LM scoring. This means beams can be pruned based purely on acoustic scores, even if the LM would have boosted them significantly.

**Example scenario:**
- Beam A: high acoustic score, emits word boundary for the first time
- Beam B: slightly lower acoustic score, was already at word boundary
- Both have same transcript hash and last_label after this frame
- If B survives pruning (better acoustic) but A doesn't, the word never gets LM scored because B's `prev_last_label` was already at boundary (`was_not_boundary=False`)

## Solution
Move threshold pruning to AFTER `apply_word_ngram_lm_scoring()`.

### Current Flow
```
1. topk() - select beam_size candidates
2. THRESHOLD PRUNING - mark low-scoring candidates as INACTIVE
3. add_results_() - update beam hypotheses
4. apply_word_ngram_lm_scoring() - score word completions
5. recombine_hyps_() - merge equivalent beams
```

### Proposed Flow
```
1. topk() - select beam_size candidates
2. add_results_() - update beam hypotheses (ALL candidates)
3. apply_word_ngram_lm_scoring() - score word completions
4. recombine_hyps_() - merge equivalent beams
5. THRESHOLD PRUNING - prune based on combined acoustic+LM scores
```

## Implementation

### Step 1: Add pruning method to BatchedBeamHyps
**File:** `src/brainaudio/inference/decoder/batched_beam_decoding_utils.py`

```python
def apply_threshold_pruning_(self, beam_threshold: float):
    """Prune beams that fall below threshold after LM scoring."""
    INACTIVE = float('-inf')
    for b in range(self.batch_size):
        batch_scores = self.scores[b]
        valid_mask = batch_scores != INACTIVE
        if valid_mask.any():
            max_score = batch_scores[valid_mask].max()
            threshold = max_score - beam_threshold
            self.scores[b] = torch.where(
                valid_mask & (batch_scores <= threshold),
                self.INACTIVE_SCORE_TENSOR,
                batch_scores
            )
```

### Step 2: Modify CTC beam search loop
**File:** `src/brainaudio/inference/decoder/ctc_batched_beam_decoding.py`

Remove early pruning block (lines 406-425) and add late pruning:

```python
# Remove this block:
# batch_next_scores = next_scores.view(...)
# max_next_score = batch_next_scores.max(...)
# batch_next_scores.masked_fill_(...)
# inactive_candidate_mask = ...
# if inactive_candidate_mask.any(): ...

# Keep the flow as:
batched_beam_hyps.add_results_(next_indices, next_labels, next_scores)
apply_word_ngram_lm_scoring(...)
batched_beam_hyps.recombine_hyps_()

# Add late threshold pruning (only when word LM is active)
if self.word_ngram_lm is not None:
    batched_beam_hyps.apply_threshold_pruning_(self.beam_threshold)
```

## Files to Modify
1. `src/brainaudio/inference/decoder/batched_beam_decoding_utils.py` - add `apply_threshold_pruning_()` method
2. `src/brainaudio/inference/decoder/ctc_batched_beam_decoding.py` - remove early pruning, add late pruning call

## Trade-offs

| Aspect | Early Pruning (Current) | Late Pruning (Proposed) |
|--------|-------------------------|-------------------------|
| LM scoring work | Less (only surviving beams) | More (all top-k beams) |
| Pruning quality | Acoustic-only decisions | Acoustic+LM decisions |
| Potential WER | Worse (good words pruned early) | Better (LM can rescue) |
| Performance | Faster | Slightly slower |

## Verification
1. Run existing tests to ensure no regressions
2. Test with and without word N-gram LM to verify both paths work
3. Compare WER on validation set with early vs late pruning
