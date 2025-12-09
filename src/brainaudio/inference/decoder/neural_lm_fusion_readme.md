# neural_lm_fusion.py

## Overview

The function which applies LLM-based shallow fusion inside `ctc_batched_beam_decoding.py` is `_apply_lm_fusion`. 

### `_apply_lm_fusion`

* **Inputs:**
    * `log_probs`: [B, beam_size, V] current log probabilities
    * `beam_hyps`: BatchedBeamHyps containing current sequences
    * `curr_batch_size`: Current batch size
    * `token_to_symbol`: Optional dict mapping token ID to phoneme symbol

* **Logic:**
    1. For each beam within each batch, retrieves CTC sequence and applies CTC merging rules. 
    2. Calls `lexicon.get_valid_next_tokens_with_word_indices` on the merged sequence, which returns 
    the next possible valid tokens, whether we are at a word boundary, and word indices. 
    3. If beam is at a word boundary but there are no valid words, set log probs to -INF. I believe this case
    should have already been applied because we apply the lexicon mask to log probs before calling this function,
    but I'll leave it in there for now because I don't think it has unintended side effects. 
    4. If the beam is at a word boundary, we call `decode_sequence_to_text` (from `beam_helpers.py`) on that beam, excluding the last word.
    This provides the prefix context. 
    5. If there are beams which are at a word boundary within a batch, we call score_continuations to generate scores for each candidate word for each beam. 
    6. 


