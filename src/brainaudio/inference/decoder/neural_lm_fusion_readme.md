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
    1. For each beam, retrieves CTC sequence and applies CTC merging rules. 
    2. Calls self.lexicon.get_valid_next_tokens_with_word_indices on the merged sequence, which returns 
    the next possible valid tokens, whether we are at a word boundary, and word indices. 
    3. If the beam is at a word boundary, we call _decode_sequence_to_text on that beam.
    4. 


