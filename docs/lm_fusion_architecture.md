# Language Model Fusion Architecture in Beam Search

This document explains how N-gram language models are integrated into the CTC beam search decoder, to help you understand where and how to integrate a neural language model.

## Overview

The decoder supports **shallow fusion**: adding LM scores to acoustic model scores during beam search. Two integration points exist:

1. **Token-level fusion** (N-gram LM): Uses `fusion_models` - applies at every token emission
2. **Word-level fusion** (Neural LM): Uses `lm_fusion` - applies only at word boundaries (NOT YET IMPLEMENTED)

## Token-Level Fusion (N-gram LM) - Current Implementation

### Key Components

#### 1. Fusion Model Interface

The N-gram LM (`NGramGPULanguageModel`) implements a **stateful** interface:

```python
class NGramGPULanguageModel:
    def get_init_states(batch_size: int, bos: bool) -> torch.Tensor:
        """Initialize LM states for all beams.
        
        Returns:
            states: [B * beam_size] tensor of state indices
                   Each beam gets its own LM state (typically starting at BOS)
        """
        
    def advance(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scores for all possible next tokens given current states.
        
        Args:
            states: [B * beam_size] current LM state for each beam
            
        Returns:
            scores: [B * beam_size, V-1] log-prob scores for each next token
                   (V-1 because blank is not in LM vocabulary)
            next_states: [B * beam_size, V-1] next LM state for each token choice
        """
        
    def get_final(states: torch.Tensor) -> torch.Tensor:
        """Get end-of-sequence scores.
        
        Args:
            states: [B * beam_size] final LM states
            
        Returns:
            eos_scores: [B * beam_size] log-prob of ending at this state
        """
```

#### 2. Integration Points in Beam Search Loop

Location: `ctc_batched_beam_decoding.py`, function `batched_beam_search_torch()`

**Step 1: Initialization (before frame loop)**
```python
# Lines 330-348
if self.fusion_models is not None:
    fusion_states_list = []
    fusion_states_candidates_list = []
    
    for fusion_model in self.fusion_models:
        fusion_model.to(decoder_outputs.device)
        # Initialize: give each beam a starting LM state (BOS)
        fusion_states_list.append(
            fusion_model.get_init_states(
                batch_size=curr_batch_size * self.beam_size, 
                bos=True
            )
        )
        fusion_states_candidates_list.append(None)
```

**Step 2: Score Computation (every frame)**
```python
# Lines 393-403
if self.fusion_models is not None:
    for fusion_idx, fusion_model in enumerate(self.fusion_models):
        # Compute LM scores for all possible next tokens
        fusion_scores, fusion_states_candidates = fusion_model.advance(
            states=fusion_states_list[fusion_idx].view(-1)
        )
        # Don't add LM score if token is blank or repeat
        fusion_scores = torch.where(
            repeated_mask[..., :-1], 0, 
            fusion_scores.view(curr_batch_size, self.beam_size, -1)
        )
        # Add weighted LM scores to acoustic scores
        log_probs[..., :-1] += self.fusion_models_alpha[fusion_idx] * fusion_scores
        
        # Store candidates for state updates later
        fusion_states_candidates_list[fusion_idx] = fusion_states_candidates
```

**Step 3: State Update (after selecting top-k beams)**
```python
# Lines 421-449
if self.fusion_models is not None:
    last_labels = torch.gather(batched_beam_hyps.last_label, dim=-1, index=next_indices)
    blank_mask = next_labels == self._blank_index
    repeating_mask = next_labels == last_labels
    preserve_state_mask = repeating_mask | blank_mask | ~active_mask
    
    for fusion_idx, fusion_model in enumerate(self.fusion_models):
        # Get the LM state that corresponds to the selected beam+token
        # This is a gather operation: next_indices tells us which beam to copy from,
        # next_labels tells us which token (and thus which next_state) was chosen
        fusion_states = torch.gather(
            fusion_states_candidates, dim=-1, 
            index=next_labels_masked.unsqueeze(-1)
        ).squeeze(-1)
        
        # Only update state if we emitted a new non-blank token
        fusion_states_list[fusion_idx] = torch.where(
            preserve_state_mask, 
            fusion_states_prev,  # Keep old state (blank/repeat/inactive)
            fusion_states        # Use new state (advanced)
        ).view(-1)
```

**Step 4: Finalization (after all frames)**
```python
# Lines 461-467
if self.fusion_models is not None:
    for fusion_idx, fusion_model in enumerate(self.fusion_models):
        if not isinstance(fusion_model, GPUBoostingTreeModel):
            # Add end-of-sequence score bonus
            eos_score = fusion_model.get_final(fusion_states_list[fusion_idx])
            batched_beam_hyps.scores += eos_score * self.fusion_models_alpha[fusion_idx]
```

### Key Observations

1. **Stateful tracking**: Each beam maintains its own LM state throughout decoding
2. **Advance-once pattern**: `advance()` is called once per frame, returns scores for ALL tokens
3. **State preservation**: LM state only advances on new non-blank, non-repeat tokens
4. **No blank token**: LM operates on `V-1` tokens (acoustic model has `V` including blank)
5. **Weighted fusion**: LM scores are multiplied by `fusion_models_alpha` before adding

### Data Flow Example

```
Frame t=5, Beam k=3, Batch b=0:
1. Current state: fusion_states_list[0][b*beam_size + k] = 42
2. Call advance(states=[..., 42, ...])
   → Returns: scores=[..., [s_tok0, s_tok1, ..., s_tokV-2], ...]  # LM scores
              next_states=[..., [state_tok0, state_tok1, ..., state_tokV-2], ...]
3. Add weighted LM scores to acoustic log_probs
4. Select top-k, say beam k=3 chose token 15
5. Update: fusion_states_list[0][b*beam_size + k] = next_states[k, 15]
6. On next frame, repeat with new state 15
```

## Word-Level Fusion (Neural LM) - To Be Implemented

### Planned Architecture

Location: Lines 385-391 show the intended integration point:

```python
# step 2.2.6: apply LM fusion at word boundaries
if self.lm_fusion is not None:
    log_probs = self._apply_lm_fusion(
        log_probs=log_probs,
        beam_hyps=batched_beam_hyps,
        lexicon_mask=lexicon_mask,
        curr_batch_size=curr_batch_size,
    )
```

**Key Differences from Token-Level:**

1. **Word boundary detection**: Only triggers when lexicon indicates word completion
2. **Uses `get_valid_next_tokens_with_word_info()`**: 
   ```python
   valid_tokens, at_word_boundary, word_indices = \
       lexicon.get_valid_next_tokens_with_word_info(sequence)
   ```
3. **Partial sequence decoding**: Must decode token IDs → words to query LM
4. **Selective scoring**: Only rescores tokens that would complete a word

### Proposed Neural LM Interface

```python
class NeuralLanguageModelFusion:
    def __init__(self, model, tokenizer, weight=0.3):
        self.model = model  # e.g., GPT-2, LSTM LM
        self.tokenizer = tokenizer
        self.weight = weight
        
    def score_word_continuations(
        self, 
        partial_texts: List[str],  # ["hello", "hello world", ...]
        candidate_words: List[List[str]]  # [["there", "world"], ["and"], ...]
    ) -> torch.Tensor:
        """Score each candidate word given partial text context.
        
        Args:
            partial_texts: List of decoded text so far for each beam
            candidate_words: List of possible next words for each beam
            
        Returns:
            scores: [B, beam_size, max_candidates] log-probabilities
                   Use -inf for padding when beams have different numbers of candidates
        """
        # Pseudo-code:
        # 1. For each beam, create prompts: partial_text + " " + candidate_word
        # 2. Query LLM for P(candidate_word | partial_text)
        # 3. Return log probabilities
```

### Implementation Steps for Neural LM

1. **Detect word boundaries**: Use `at_word_boundary` flag from lexicon
2. **Decode current sequence to words**: Use `lexicon.decode_sequence_to_words()`
3. **Get word alternatives**: Use `word_indices` to find all homophones at this position
4. **Score alternatives with LLM**: Query neural LM for P(word | context)
5. **Backpropagate to tokens**: Map word scores back to phoneme tokens that complete the word
6. **Add weighted scores**: `log_probs[valid_tokens] += weight * lm_score`

### Integration Points You Need

**In `BatchedBeamCTCComputer.__init__()` (add parameter):**
```python
def __init__(
    self,
    ...,
    lm_fusion: NeuralLanguageModelFusion = None,  # ADD THIS
):
    self.lm_fusion = lm_fusion
```

**Implement `_apply_lm_fusion()` method:**
```python
def _apply_lm_fusion(
    self,
    log_probs: torch.Tensor,  # [B, beam_size, V]
    beam_hyps: BatchedBeamHyps,
    lexicon_mask: torch.Tensor,
    curr_batch_size: int,
) -> torch.Tensor:
    """Apply neural LM rescoring at word boundaries."""
    
    # For each beam, check if at word boundary
    for b in range(curr_batch_size):
        for k in range(self.beam_size):
            seq = beam_hyps.transcript_wb[b, k]
            seq_filtered = seq[seq >= 0].tolist()
            
            # Check word boundary
            valid_tokens, at_boundary, word_indices = \
                self.lexicon.get_valid_next_tokens_with_word_info(seq_filtered)
            
            if at_boundary and len(word_indices) > 0:
                # Decode to text so far
                partial_text = self.lexicon.decode_sequence_to_words(
                    seq_filtered[:-N],  # Exclude last word
                    token_to_symbol, 
                    lexicon_word_map
                )
                
                # Get alternative words for last completed word
                alternatives = [self.lexicon.word_list[idx] for idx in word_indices]
                
                # Score with LLM
                lm_scores = self.lm_fusion.score_word_continuations(
                    [partial_text], 
                    [alternatives]
                )
                
                # Map back to valid next tokens and add scores
                for token_id in valid_tokens:
                    # Figure out which word this token would start/continue
                    # Add weighted LM score to log_probs[b, k, token_id]
                    pass
    
    return log_probs
```

## Comparison: N-gram vs Neural LM Fusion

| Aspect | N-gram (Token-Level) | Neural (Word-Level) |
|--------|---------------------|---------------------|
| **Trigger** | Every frame | Only at word boundaries |
| **Granularity** | Per phoneme token | Per word |
| **State** | Integer state index | Text string (context) |
| **Scoring** | All tokens at once | Selected word candidates |
| **Complexity** | O(V) per beam | O(num_words) per boundary |
| **Overhead** | Constant per frame | Sparse (only at boundaries) |
| **Context** | Fixed n-gram window | Full sequence (transformer) |

## Testing Strategy

1. **Start simple**: Implement without CUDA graphs (`allow_cuda_graphs=False`)
2. **Use breakpoints**: Set breakpoint in `get_valid_next_tokens_with_word_info()` to see word boundaries
3. **Mock LLM first**: Return uniform scores to test integration
4. **Verify word decoding**: Print decoded text at each boundary
5. **Add real LLM**: Replace mock with actual model calls
6. **Optimize**: Move to CUDA graphs once working

## References

- **Token-level fusion**: Lines 330-467 in `ctc_batched_beam_decoding.py`
- **Word boundary detection**: `get_valid_next_tokens_with_word_info()` in `lexicon_constraint.py`
- **Lexicon structure**: Lines 254-288 in `lexicon_constraint.py` (word tracking in trie)
- **Planned integration point**: Lines 385-391 in `ctc_batched_beam_decoding.py`
