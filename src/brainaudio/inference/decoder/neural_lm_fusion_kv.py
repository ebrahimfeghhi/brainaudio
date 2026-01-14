
# Copyright (c) 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural Language Model fusion for CTC beam search decoding."""

import copy
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .batched_beam_decoding_utils import BatchedBeamHyps
    from .lexicon_constraint import LexiconConstraint

import torch


def get_initial_rwkv_state(
    model,
    tokenizer,
    batch_size: int = 1,
    beam_size: int = 1,
    num_homophones: int = 1,
    device: str = "cuda"
):
    """
    Run a forward pass with BOS token and return the initial RWKV state and logits,
    expanded for batched beam search.

    Args:
        model: HuggingFace RWKV model
        tokenizer: Corresponding tokenizer
        batch_size: Number of utterances being decoded in parallel
        beam_size: Number of beams per utterance
        num_homophones: Number of text interpretations tracked per beam
        device: Device for tensors

    Returns:
        Tuple of (state, logits):
            - state: List of layer state dicts, each tensor expanded to
                     [total_beams, ...] where total_beams = batch_size * beam_size * num_homophones
            - logits: Tensor of shape [total_beams, vocab_size] - logits after BOS token
    """
    total_beams = batch_size * beam_size * num_homophones

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    bos_tensor = torch.tensor([[bos_id]], device=device)

    with torch.no_grad():
        outputs = model(bos_tensor, use_cache=True)

    seed_state = outputs.past_key_values
    seed_logits = outputs.logits[:, -1, :]  # Shape: [1, vocab_size]

    # Expand logits for all beams
    expanded_logits = seed_logits.expand(total_beams, -1).contiguous()

    # Expand state for all beams
    # State is a list of layer dicts (or tensors for older RWKV)
    seed_layers = list(seed_state)
    expanded_layers = []

    for layer in seed_layers:
        if isinstance(layer, dict):
            expanded_layer = {}
            for key, tensor in layer.items():
                if tensor is not None:
                    # Tensor shape is typically [batch, ...], expand batch dim
                    expanded_layer[key] = tensor.expand(total_beams, *tensor.shape[1:]).contiguous()
                else:
                    expanded_layer[key] = None
            expanded_layers.append(expanded_layer)
        else:
            # Tensor mode (older RWKV)
            if layer is not None:
                expanded_layers.append(layer.expand(total_beams, *layer.shape[1:]).contiguous())
            else:
                expanded_layers.append(None)

    return expanded_layers, expanded_logits

def get_capitalization_variants(word: str) -> List[str]:
    """
    Generate capitalization variants of a word.

    Args:
        word: The input word (typically lowercase from lexicon)

    Returns:
        List of unique capitalization variants: [lowercase, Capitalized]
    """
    if not word:
        return [word]

    variants = set()
    variants.add(word.lower())        # lowercase: "their"
    variants.add(word.capitalize())   # Capitalized: "Their"

    return list(variants)


class NeuralLanguageModelFusionKV(ABC):
    
    """
    Base class for neural language model fusion during beam search.
    
    This class defines the interface for integrating neural LMs (e.g., GPT-2, LLaMA)
    into CTC beam search decoding. The LM is applied at word boundaries to rescore
    hypotheses based on linguistic context.
    
    Design:
        - Sparse fusion: Only applied at word boundaries (not every frame)
        - Phoneme-space beams: Beams remain in phoneme space, homophones scored separately
        - Aggregation: Multiple homophone scores combined via max or logsumexp
        - Before pruning: LM scores added before topk selection to guide search
    """
    
    def __init__(
        self, 
        weight: float = 0.3,
        homophone_aggregation: str = 'max',
        device: torch.device = None,
    ):
        """
        Initialize the neural LM fusion module.
        
        Args:
            weight: Weight for LM scores when combining with acoustic scores.
                   Typical range: 0.1-0.5. Higher values give LM more influence.
            homophone_aggregation: How to combine scores across homophones:
                - 'max': Take best homophone (recommended for 1-best decoding)
                - 'logsumexp': Bayesian average (better for N-best/confidence)
            device: Device for LM inference. If None, uses cuda if available.
        """
        self.weight = weight
        self.homophone_aggregation = homophone_aggregation
        self.device = device
        
        if homophone_aggregation not in ['max', 'logsumexp']:
            raise ValueError(f"homophone_aggregation must be 'max' or 'logsumexp', got {homophone_aggregation}")
    
  

    def aggregate_homophone_scores(self, scores: List[float]) -> float:
        """
        Aggregate multiple homophone scores into a single score.
        
        Args:
            scores: List of log-probability scores for different homophones.
                   Example: [-0.357, -4.605] for ["aunt", "ant"]
                   
        Returns:
            Aggregated log-probability score.
            
        Examples:
            >>> # max aggregation (winner-take-all)
            >>> self.aggregate_homophone_scores([-0.357, -4.605])
            -0.357  # Best homophone wins
            
            >>> # logsumexp aggregation (Bayesian)
            >>> self.aggregate_homophone_scores([-0.357, -4.605])
            -0.342  # log(exp(-0.357) + exp(-4.605)) ≈ log(0.7 + 0.01)
        """
        if len(scores) == 1:
            return scores[0]
        
        if self.homophone_aggregation == 'max':
            return max(scores)
        elif self.homophone_aggregation == 'logsumexp':
            return torch.logsumexp(torch.tensor(scores), dim=0).item()
        else:
            raise ValueError(f"Unknown aggregation method: {self.homophone_aggregation}")
    
    def to(self, device: torch.device):
        """Move LM to specified device."""
        self.device = device
        return self

class HuggingFaceLMFusionKV(NeuralLanguageModelFusionKV):
    """
    Neural LM fusion using HuggingFace transformers (GPT-2, LLaMA, etc).
    
    Example usage:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> lm_fusion = HuggingFaceLMFusion(model, tokenizer, weight=0.3)
        >>> decoder = BatchedBeamCTCComputer(..., lm_fusion=lm_fusion)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        weight: float = 0.3,
        homophone_aggregation: str = 'max',
        device: torch.device = None,
        max_context_length: int = 512,
        word_insertion_bonus: float = 0.0,
        scoring_chunk_size: int = 256,
    ):
        """
        Initialize HuggingFace LM fusion.

        Args:
            model: HuggingFace causal LM model (e.g., GPT2LMHeadModel)
            tokenizer: Corresponding tokenizer
            weight: LM weight for fusion
            homophone_aggregation: 'max' or 'logsumexp'
            device: Device for inference
            max_context_length: Maximum context length in tokens (truncate if longer)
            scoring_chunk_size: Maximum number of sequences to score in one forward pass.
                Reduces peak memory usage at the cost of slightly more compute. Default 32.
        """
        super().__init__(weight, homophone_aggregation, device)
        self.model = model
        self.tokenizer = tokenizer
        
        if self.tokenizer.pad_token is None:
            # Common fix for Llama/GPT models: use EOS token as padding
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Update the model config to match, just in case
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
        self.tokenizer.padding_side = "right"
        self.max_context_length = max_context_length
        self.word_insertion_bonus = word_insertion_bonus
        self.scoring_chunk_size = scoring_chunk_size
        
        self.device = device if device is not None else next(self.model.parameters()).device

        # Counter for tracking LLM forward passes
        self.llm_call_count = 0

        print(f"[HuggingFaceLMFusion] word_insertion_bonus={self.word_insertion_bonus}, weight={self.weight}")

        # Ensure we have a BOS token for scoring words at sentence start
        # Some models (e.g., Qwen) don't have bos_token set, so fall back to eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id

        # Move model to device and set to eval mode
        # Skip .to() for quantized models as they may already be on the correct device
        if device is not None:
            current_device = next(self.model.parameters()).device
            if current_device != self.device:
                self.model.to(self.device)
        self.model.eval()

    def reset_call_count(self):
        """Reset the LLM call counter to 0."""
        self.llm_call_count = 0

    def get_call_count(self) -> int:
        """Get the current LLM call count."""
        return self.llm_call_count

    def score_and_step_words(self, current_states, prev_logits, words_batch, return_states=True):
        """
        Score a batch of words using cached state, with PAD masking for RWKV.

        Args:
            current_states: List of state dicts/tensors
            prev_logits: Tensor [Batch, Vocab] (Logits from the end of the PREVIOUS word)
            words_batch: List of strings e.g. [" cat", " a", " to"] (with leading space)
            return_states: If True, return updated states. If False, return None for states
                          to save memory when only scores are needed.

        Returns:
            total_word_scores: [Batch] (Sum of log-probs for the whole word)
            new_states: Updated states (or None if return_states=False)
            last_token_logits: [Batch, Vocab] (Logits for the LAST token of this word)
        """
        batch_size = len(words_batch)
        device = self.device
        pad_id = self.tokenizer.pad_token_id

        # 1. Tokenize and Pad
        # Shape: [Batch, Max_Len]
        encoded = self.tokenizer(words_batch, padding=True, return_tensors="pt").to(device)
        input_ids = encoded.input_ids
        # Mask where 1=Valid, 0=Pad
        attention_mask = encoded.attention_mask
        max_len = input_ids.shape[1]

        # Initialize Scores
        # We will accumulate log_probs here
        total_word_scores = torch.zeros(batch_size, device=device)

        # We need to track the "active" logits for the NEXT token prediction.
        # For the first token (t=0), the active logits are 'prev_logits' passed in.
        active_logits = prev_logits

        # We also need to capture the logits from the *very last* valid token of each word
        final_next_token_logits = torch.zeros_like(prev_logits)

        # 2. Iterate Token-by-Token
        for t in range(max_len):
            # Current token IDs: [Batch, 1]
            curr_tokens = input_ids[:, t].unsqueeze(1)

            # Check which beams are still processing valid words
            # Shape: [Batch]
            is_valid_step = (curr_tokens.squeeze() != pad_id)

            # --- A. SCORING (Business Logic) ---
            # We calculate P(token | history) using 'active_logits'
            # active_logits are from step t-1 (or the previous word if t=0)
            log_probs = torch.log_softmax(active_logits, dim=-1)

            # Gather the log_prob of the specific token we are looking at
            # torch.gather needs indices of shape [Batch, 1]
            token_log_probs = torch.gather(log_probs, 1, curr_tokens).squeeze()

            # Only add to the score if this is a valid token (not padding)
            # We multiply by the mask (0 or 1) to zero out padding scores
            total_word_scores += (token_log_probs * is_valid_step.float())

            # --- B. PHYSICS (State Update) ---
            # We use the masking function we wrote earlier
            # It returns the logits for step t+1
            if return_states:
                new_logits, current_states = self.rwkv7_step_with_masking(
                    curr_tokens, current_states, pad_id
                )
            else:
                # Lightweight forward without state preservation
                new_logits, _ = self.rwkv7_step_with_masking(
                    curr_tokens, current_states, pad_id
                )

            # --- C. SAVE OUTPUTS ---
            # If this token is the LAST valid token for a beam, we must save 'new_logits'
            # because those will be the 'prev_logits' for the NEXT word in the beam search.

            # A token is the "last" if:
            # 1. It is valid AND
            # 2. (It is the end of the loop OR The next token is a pad)
            if t == max_len - 1:
                next_is_pad = torch.ones(batch_size, device=device, dtype=torch.bool) # effectively true
            else:
                next_is_pad = (input_ids[:, t+1] == pad_id)

            is_last_token = is_valid_step & next_is_pad

            if is_last_token.any():
                final_next_token_logits[is_last_token] = new_logits[is_last_token]

            # Update active_logits for the next loop iteration
            active_logits = new_logits

        return total_word_scores, current_states if return_states else None, final_next_token_logits


    def rwkv7_step_with_masking(self, input_ids, current_states, pad_token_id):
        """
        Performs one step of inference for a batch, masking out updates for PAD tokens.

        Args:
            input_ids: Tensor [Batch, 1] of current tokens.
            current_states: The state from the previous step.
            pad_token_id: int

        Returns:
            logits: [Batch, Vocab]
            next_states: The updated state (safe for next step).
        """
        batch_size = input_ids.shape[0]
        
        # 1. Create Mask
        # Shape: [Batch, 1] -> True if valid, False if PAD
        is_valid = (input_ids != pad_token_id)
        
        # 2. CRITICAL: Deep Clone the 'current_states' to preserve them as 'old_states'
        # We need the OLD state untouched to roll back to it.
        # Since 'fla' modifies in-place, we must save a clean copy of the BEFORE state.
        old_states = []
        for layer in current_states:
            old_layer = {}
            for k, v in layer.items():
                if isinstance(v, torch.Tensor):
                    old_layer[k] = v.clone()
                else:
                    old_layer[k] = copy.deepcopy(v)
            old_states.append(old_layer)

        # 3. Forward Pass (Corrupts 'current_states' in-place or returns new ones)
        with torch.no_grad():
            outputs = self.model(input_ids, past_key_values=current_states, use_cache=True)
            logits = outputs.logits[:, -1, :]  # [Batch, Vocab] - last position
            # The model might return a new object or modify 'current_states'.
            # We'll treat 'candidate_states' as the potentially corrupted new state.
            candidate_states = outputs.past_key_values 

        # 4. Apply Mask (Rollback)
        # Logic: Final = (Candidate * Mask) + (Old * (1-Mask))
        final_states = []
        
        for old_layer, cand_layer in zip(old_states, candidate_states):
            final_layer = {}
            # Iterate over keys (e.g., 'v', 'a', 'b' in RWKV7)
            for key in old_layer.keys():
                old_t = old_layer[key]
                cand_t = cand_layer[key]
                
                if isinstance(old_t, torch.Tensor):
                    # Reshape mask for broadcasting
                    # Mask is [Batch, 1], Tensor might be [Batch, Heads, Dim]
                    ndim_diff = old_t.ndim - is_valid.ndim
                    mask_broadcast = is_valid
                    for _ in range(ndim_diff):
                        mask_broadcast = mask_broadcast.unsqueeze(-1)
                    
                    # Ensure types match
                    mask_broadcast = mask_broadcast.to(dtype=old_t.dtype, device=old_t.device)
                    
                    # Soft Rollback
                    # If Valid (1) -> Returns cand_t
                    # If Pad (0)   -> Returns old_t
                    final_layer[key] = (cand_t * mask_broadcast) + (old_t * (1.0 - mask_broadcast))
                else:
                    # Pass through non-tensor metadata (if any)
                    final_layer[key] = cand_t
                    
            final_states.append(final_layer)
            
        return logits, final_states

def apply_lm_fusion_post_selection(
    lm_fusion: NeuralLanguageModelFusionKV | None,
    lexicon: 'LexiconConstraint',
    beam_hyps: 'BatchedBeamHyps',
    blank_index: int,
    boundary_token: int,
    next_labels: torch.Tensor,
    prev_last_labels: torch.Tensor,
    homophone_prune_threshold: float | None = 10.0,
    frame_idx: Optional[int] = None
) -> None:
    

    from .beam_helpers import (
        materialize_beam_transcripts_batched,
        collapse_ctc_sequence
    )

    if lm_fusion is None:
        return

    batch_size, beam_size = beam_hyps.scores.shape
    to_score = []  # List to store: (b, k, context_text, candidate_words)

    # --- PHASE 1: Identify beams that need LM Scoring ---
    # Vectorized boundary detection - check all beams at once
    valid_mask = beam_hyps.scores != float('-inf')
    at_boundary = (next_labels == boundary_token)
    was_not_boundary = (prev_last_labels != boundary_token)

    # Combined mask: beams that crossed a word boundary and need LM scoring
    needs_scoring_mask = valid_mask & at_boundary & was_not_boundary

    # Get indices of beams that need scoring (sparse iteration)
    batch_indices, beam_indices = torch.where(needs_scoring_mask)

    # Batch fetch all transcripts at once
    batch_indices_list = batch_indices.tolist()
    beam_indices_list = beam_indices.tolist()
    all_transcripts = materialize_beam_transcripts_batched(
        beam_hyps, batch_indices_list, beam_indices_list
    )

    # Cache for lexicon lookups to avoid redundant trie traversals
    lexicon_cache = {}

    for i, (b, k) in enumerate(zip(batch_indices_list, beam_indices_list)):
        # Get the sequence to find valid words
        seq_raw = all_transcripts[i]
        seq_ctc = collapse_ctc_sequence(seq_raw.tolist(), blank_index)

        # Use cache to avoid redundant lexicon lookups
        seq_key = tuple(seq_ctc)
        if seq_key in lexicon_cache:
            at_boundary_flag, word_indices = lexicon_cache[seq_key]
        else:
            _, at_boundary_flag, word_indices = lexicon.get_valid_next_tokens_with_word_info(seq_ctc)
            lexicon_cache[seq_key] = (at_boundary_flag, word_indices)

        if not at_boundary_flag or not word_indices:
            continue

        # Get the list of text interpretations for this beam
        # context_text_tuples: List[Tuple[float, str]] where each tuple is (lm_score, text)
        context_text_tuples = beam_hyps.context_texts[b][k]

        # Get base homophones from lexicon
        base_words = [lexicon.word_list[idx] for idx in word_indices]

        # Expand with capitalization variants (e.g., "their" -> ["their", "Their"])
        candidate_words = []
        for word in base_words:
            candidate_words.extend(get_capitalization_variants(word))

        # Remove duplicates while preserving order
        candidate_words = list(dict.fromkeys(candidate_words))

        to_score.append((b, k, context_text_tuples, candidate_words))


    if not to_score:
        return

    # --- PHASE 2: Batch LM Scoring ---
    # We need to score EACH text interpretation from each beam against all its homophones.
    # Example: beam (b=0, k=3) has 2 text interpretations and 3 candidate homophones
    #          -> we need 2 LM calls for this beam (one per text interpretation)
    #
    # Flatten all texts into a single list for one batched LM call:
    # - flat_contexts[i] = text string to use as context
    # - flat_candidates[i] = list of candidate words to score
    # - beam_mapping[i] = (b, k, tuple_idx) to know where to put the results back
    flat_contexts = []
    flat_candidates = []
    all_words = []
    beam_mapping = []  # Maps flat index -> (batch_idx, beam_idx, tuple_idx)
    cache_gather_indices = []
    word_start_by_beam = {}  # (b, k, tuple_idx) -> starting index in all_words

    for (b, k, context_text_tuples, candidate_words) in to_score:
        # loop through available homophone texts for a given beam
        for tuple_idx, (lm_score, text) in enumerate(context_text_tuples):
            # Track starting index in all_words for this (b, k, tuple_idx)
            word_start_by_beam[(b, k, tuple_idx)] = len(all_words)

            flat_contexts.append(text)
            if text == "":
                prefix = ""
            else:
                prefix = " "
            flat_candidates.append(candidate_words)
            all_words.extend(prefix + word.lstrip() for word in candidate_words)

            beam_mapping.append((b, k, tuple_idx))

            # --- INDEX CALCULATION ---
            # Layout: homophones are contiguous for each beam
            # [B0K0H0, B0K0H1, B0K0H2, B0K1H0, B0K1H1, B0K1H2, ...]
            num_homophones = beam_hyps.num_homophone_beams
            b_offset = b * beam_size * num_homophones
            k_offset = k * num_homophones
            h_offset = tuple_idx

            flat_idx = b_offset + k_offset + h_offset
            # Repeat for each candidate word (all share the same cached context)
            cache_gather_indices.extend([flat_idx] * len(candidate_words))

    # --- Gather cached state and logits for scoring ---
    # Get device from cached_state tensors (more reliable for multi-GPU)
    state_device = beam_hyps.cached_state[0]['recurrent_state'].device
    indices = torch.tensor(cache_gather_indices, device=state_device)

    # Index cached_logits: [total_beams, vocab_size] -> [num_pairs, vocab_size]
    gathered_logits = beam_hyps.cached_logits[indices]

    # Index cached_state: list of layer dicts, each tensor [total_beams, ...] -> [num_pairs, ...]
    gathered_state = []
    for layer in beam_hyps.cached_state:
        if isinstance(layer, dict):
            gathered_layer = {
                key: tensor[indices] if tensor is not None else None
                for key, tensor in layer.items()
            }
            gathered_state.append(gathered_layer)
        else:
            gathered_state.append(layer[indices] if layer is not None else None)


    # Phase 1: Score all words without storing states (memory efficient)
    # We only need scores here; states and logits will be recomputed for selected candidates
    total_word_scores, _, _ = lm_fusion.score_and_step_words(
        current_states=gathered_state,
        prev_logits=gathered_logits,
        words_batch=all_words,
        return_states=False
    )

    # Apply LM weight and word insertion bonus
    weighted_scores = lm_fusion.weight * total_word_scores + lm_fusion.word_insertion_bonus

    # --- Unflatten scores back to per-context structure ---
    # word_counts tells us how many words per context
    word_counts = [len(cands) for cands in flat_candidates]
    all_lm_scores = []
    score_idx = 0
    for count in word_counts:
        context_scores = weighted_scores[score_idx:score_idx + count].tolist()
        all_lm_scores.append(context_scores)
        score_idx += count

    # Reorganize results back by (b, k) for easy lookup
    # scores_by_beam[(b, k)][tuple_idx] = list of scores for each candidate word
    scores_by_beam = {}
    for flat_idx, (b, k, tuple_idx) in enumerate(beam_mapping):
        key = (b, k)
        if key not in scores_by_beam:
            scores_by_beam[key] = {}
        scores_by_beam[key][tuple_idx] = all_lm_scores[flat_idx]
    
    # --- PHASE 3: Update Beams ---
    # For each beam, combine each text interpretation with each homophone candidate.
    # Keep top K candidates (sorted by LM score for this word).
    # Update beam's acoustic score by adding the best candidate's LM score.
    num_homophone_beams = beam_hyps.num_homophone_beams

    # Collect selected candidates for state recomputation
    # Each entry: (source_cache_idx, word_string, target_cache_idx)
    selected_for_state_update = []

    for (b, k, context_text_tuples, candidate_words) in to_score:
        base_score = beam_hyps.scores[b, k].item()
        lm_scores_dict = scores_by_beam[(b, k)]

        # Collect all (lm_score, text, all_words_idx) candidates by combining each text with each homophone
        # Accumulate LM scores so we track full sequence probability
        # all_words_idx tracks which entry in new_states/last_token_logits corresponds to this candidate
        all_candidates = []
        for tuple_idx, (prev_lm_score, prev_text) in enumerate(context_text_tuples):
            lm_scores_for_tuple = lm_scores_dict[tuple_idx]
            word_start_idx = word_start_by_beam[(b, k, tuple_idx)]
            for word_idx, (word, word_lm_score) in enumerate(zip(candidate_words, lm_scores_for_tuple)):
                # Accumulate: new score = previous accumulated LM + this word's LM score
                new_lm_score = prev_lm_score + word_lm_score
                new_text = f"{prev_text} {word}".strip()
                all_words_idx = word_start_idx + word_idx
                all_candidates.append((new_lm_score, new_text, all_words_idx))

        # Deduplicate by lowercase text, keeping the best-scoring capitalization variant
        # This prevents redundant entries like "Get arrested" vs "get arrested" vs "Get Arrested"
        lowercase_to_best = {}  # lowercase_text -> (best_score, best_text, all_words_idx)
        for lm_score, text, all_words_idx in all_candidates:
            text_lower = text.lower()
            if text_lower not in lowercase_to_best or lm_score > lowercase_to_best[text_lower][0]:
                lowercase_to_best[text_lower] = (lm_score, text, all_words_idx)
        all_candidates = list(lowercase_to_best.values())

        # Sort by accumulated LM score (descending)
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        # Prune candidates that are too far below the best
        # This saves memory/compute by not tracking very unlikely homophones
        if homophone_prune_threshold is not None and all_candidates:
            best_score = all_candidates[0][0]
            all_candidates = [c for c in all_candidates if best_score - c[0] <= homophone_prune_threshold]

        # Keep top K (after pruning)
        selected_candidates = all_candidates[:num_homophone_beams]

        # Extract (score, text) tuples for context_texts (without the all_words_idx)
        new_tuples = [(score, text) for score, text, _ in selected_candidates]

        # Update beam score using formula: score = acoustic_score + lm_score
        # Extract acoustic component by subtracting old LM, then add new LM
        old_best_lm_score = context_text_tuples[0][0] if context_text_tuples else 0.0
        new_best_lm_score = new_tuples[0][0]
        acoustic_score = base_score - old_best_lm_score
        beam_hyps.scores[b, k] = acoustic_score + new_best_lm_score

        # Update context_texts with new tuples
        beam_hyps.context_texts[b][k] = new_tuples

        # Update hash based on best text (index 0)
        beam_hyps.context_texts_hash[b, k] = hash(new_tuples[0][1])

        # Collect info for state recomputation
        # We need: (source_cache_idx, word_string, target_cache_idx)
        num_homophones = beam_hyps.num_homophone_beams
        for h, (_, _, all_words_idx) in enumerate(selected_candidates):
            source_cache_idx = cache_gather_indices[all_words_idx]
            word_string = all_words[all_words_idx]
            target_cache_idx = b * beam_size * num_homophones + k * num_homophones + h
            selected_for_state_update.append((source_cache_idx, word_string, target_cache_idx))

    # --- PHASE 4: Recompute states for selected candidates only ---
    # This is a much smaller batch than all_words, so memory efficient
    if selected_for_state_update:
        # Unpack selected candidates
        source_indices = [s[0] for s in selected_for_state_update]
        selected_words = [s[1] for s in selected_for_state_update]
        target_indices = [s[2] for s in selected_for_state_update]

        # Gather source states for selected words
        source_indices_tensor = torch.tensor(source_indices, device=state_device)

        selected_logits = beam_hyps.cached_logits[source_indices_tensor]
        selected_state = []
        for layer in beam_hyps.cached_state:
            if isinstance(layer, dict):
                selected_layer = {
                    key: tensor[source_indices_tensor] if tensor is not None else None
                    for key, tensor in layer.items()
                }
                selected_state.append(selected_layer)
            else:
                selected_state.append(layer[source_indices_tensor] if layer is not None else None)

        # Re-run selected words with state tracking
        _, new_states, new_logits = lm_fusion.score_and_step_words(
            current_states=selected_state,
            prev_logits=selected_logits,
            words_batch=selected_words,
            return_states=True
        )

        # Scatter new states back to target cache positions
        for i, target_idx in enumerate(target_indices):
            # Update cached_logits
            beam_hyps.cached_logits[target_idx] = new_logits[i]
            # Update cached_state
            for layer_idx, layer in enumerate(new_states):
                if isinstance(layer, dict):
                    for key, tensor in layer.items():
                        if tensor is not None:
                            beam_hyps.cached_state[layer_idx][key][target_idx] = tensor[i]
                elif layer is not None:
                    beam_hyps.cached_state[layer_idx][target_idx] = layer[i]


def apply_lm_end_of_sentence_with_incomplete_word(
    lm_fusion: NeuralLanguageModelFusionKV | None,
    lexicon: 'LexiconConstraint',
    beam_hyps: 'BatchedBeamHyps',
    blank_index: int,
) -> None:
    """
    Add end-of-sentence probability to beam scores, considering incomplete words.

    For each beam:
    1. Check if the final phoneme sequence (after last boundary) maps to a valid word
    2. If so, compute two options:
       - Option A: current_text + best_eos (ignore incomplete word)
       - Option B: current_text + completed_word + best_eos (include incomplete word)
    3. Take the max of these two scores

    Args:
        lm_fusion: The neural LM fusion module (or None to skip).
        lexicon: Lexicon constraint for word lookup.
        beam_hyps: The beam hypotheses to update in-place.
        blank_index: Index of the CTC blank token.
    """
    from .beam_helpers import (
        materialize_beam_transcripts_batched,
        collapse_ctc_sequence
    )

    if lm_fusion is None:
        return

    batch_size, beam_size = beam_hyps.scores.shape
    eos_candidates = [".", "?", "!"]
    boundary_token = lexicon.word_boundary_token

    # Collect info for all beams
    beam_info = []  # List of (b, k, text, has_incomplete_word, candidate_words)

    # Vectorized: get indices of valid (non-pruned) beams
    valid_mask = beam_hyps.scores != float('-inf')
    batch_indices, beam_indices = torch.where(valid_mask)

    # First pass: collect beams that pass text-based filters
    filtered_beams = []  # List of (b, k, text)
    for b, k in zip(batch_indices.tolist(), beam_indices.tolist()):
        # Get the best text interpretation for this beam
        context_tuples = beam_hyps.context_texts[b][k]
        if not context_tuples:
            continue

        _, text = context_tuples[0]

        # Skip if text already ends with sentence-ending punctuation
        if text.rstrip() and text.rstrip()[-1] in '.?!':
            continue

        filtered_beams.append((b, k, text))

    if not filtered_beams:
        return

    # Batch fetch transcripts for filtered beams only
    filtered_batch_indices = [b for b, k, text in filtered_beams]
    filtered_beam_indices = [k for b, k, text in filtered_beams]
    all_transcripts = materialize_beam_transcripts_batched(
        beam_hyps, filtered_batch_indices, filtered_beam_indices
    )

    # Cache for lexicon lookups to avoid redundant trie traversals
    lexicon_cache = {}

    # Second pass: process with transcripts
    for i, (b, k, text) in enumerate(filtered_beams):
        seq_raw = all_transcripts[i]
        seq_ctc = collapse_ctc_sequence(seq_raw.tolist(), blank_index)

        # Check if adding a boundary token would complete a valid word
        # We do this by checking what valid tokens can follow, and if boundary is among them
        seq_key = tuple(seq_ctc)
        if seq_key in lexicon_cache:
            valid_tokens, at_boundary, word_indices = lexicon_cache[seq_key]
        else:
            valid_tokens, at_boundary, word_indices = lexicon.get_valid_next_tokens_with_word_info(seq_ctc)
            lexicon_cache[seq_key] = (valid_tokens, at_boundary, word_indices)

        candidate_words = []
        # If boundary token is valid AND we're not already at a boundary (i.e., we have an incomplete word)
        if boundary_token in valid_tokens and not at_boundary:
            # Simulate adding boundary token to get the word indices
            seq_with_boundary_key = seq_key + (boundary_token,)
            if seq_with_boundary_key in lexicon_cache:
                _, _, word_indices = lexicon_cache[seq_with_boundary_key]
            else:
                seq_with_boundary = seq_ctc + [boundary_token]
                result = lexicon.get_valid_next_tokens_with_word_info(seq_with_boundary)
                lexicon_cache[seq_with_boundary_key] = result
                _, _, word_indices = result
            # Get the words that would be formed
            for idx in word_indices:
                word = lexicon.word_list[idx]
                # Add capitalization variants
                candidate_words.extend(get_capitalization_variants(word))
            # Remove duplicates
            candidate_words = list(dict.fromkeys(candidate_words))

        beam_info.append((b, k, text, candidate_words))

    if not beam_info:
        return

    # --- Batch LM scoring ---
    # For each beam, we need to score:
    # 1. text + eos (without incomplete word)
    # 2. text + word + eos (with incomplete word, for each candidate word)

    flat_contexts = []
    flat_candidates = []
    beam_mapping = []  # (beam_info_idx, scoring_type, word_idx)
    # scoring_type: 'eos_only' or 'word_then_eos'

    for info_idx, (b, k, text, candidate_words) in enumerate(beam_info):
        # Option A: just EOS
        flat_contexts.append(text)
        flat_candidates.append(eos_candidates)
        beam_mapping.append((info_idx, 'eos_only', None))

        # Option B: word + EOS for each candidate word
        for word_idx, word in enumerate(candidate_words):
            word_with_space = f"{text} {word}".strip() if text else word
            flat_contexts.append(word_with_space)
            flat_candidates.append(eos_candidates)
            beam_mapping.append((info_idx, 'word_then_eos', word_idx))

    # Single batched LM call
    all_scores = lm_fusion.score_continuations(flat_contexts, flat_candidates)

    # --- Process results ---
    # For each beam, find the best option
    results_by_beam = {}  # info_idx -> {'eos_only': score, 'word_options': [(word, score), ...]}

    for flat_idx, (info_idx, scoring_type, word_idx) in enumerate(beam_mapping):
        if info_idx not in results_by_beam:
            results_by_beam[info_idx] = {'eos_only': None, 'word_options': []}

        eos_scores = all_scores[flat_idx]
        best_eos_idx = max(range(len(eos_candidates)), key=lambda i: eos_scores[i])
        best_eos_score = eos_scores[best_eos_idx]
        best_eos_punct = eos_candidates[best_eos_idx]

        if scoring_type == 'eos_only':
            results_by_beam[info_idx]['eos_only'] = (best_eos_score, best_eos_punct)
        else:
            # For word_then_eos, we need word score + eos score
            # But we only scored eos here. We need to also get word score.
            # Actually, let's restructure: score word first, then eos
            b, k, text, candidate_words = beam_info[info_idx]
            word = candidate_words[word_idx]
            results_by_beam[info_idx]['word_options'].append((word, best_eos_score, best_eos_punct))

    # Now we need word scores too - let's do another batch call for words
    word_contexts = []
    word_candidates = []
    word_mapping = []  # (info_idx, word)

    for info_idx, (b, k, text, candidate_words) in enumerate(beam_info):
        if candidate_words:
            word_contexts.append(text)
            word_candidates.append(candidate_words)
            word_mapping.append(info_idx)

    word_scores_by_beam = {}
    if word_contexts:
        all_word_scores = lm_fusion.score_continuations(word_contexts, word_candidates)
        for map_idx, info_idx in enumerate(word_mapping):
            b, k, text, candidate_words = beam_info[info_idx]
            word_scores_by_beam[info_idx] = dict(zip(candidate_words, all_word_scores[map_idx]))

    # --- Update beams ---
    for info_idx, (b, k, text, candidate_words) in enumerate(beam_info):
        result = results_by_beam[info_idx]
        eos_only_score, eos_only_punct = result['eos_only']

        # Option A: just EOS
        option_a_score = eos_only_score
        option_a_text = text + eos_only_punct

        # Option B: best word + EOS
        option_b_score = float('-inf')
        option_b_text = None
        option_b_word = None

        if candidate_words and info_idx in word_scores_by_beam:
            for word, eos_score, eos_punct in result['word_options']:
                word_score = word_scores_by_beam[info_idx].get(word, float('-inf'))
                total = word_score + eos_score
                if total > option_b_score:
                    option_b_score = total
                    option_b_text = f"{text} {word}".strip() + eos_punct
                    option_b_word = word

        # Take the max
        if option_b_score > option_a_score:
            best_score = option_b_score
            best_text = option_b_text
        else:
            best_score = option_a_score
            best_text = option_a_text

        # Update context_texts
        context_tuples = beam_hyps.context_texts[b][k]
        updated_tuples = []
        for lm_score, old_text in context_tuples:
            # Use the best text for the first tuple, append punct for others
            if not updated_tuples:
                updated_tuples.append((lm_score + best_score, best_text))
            else:
                # For other homophones, just add EOS
                updated_tuples.append((lm_score + eos_only_score, old_text + eos_only_punct))
        beam_hyps.context_texts[b][k] = updated_tuples

        # Update beam score
        beam_hyps.scores[b, k] += best_score

        # Update hash
        beam_hyps.context_texts_hash[b, k] = hash(updated_tuples[0][1])