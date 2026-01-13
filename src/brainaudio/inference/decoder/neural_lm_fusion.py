
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

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .batched_beam_decoding_utils import BatchedBeamHyps
    from .lexicon_constraint import LexiconConstraint



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


class NeuralLanguageModelFusion(ABC):
    
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
    
    @abstractmethod
    def score_continuations(
        self,
        contexts: List[str],
        candidate_words: List[List[str]]
    ) -> List[List[float]]:
        """
        Score candidate words given context.

        This is the main method that subclasses must implement. It queries the
        neural LM to get log-probabilities for each candidate word continuation.

        Args:
            contexts: List of text contexts (one per beam).
                     Example: ["I saw my", "picnic with"]
            candidate_words: List of candidate word lists (one list per beam).
                           Example: [["aunt", "ant"], ["aunt", "ant"]]

        Returns:
            scores: List of score lists matching structure of candidate_words.
                   Each score is a log-probability (negative values, higher is better).
                   Example: [[-0.3, -4.6], [-1.2, -0.5]] for the inputs above.

        Notes:
            - Return scores in the same order as candidate_words
            - Use log-probabilities (not raw probs)
            - Can return approximate scores (e.g., from sampling) if exact scoring is slow
        """
        raise NotImplementedError("Subclasses must implement score_continuations()")

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

class HuggingFaceLMFusion(NeuralLanguageModelFusion):
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

    @torch.no_grad()
    def score_continuations(self, contexts: List[str], candidate_words: List[List[str]]) -> List[List[float]]:

        """
        Score candidate words given contexts using chunked processing to limit memory usage.

        Processes sequences in chunks of size `self.scoring_chunk_size` to prevent
        OOM errors when scoring many sequences at once.
        """
        
        flat_texts = []
        # Store where the candidate word starts for each entry
        candidate_start_indices = []

        # 1. Prepare Texts
        for context, candidates in zip(contexts, candidate_words):
            for word in candidates:
                # Construct full text
                # Don't add space before punctuation
                if not context:
                    full_text = word
                elif context.endswith(" ") or word.startswith(" "):
                    full_text = f"{context}{word}"
                elif word and word[0] in '.,!?;:\'\"':
                    full_text = f"{context}{word}"
                else:
                    full_text = f"{context} {word}"

                # Compute start_idx from the context
                if context:
                    prefix_ids = self.tokenizer.encode(context, add_special_tokens=True)
                    start_idx = len(prefix_ids) - 1
                else:
                    start_idx = 0

                flat_texts.append(full_text)
                candidate_start_indices.append(start_idx)
                

        if not flat_texts:
            return []

        # 2. Collect scores for each flat text (will be filled in by chunks)
        flat_scores = [0.0] * len(flat_texts)

        # 3. Process in chunks to limit memory usage
        chunk_size = self.scoring_chunk_size
        num_sequences = len(flat_texts)

        for chunk_start in range(0, num_sequences, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_sequences)
            chunk_texts = flat_texts[chunk_start:chunk_end]
            chunk_start_indices = candidate_start_indices[chunk_start:chunk_end]

            # Tokenize this chunk
            inputs = self.tokenizer(
                chunk_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True
            ).to(self.device)

            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            # Forward pass for this chunk
            self.llm_call_count += 1
            outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=True)

            # Shift and Gather
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            log_probs = F.log_softmax(shift_logits, dim=-1)

            gathered_probs = torch.gather(
                log_probs,
                dim=2,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Extract scores for this chunk
            for i, (start_idx, attn) in enumerate(zip(chunk_start_indices, attention_mask)):
                valid_seq_len = attn.sum().item() - 1
                safe_start = min(start_idx, valid_seq_len)
                word_log_probs = gathered_probs[i, safe_start:valid_seq_len]
                score = word_log_probs.sum().item()
                flat_scores[chunk_start + i] = self.weight * score + self.word_insertion_bonus

            # Free memory immediately after processing each chunk
            del outputs, log_probs, gathered_probs, shift_logits, shift_labels
            del inputs, input_ids, attention_mask
            torch.cuda.empty_cache()

        # 4. Reshape flat_scores back to match candidate_words structure
        final_scores = []
        flat_idx = 0
        for candidates in candidate_words:
            beam_scores = []
            for _ in candidates:
                beam_scores.append(flat_scores[flat_idx])
                flat_idx += 1
            final_scores.append(beam_scores)

        return final_scores
    
    def to(self, device: torch.device):
        """Move model to specified device."""
        self.device = device
        self.model.to(device)
        return self
    
def apply_lm_fusion_post_selection(
    lm_fusion: NeuralLanguageModelFusion | None,
    lexicon: 'LexiconConstraint',
    beam_hyps: 'BatchedBeamHyps',
    blank_index: int,
    boundary_token: int,
    next_labels: torch.Tensor,
    prev_last_labels: torch.Tensor,
    homophone_prune_threshold: float | None = 10.0,
    frame_idx: Optional[int] = None
) -> None:
    
    def gather_cache_indices(cache, indices):
        # Implementation depends on cache structure (e.g. tuple of tensors)
        # This is a placeholder for the logic: new_cache = cache[indices]
        return lm_fusion.gather_cache(cache, indices)
   
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
    beam_mapping = []  # Maps flat index -> (batch_idx, beam_idx, tuple_idx)
    cache_gather_indices = []

    for (b, k, context_text_tuples, candidate_words) in to_score:
        # loop through available homophone texts for a given beam
        for tuple_idx, (lm_score, text) in enumerate(context_text_tuples):
            flat_contexts.append(text)
            flat_candidates.append(candidate_words)
            beam_mapping.append((b, k, tuple_idx))
            
            # --- INDEX CALCULATION ---
            # 1. Batch Offset: Jump over previous batches
            b_offset = b * batch_size
            
            # 2. Homophone Rank Offset: Jump 'beam_size' for every rank (tuple_idx)
            #    Rank 0 = +0
            #    Rank 1 = +Beam_Size
            h_offset = tuple_idx * beam_size
            
            # 3. Beam Offset: The specific beam index
            k_offset = k
            
            flat_idx = b_offset + h_offset + k_offset
            cache_gather_indices.append(flat_idx)



    # Single batched LM call - scores all contexts against their candidate words
    all_lm_scores = lm_fusion.score_continuations(flat_contexts, flat_candidates)


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

    for (b, k, context_text_tuples, candidate_words) in to_score:
        base_score = beam_hyps.scores[b, k].item()
        lm_scores_dict = scores_by_beam[(b, k)]

        # Collect all (lm_score, text) candidates by combining each text with each homophone
        # Accumulate LM scores so we track full sequence probability
        all_candidates = []
        for tuple_idx, (prev_lm_score, prev_text) in enumerate(context_text_tuples):
            lm_scores_for_tuple = lm_scores_dict[tuple_idx]
            for word, word_lm_score in zip(candidate_words, lm_scores_for_tuple):
                # Accumulate: new score = previous accumulated LM + this word's LM score
                new_lm_score = prev_lm_score + word_lm_score
                new_text = f"{prev_text} {word}".strip()
                all_candidates.append((new_lm_score, new_text))

        # Deduplicate by lowercase text, keeping the best-scoring capitalization variant
        # This prevents redundant entries like "Get arrested" vs "get arrested" vs "Get Arrested"
        lowercase_to_best = {}  # lowercase_text -> (best_score, best_text)
        for lm_score, text in all_candidates:
            text_lower = text.lower()
            if text_lower not in lowercase_to_best or lm_score > lowercase_to_best[text_lower][0]:
                lowercase_to_best[text_lower] = (lm_score, text)
        all_candidates = list(lowercase_to_best.values())

        # Sort by accumulated LM score (descending)
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        # Prune candidates that are too far below the best
        # This saves memory/compute by not tracking very unlikely homophones
        if homophone_prune_threshold is not None and all_candidates:
            best_score = all_candidates[0][0]
            all_candidates = [c for c in all_candidates if best_score - c[0] <= homophone_prune_threshold]

        # Keep top K (after pruning)
        new_tuples = all_candidates[:num_homophone_beams]

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

def apply_lm_end_of_sentence_with_incomplete_word(
    lm_fusion: NeuralLanguageModelFusion | None,
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