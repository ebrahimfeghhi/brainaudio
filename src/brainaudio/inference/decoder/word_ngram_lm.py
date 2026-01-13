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

"""
Word-level N-gram Language Model for CTC beam search decoding.

This module provides fast, state-based N-gram LM scoring using KenLM.
The key advantage over neural LM scoring is O(1) per-word scoring via
incremental state updates, compared to O(context_length) for neural LMs.

Usage:
    >>> lm = WordNGramLMFusion.from_kenlm("/path/to/model.kenlm", alpha=0.5)
    >>> state = lm.get_start_state()
    >>> score, new_state = lm.score_word(state, "hello")
    >>> score2, new_state2 = lm.score_word(new_state, "world")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import kenlm


# KenLM returns log10 probabilities, but we want natural log for consistency
# with acoustic model scores. This factor converts log10 to ln.
LOG10_TO_LN = 2.302585092994046  # ln(10)


@dataclass
class KenlmState:
    """
    Wrapper around kenlm.State for cleaner API.

    KenLM states encode the last N-1 words (for an N-gram model) and allow
    O(1) scoring of the next word without re-processing the entire context.

    Each state is approximately 64 bytes (depends on N-gram order).
    """
    state: "kenlm.State"

    def copy(self) -> "KenlmState":
        """Create a copy of this state (for beam branching)."""
        new_state = kenlm.State()
        # KenLM states can be copied by assignment
        new_state.__setstate__(self.state.__getstate__())
        return KenlmState(new_state)


class WordNGramLMFusion:
    """
    Fast word-level N-gram LM for CTC beam search using KenLM.

    Uses KenLM's state-based scoring for O(1) incremental word scoring.
    This is ~1000x faster than re-scoring the full context for each word.

    Attributes:
        alpha: LM weight for shallow fusion. Score contribution = alpha * lm_score + beta
        beta: Word insertion bonus (added per word).
        order: N-gram order (e.g., 4 for 4-gram model).

    Example:
        >>> lm = WordNGramLMFusion.from_kenlm("model.kenlm", alpha=0.5, beta=0.0)
        >>> state = lm.get_start_state()
        >>>
        >>> # Score words incrementally
        >>> score1, state1 = lm.score_word(state, "hello")
        >>> score2, state2 = lm.score_word(state1, "world")
        >>>
        >>> # Score multiple homophones from same state
        >>> results = lm.score_candidates(state, ["aunt", "ant"])
    """

    def __init__(
        self,
        kenlm_model: "kenlm.Model",
        alpha: float = 0.5,
        beta: float = 0.0,
        unk_score_offset: float = -10.0,
        score_boundary: bool = True,
    ):
        """
        Initialize the word N-gram LM.

        Args:
            kenlm_model: Loaded KenLM model instance.
            alpha: LM weight for shallow fusion (multiplies log-prob).
            beta: Word insertion bonus (added per word).
            unk_score_offset: Additional penalty for OOV words (in log-prob space).
            score_boundary: Whether to include <s> and </s> in scoring.
        """
        self._model = kenlm_model
        self.alpha = alpha
        self.beta = beta
        self.unk_score_offset = unk_score_offset
        self.score_boundary = score_boundary

    @classmethod
    def from_kenlm(
        cls,
        model_path: Union[str, Path],
        alpha: float = 0.5,
        beta: float = 0.0,
        unk_score_offset: float = -10.0,
        score_boundary: bool = True,
    ) -> "WordNGramLMFusion":
        """
        Load a KenLM model from file.

        Supports both ARPA (.arpa) and binary (.bin, .kenlm, .binary) formats.
        Binary format is recommended for faster loading and lower memory usage.

        Args:
            model_path: Path to KenLM model file (.arpa, .bin, .kenlm, or .binary).
            alpha: LM weight for shallow fusion.
            beta: Word insertion bonus.
            unk_score_offset: OOV penalty.
            score_boundary: Whether to score sentence boundaries.

        Returns:
            Initialized WordNGramLMFusion instance.

        Example:
            >>> lm = WordNGramLMFusion.from_kenlm("/path/to/model.kenlm", alpha=0.5)
        """
        model_path = str(model_path)
        model = kenlm.Model(model_path)

        return cls(
            kenlm_model=model,
            alpha=alpha,
            beta=beta,
            unk_score_offset=unk_score_offset,
            score_boundary=score_boundary,
        )

    @property
    def order(self) -> int:
        """Get the N-gram order (e.g., 4 for a 4-gram model)."""
        return self._model.order

    def get_start_state(self) -> KenlmState:
        """
        Get the initial LM state for beginning of sentence.

        If score_boundary=True, this state encodes <s> (begin sentence marker).
        Otherwise, it's a null context state.

        Returns:
            Initial KenlmState for starting a new sentence.
        """
        state = kenlm.State()
        if self.score_boundary:
            self._model.BeginSentenceWrite(state)
        else:
            self._model.NullContextWrite(state)
        return KenlmState(state)

    def score_word(
        self,
        prev_state: KenlmState,
        word: str,
        is_last_word: bool = False,
    ) -> Tuple[float, KenlmState]:
        """
        Score a single word given the previous LM state.

        This is an O(1) operation - just a hash table lookup in the N-gram model.
        The returned state encodes the new context for scoring subsequent words.

        Args:
            prev_state: LM state encoding the previous N-1 words.
            word: The word to score.
            is_last_word: If True, also adds </s> (end of sentence) score.

        Returns:
            Tuple of (weighted_score, new_state) where:
                - weighted_score = alpha * ln_prob + beta
                - new_state encodes context including this word

        Example:
            >>> state = lm.get_start_state()
            >>> score, state = lm.score_word(state, "hello")
            >>> print(f"P(hello|<s>) contribution: {score}")
        """
        end_state = kenlm.State()

        # KenLM's BaseScore returns log10 probability
        log10_score = self._model.BaseScore(prev_state.state, word, end_state)

        # Check for OOV (KenLM returns a very negative score for OOV)
        # We can also explicitly check if word is in vocabulary
        if word not in self._model:
            log10_score += self.unk_score_offset

        # Add end-of-sentence score if this is the last word
        if is_last_word and self.score_boundary:
            eos_state = kenlm.State()
            log10_score += self._model.BaseScore(end_state, "</s>", eos_state)

        # Convert log10 to natural log and apply weighting
        ln_score = log10_score * LOG10_TO_LN
        weighted_score = self.alpha * ln_score + self.beta

        return weighted_score, KenlmState(end_state)

    def score_candidates(
        self,
        prev_state: KenlmState,
        candidate_words: List[str],
        is_last_word: bool = False,
    ) -> List[Tuple[float, KenlmState]]:
        """
        Score multiple candidate words from the same parent state.

        Useful for scoring homophones at a word boundary. Each candidate
        gets its own score and resulting state.

        Args:
            prev_state: The shared previous LM state.
            candidate_words: List of candidate words (e.g., ["aunt", "ant"]).
            is_last_word: Whether these are sentence-final words.

        Returns:
            List of (weighted_score, new_state) tuples, one per candidate.

        Example:
            >>> state = lm.get_start_state()
            >>> # Score context "I saw my"
            >>> for word in ["I", "saw", "my"]:
            ...     _, state = lm.score_word(state, word)
            >>> # Now score homophones
            >>> results = lm.score_candidates(state, ["aunt", "ant"])
            >>> for word, (score, _) in zip(["aunt", "ant"], results):
            ...     print(f"P({word}|I saw my) contribution: {score}")
        """
        results = []
        for word in candidate_words:
            score, new_state = self.score_word(prev_state, word, is_last_word)
            results.append((score, new_state))
        return results

    def get_eos_score(self, state: KenlmState) -> float:
        """
        Get the end-of-sentence score for a given state.

        Call this at the end of decoding to add </s> probability.

        Args:
            state: Current LM state after the last word.

        Returns:
            Weighted EOS score (alpha * ln_prob, no beta since not a word).
        """
        if not self.score_boundary:
            return 0.0

        eos_state = kenlm.State()
        log10_score = self._model.BaseScore(state.state, "</s>", eos_state)
        ln_score = log10_score * LOG10_TO_LN

        return self.alpha * ln_score


# =============================================================================
# Text-keyed LM State Cache (following pyctcdecode pattern)
# =============================================================================
#
# Instead of tracking LM state per beam index, we use a text-keyed cache.
# This approach (from pyctcdecode) is simpler and more robust because:
#   1. No state/text desync risk - text IS the key
#   2. Natural deduplication - beams with same text share cache entry
#   3. Survives beam recombination - no gather operations needed
#   4. Easy debugging - can inspect cache by text

# Type aliases for the cache
WordLMCacheKey = Tuple[str, bool]  # (text, is_eos)
WordLMCacheValue = Tuple[float, KenlmState]  # (accumulated_score, state)
WordLMCache = Dict[WordLMCacheKey, WordLMCacheValue]


def create_word_lm_cache(word_lm: WordNGramLMFusion) -> WordLMCache:
    """
    Initialize an LM cache with the start state for empty text.

    The cache maps (text, is_eos) -> (accumulated_score, state).

    Args:
        word_lm: The WordNGramLMFusion instance.

    Returns:
        Initialized cache with entry for empty text.
    """
    start_state = word_lm.get_start_state()
    return {("", False): (0.0, start_state)}


def score_word_with_cache(
    cache: WordLMCache,
    word_lm: WordNGramLMFusion,
    parent_text: str,
    new_word: str,
    is_eos: bool = False,
) -> Tuple[float, KenlmState]:
    """
    Score a word using cached parent state.

    Looks up the parent state by text, scores the new word, caches the result,
    and returns the accumulated score and new state.

    Args:
        cache: The LM score cache.
        word_lm: The WordNGramLMFusion instance.
        parent_text: The text before this word (key for parent state lookup).
        new_word: The word to score.
        is_eos: Whether this is end-of-sentence scoring.

    Returns:
        Tuple of (accumulated_score, new_state).

    Raises:
        KeyError: If parent_text is not in the cache.
    """
    # Construct new text
    if parent_text:
        new_text = f"{parent_text} {new_word}"
    else:
        new_text = new_word

    cache_key = (new_text, is_eos)

    # Check cache first
    if cache_key in cache:
        return cache[cache_key]

    # Look up parent state
    parent_key = (parent_text, False)
    if parent_key not in cache:
        raise KeyError(
            f"Parent text '{parent_text}' not in cache. "
            f"Available keys: {list(cache.keys())[:5]}..."
        )

    parent_score, parent_state = cache[parent_key]

    # Score the new word (O(1) operation)
    word_score, new_state = word_lm.score_word(parent_state, new_word, is_last_word=is_eos)

    # Accumulate scores
    accumulated_score = parent_score + word_score

    # Cache the result
    cache[cache_key] = (accumulated_score, new_state)

    return accumulated_score, new_state


# =============================================================================
# N-gram LM Scoring Function (called after beam emits word boundary)
# =============================================================================

def apply_word_ngram_lm_scoring(
    word_lm: "WordNGramLMFusion",
    word_lm_cache: WordLMCache,
    lexicon: "VectorizedLexiconConstraint",
    beam_hyps: "BatchedBeamHyps",
    boundary_token: int,
    next_labels: "torch.Tensor",
    prev_last_labels: "torch.Tensor",
    parent_lexicon_states: "torch.Tensor",
    homophone_prune_threshold: Optional[float] = 10.0,
) -> None:
    """
    Apply word-level N-gram LM scoring after beams emit word boundary tokens.

    This function is called AFTER topk selection, when some beams have just
    completed a word by emitting the boundary token '|'. For each such beam:

    1. Get the completed word(s) from the lexicon state (may be homophones)
    2. For each existing text interpretation × each homophone candidate:
       - Score with N-gram LM using the cache (O(1) per word)
       - Create new (accumulated_score, new_text) tuple
    3. Deduplicate by text (since all lowercase, no capitalization variants)
    4. Sort by score, keep top K
    5. Update context_texts and beam_hyps.scores

    Note: All text is lowercase since the N-gram LM was trained on lowercase.
    Capitalization is handled separately by the LLM (if used).

    Note: The score is only applied once per word. For sequences like "C A T | |",
    the score is applied after the first "|" only (when prev_last_labels != "|").

    Args:
        word_lm: WordNGramLMFusion instance for scoring.
        word_lm_cache: Text-keyed cache {(text, is_eos): (score, state)}.
        lexicon: VectorizedLexiconConstraint for word lookup.
        beam_hyps: BatchedBeamHyps to update in-place.
        boundary_token: Word boundary token index (e.g., '|').
        next_labels: [B, beam_size] - tokens just selected at this frame.
        prev_last_labels: [B, beam_size] - previous last non-blank tokens.
        parent_lexicon_states: [B, beam_size] - lexicon states BEFORE boundary was emitted.
            This tells us which word completed at each beam.
        homophone_prune_threshold: Max score difference from best to keep.
    """
    import torch

    if word_lm is None:
        return

    batch_size, beam_size = beam_hyps.scores.shape

    # --- PHASE 1: Identify beams that just completed a word ---
    # A word is completed when:
    #   - The beam just emitted boundary token '|'
    #   - The previous token was NOT a boundary (prevents double-scoring "| |")
    valid_mask = beam_hyps.scores != float('-inf')
    at_boundary = (next_labels == boundary_token)
    was_not_boundary = (prev_last_labels != boundary_token)
    completed_word_mask = valid_mask & at_boundary & was_not_boundary

    batch_indices, beam_indices = torch.where(completed_word_mask)
    if len(batch_indices) == 0:
        return

    batch_indices_list = batch_indices.tolist()
    beam_indices_list = beam_indices.tolist()

    num_homophone_beams = beam_hyps.num_homophone_beams

    # --- PHASE 2: Process each beam that completed a word ---
    for b, k in zip(batch_indices_list, beam_indices_list):
        # Get word indices directly from parent lexicon state (efficient!)
        # parent_lexicon_states[b, k] is the state BEFORE '|' was emitted
        parent_state = parent_lexicon_states[b, k].item()
        word_indices = lexicon.get_words_at_state(parent_state)
        

        if not word_indices:
            continue
        
        # Get homophones from lexicon (all lowercase)
        candidate_words = [lexicon.word_list[idx].lower() for idx in word_indices]
        # Remove duplicates while preserving order
        candidate_words = list(dict.fromkeys(candidate_words))
        
        # Get current context texts for this beam
        context_text_tuples = beam_hyps.context_texts[b][k]

        # Handle first word case (empty context_texts)
        if not context_text_tuples:
            context_text_tuples = [(0.0, "")]

        # --- PHASE 3: Score all combinations (existing texts × homophones) ---
        all_candidates = []

        for prev_score, prev_text in context_text_tuples:
            for word in candidate_words:
                try:
                    # Score with N-gram LM using cache (O(1) operation)
                    accumulated_score, _ = score_word_with_cache(
                        cache=word_lm_cache,
                        word_lm=word_lm,
                        parent_text=prev_text,
                        new_word=word,
                        is_eos=False,
                    )
                    new_text = f"{prev_text} {word}".strip() if prev_text else word
                    all_candidates.append((accumulated_score, new_text))
                except KeyError:
                    # Parent text not in cache - this shouldn't happen normally
                    continue

        if not all_candidates:
            continue

        # --- PHASE 4: Deduplicate, sort, and keep top K ---
        # Deduplicate by text (all lowercase, so just use text directly)
        seen_texts = {}
        for score, text in all_candidates:
            if text not in seen_texts or score > seen_texts[text]:
                seen_texts[text] = score
        all_candidates = [(score, text) for text, score in seen_texts.items()]

        # Sort by score (descending)
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        # Prune candidates too far below the best
        if homophone_prune_threshold is not None and all_candidates:
            best_score = all_candidates[0][0]
            all_candidates = [
                c for c in all_candidates
                if best_score - c[0] <= homophone_prune_threshold
            ]

        # Keep top K
        new_tuples = all_candidates[:num_homophone_beams]

        # --- PHASE 5: Update beam_hyps ---
        # Update beam score: new_score = acoustic_score + best_ngram_score
        # acoustic_score = current_score - old_best_lm_score
        old_tuples = beam_hyps.context_texts[b][k]
        old_best_lm_score = old_tuples[0][0] if old_tuples else 0.0
        new_best_lm_score = new_tuples[0][0]

        current_score = beam_hyps.scores[b, k].item()
        acoustic_score = current_score - old_best_lm_score
        beam_hyps.scores[b, k] = acoustic_score + new_best_lm_score

        # Update context_texts with new tuples
        beam_hyps.context_texts[b][k] = new_tuples

        # Update hash based on best text
        beam_hyps.context_texts_hash[b][k] = hash(new_tuples[0][1])


def apply_word_ngram_eos_scoring(
    word_lm: WordNGramLMFusion,
    word_lm_cache: WordLMCache,
    beam_hyps: "BatchedBeamHyps",
) -> None:
    """
    Apply end-of-sentence scoring from the word-level N-gram LM to all active beams.

    Called once at the end of decoding to add </s> probability to beam scores.
    Updates both beam_hyps.scores and the LM scores in context_texts.

    Args:
        word_lm: The WordNGramLMFusion instance.
        word_lm_cache: Text-keyed cache {(text, is_eos): (score, state)}.
        beam_hyps: The beam hypotheses to update.
    """
    if word_lm is None:
        return

    batch_size = beam_hyps.batch_size
    beam_size = beam_hyps.beam_size
    INACTIVE_SCORE = float("-inf")

    for b in range(batch_size):
        for k in range(beam_size):
            # Skip inactive beams
            if beam_hyps.scores[b, k].item() == INACTIVE_SCORE:
                continue

            # Get current text for this beam
            context_tuples = beam_hyps.context_texts[b][k]
            if not context_tuples:
                continue

            # Process each text interpretation (homophones)
            updated_tuples = []
            for lm_score, text in context_tuples:
                # Look up state for this text in cache
                cache_key = (text, False)
                if cache_key not in word_lm_cache:
                    # Text not in cache - keep original score
                    updated_tuples.append((lm_score, text))
                    continue

                _, state = word_lm_cache[cache_key]

                # Get EOS score and add to accumulated LM score
                eos_score = word_lm.get_eos_score(state)
                new_lm_score = lm_score + eos_score
                updated_tuples.append((new_lm_score, text))

            if not updated_tuples:
                continue

            # Sort by score descending
            updated_tuples.sort(key=lambda x: x[0], reverse=True)

            # Update beam score: add the delta from the best interpretation
            old_best_lm_score = context_tuples[0][0]
            new_best_lm_score = updated_tuples[0][0]
            score_delta = new_best_lm_score - old_best_lm_score
            beam_hyps.scores[b, k] += score_delta

            # Update context_texts
            beam_hyps.context_texts[b][k] = updated_tuples
