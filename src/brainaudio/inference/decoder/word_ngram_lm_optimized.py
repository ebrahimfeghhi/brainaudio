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

MAXIMUM SPEED VERSION v2:
- Uses Integer State IDs (No string concatenation).
- Uses Parallel List History with DEDUPLICATION (No redundant paths).
- Single KenLM call on cache miss (was 2x before).
- heapq.nlargest instead of full sort.
- Cached UNK word lookup.
- Avoids .item() calls in hot loop.
- Fully Inlined Scoring Loop (No function call overhead).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union
import heapq

import kenlm

if TYPE_CHECKING:
    import torch
    from brainaudio.inference.decoder.batched_beam_decoding_utils import BatchedBeamHyps
    from brainaudio.inference.decoder.vectorized_lexicon_constraint import VectorizedLexiconConstraint

# KenLM returns log10 probabilities, but we want natural log.
LOG10_TO_LN = 2.302585092994046


class WordHistory:
    """
    Efficiently stores word sequences using backpointers (Trie structure).
    
    OPTIMIZED v2: 
    - Uses parallel lists instead of a list of tuples to avoid allocating 
      millions of tuple objects during decoding.
    - Adds path deduplication cache to avoid creating duplicate history entries
      for identical (parent, word) combinations.
    """
    __slots__ = ('words', 'parents', '_path_cache')

    def __init__(self):
        # Parallel lists: index `i` in words corresponds to index `i` in parents
        self.words: List[str] = []
        self.parents: List[int] = []
        # Deduplication cache: (parent_idx, word) -> node_id
        self._path_cache: Dict[Tuple[int, str], int] = {}

    def add(self, parent_idx: int, word: str) -> int:
        """Register a new word with deduplication. Returns Node ID."""
        key = (parent_idx, word)
        if key in self._path_cache:
            return self._path_cache[key]
        
        idx = len(self.words)
        self.words.append(word)
        self.parents.append(parent_idx)
        self._path_cache[key] = idx
        return idx

    def get_text(self, node_idx: int) -> str:
        """Reconstruct full sentence from a Node ID."""
        if node_idx < 0:
            return ""

        # Unroll logic locally for speed
        out_words = []
        curr = node_idx
        # Cache lists locally to avoid self. lookups in loop
        p_list = self.parents
        w_list = self.words

        while curr != -1:
            if curr >= len(w_list):
                break
            out_words.append(w_list[curr])
            curr = p_list[curr]

        # Smart join: no space before punctuation
        result = []
        for word in reversed(out_words):
            if result and word not in '.?!,;:':
                result.append(' ')
            result.append(word)
        return ''.join(result)
    
    def reset(self):
        """Clear history to free memory between trials."""
        self.words.clear()
        self.parents.clear()
        self._path_cache.clear()


class FastNGramLM:
    """
    Optimized KenLM wrapper using Integer State IDs.
    
    OPTIMIZED v2:
    - Caches UNK word lookups to avoid repeated `word in model` checks.
    """
    __slots__ = (
        'model', 'alpha', 'beta', 'unk_score_offset', 'score_boundary',
        'states', 'transition_cache', '_unk_words', '_known_words'
    )
    
    def __init__(
        self,
        model_path: Union[str, Path],
        alpha: float = 0.5,
        beta: float = 0.0,
        unk_score_offset: float = -10.0,
        score_boundary: bool = True,
    ):
        self.model = kenlm.Model(str(model_path))
        self.alpha = alpha
        self.beta = beta
        self.unk_score_offset = unk_score_offset
        self.score_boundary = score_boundary
        
        # State Registry
        self.states: List[kenlm.State] = []
        
        # Cache: (parent_state_id, word) -> (score, child_state_id)
        self.transition_cache: Dict[Tuple[int, str], Tuple[float, int]] = {}
        
        # UNK word caches (avoid repeated `word in model` checks)
        self._unk_words: Set[str] = set()
        self._known_words: Set[str] = set()
        
        self._init_start_state()

    def _init_start_state(self):
        start_state = kenlm.State()
        if self.score_boundary:
            self.model.BeginSentenceWrite(start_state)
        else:
            self.model.NullContextWrite(start_state)
        self.states.append(start_state)

    def get_start_state_id(self) -> int:
        return 0

    def is_unk(self, word: str) -> bool:
        """Check if word is UNK with caching."""
        if word in self._known_words:
            return False
        if word in self._unk_words:
            return True
        # First time seeing this word - check model
        if word in self.model:
            self._known_words.add(word)
            return False
        else:
            self._unk_words.add(word)
            return True

    def get_eos_score(self, state_id: int) -> float:
        if not self.score_boundary:
            return 0.0
        state = self.states[state_id]
        log10_prob = self.model.BaseScore(state, "</s>", kenlm.State())
        return self.alpha * (log10_prob * LOG10_TO_LN)

    def reset_cache(self):
        """Optional: Reset cache to manage memory usage."""
        self.states = []
        self.transition_cache = {}
        self._init_start_state()
        # Note: Keep UNK caches as they're word-level, not state-level


# =============================================================================
# INLINED & OPTIMIZED Scoring Function v2
# =============================================================================

def apply_word_ngram_lm_scoring(
    word_lm: "FastNGramLM",
    word_history: "WordHistory",
    lexicon: "VectorizedLexiconConstraint",
    beam_hyps: "BatchedBeamHyps",
    boundary_token: int,
    next_labels: "torch.Tensor",
    prev_last_labels: "torch.Tensor",
    parent_lexicon_states: "torch.Tensor",
    homophone_prune_threshold: Optional[float] = 10.0,
) -> None:
    """
    Highly optimized scoring loop v2.
    
    Key optimizations over v1:
    - Single KenLM call on cache miss (allocate state first, score into it)
    - Avoids .item() calls by converting tensor to list once
    - Uses heapq.nlargest instead of full sort
    - History deduplication via WordHistory._path_cache
    - Cached UNK word checks
    """
    import torch

    if word_lm is None:
        return

    # --- Pre-fetch logic for speed (avoid dot lookups in loop) ---
    lm_cache = word_lm.transition_cache
    lm_states = word_lm.states
    lm_base_score = word_lm.model.BaseScore
    lm_is_unk = word_lm.is_unk
    
    # Constants
    alpha = word_lm.alpha
    beta = word_lm.beta
    unk_offset = word_lm.unk_score_offset
    log_conv = LOG10_TO_LN
    
    # History with deduplication
    hist_add = word_history.add

    # --- PHASE 1: Identification ---
    # Fast boolean mask operations
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
    
    # Convert tensor to list ONCE to avoid .item() calls in loop
    parent_lexicon_states_list = parent_lexicon_states.tolist()

    # --- PHASE 2: Process Beams ---
    for b, k in zip(batch_indices_list, beam_indices_list):
        
        parent_state = parent_lexicon_states_list[b][k]
        word_indices = lexicon.get_words_at_state(parent_state)
        if not word_indices:
            continue
        
        # Assumption: lexicon.word_list is already interned in main.py
        candidate_words = [lexicon.word_list[idx] for idx in word_indices]
        
        # Fast Dedupe (only if necessary)
        if len(candidate_words) > 1:
            candidate_words = list(dict.fromkeys(candidate_words))
        
        context_tuples = beam_hyps.context_texts[b][k]

        # Handle Init/Legacy format
        if not context_tuples or (context_tuples and len(context_tuples[0]) == 2):
            context_tuples = [(0.0, 0, -1)]  # (score, state_0, hist_-1)

        all_candidates = []

        # --- PHASE 3: The Hot Loop (Fully Inlined) ---
        for prev_score, parent_lm_id, parent_hist_id in context_tuples:
            for word in candidate_words:
                
                # A. Score (Try Cache First)
                # Using (int, str) key is very fast with interned strings
                cache_key = (parent_lm_id, word)
                
                if cache_key in lm_cache:
                    word_score, child_lm_id = lm_cache[cache_key]
                else:
                    # Cache Miss: Calculate KenLM
                    # OPTIMIZATION: Allocate new state FIRST, score directly into it
                    # This avoids the double-call bug in v1
                    p_state = lm_states[parent_lm_id]
                    new_st = kenlm.State()
                    
                    # Compute log10 prob directly into new_st
                    log10 = lm_base_score(p_state, word, new_st)
                    
                    # Use cached UNK check
                    if lm_is_unk(word):
                        log10 += unk_offset
                    
                    word_score = alpha * (log10 * log_conv) + beta
                    
                    # Register new state
                    lm_states.append(new_st)
                    child_lm_id = len(lm_states) - 1
                    
                    lm_cache[cache_key] = (word_score, child_lm_id)

                # B. Update History with Deduplication
                # hist_add returns existing ID if (parent, word) already exists
                child_hist_id = hist_add(parent_hist_id, word)
                
                # C. Accumulate
                total_score = prev_score + word_score
                
                all_candidates.append((total_score, child_lm_id, child_hist_id))

        if not all_candidates:
            continue

        # --- PHASE 4: Select Top-K and Prune ---
        # OPTIMIZATION: Use heapq.nlargest instead of full sort
        if len(all_candidates) <= num_homophone_beams:
            # No need to sort if we're keeping all
            new_tuples = sorted(all_candidates, key=lambda x: x[0], reverse=True)
        else:
            new_tuples = heapq.nlargest(num_homophone_beams, all_candidates, key=lambda x: x[0])

        # Prune after selecting (much smaller list now)
        if homophone_prune_threshold is not None and new_tuples:
            best_score = new_tuples[0][0]
            new_tuples = [
                c for c in new_tuples
                if best_score - c[0] <= homophone_prune_threshold
            ]

        if not new_tuples:
            continue

        # --- PHASE 5: Update Beam ---
        old_best_lm_score = context_tuples[0][0]
        new_best_lm_score = new_tuples[0][0]

        beam_hyps.scores[b, k] += (new_best_lm_score - old_best_lm_score)
        beam_hyps.context_texts[b][k] = new_tuples

        # Track that this beam has a new word pending LLM rescoring
        beam_hyps.unscored_word_count[b, k] += 1


def apply_word_ngram_eos_scoring(
    word_lm: "FastNGramLM",
    beam_hyps: "BatchedBeamHyps",
) -> None:
    """
    Apply EOS scoring (Inlined).
    
    OPTIMIZED v2: Avoids .item() calls by converting scores to list.
    """
    if word_lm is None:
        return

    batch_size = beam_hyps.batch_size
    beam_size = beam_hyps.beam_size
    INACTIVE = float("-inf")
    
    # Pre-fetch
    get_eos = word_lm.get_eos_score
    
    # Convert scores to list once to avoid .item() calls
    scores_list = beam_hyps.scores.tolist()

    for b in range(batch_size):
        for k in range(beam_size):
            if scores_list[b][k] == INACTIVE:
                continue

            context_tuples = beam_hyps.context_texts[b][k]
            
            # Init Check
            if not context_tuples or (context_tuples and len(context_tuples[0]) == 2):
                base_score = context_tuples[0][0] if context_tuples else 0.0
                context_tuples = [(base_score, 0, -1)]

            updated_tuples = []
            
            for lm_score, lm_id, hist_id in context_tuples:
                eos_score = get_eos(lm_id)
                new_lm_score = lm_score + eos_score
                updated_tuples.append((new_lm_score, lm_id, hist_id))

            if not updated_tuples:
                continue

            updated_tuples.sort(key=lambda x: x[0], reverse=True)

            old_best = context_tuples[0][0]
            new_best = updated_tuples[0][0]
            
            beam_hyps.scores[b, k] += (new_best - old_best)
            beam_hyps.context_texts[b][k] = updated_tuples