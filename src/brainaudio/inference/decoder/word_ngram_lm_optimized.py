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

MAXIMUM SPEED VERSION:
- Uses Integer State IDs (No string concatenation).
- Uses Parallel List History (No tuple allocation).
- Fully Inlined Scoring Loop (No function call overhead).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import kenlm

# KenLM returns log10 probabilities, but we want natural log.
LOG10_TO_LN = 2.302585092994046


class WordHistory:
    """
    Efficiently stores word sequences using backpointers (Trie structure).
    
    OPTIMIZED: Uses parallel lists instead of a list of tuples to avoid 
    allocating millions of tuple objects during decoding.
    """
    __slots__ = ('words', 'parents')

    def __init__(self):
        # Parallel lists: index `i` in words corresponds to index `i` in parents
        self.words: List[str] = []
        self.parents: List[int] = []

    def add(self, parent_idx: int, word: str) -> int:
        """Register a new word. Returns new Node ID."""
        self.words.append(word)
        self.parents.append(parent_idx)
        return len(self.words) - 1

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
            if curr >= len(w_list): break
            out_words.append(w_list[curr])
            curr = p_list[curr]
            
        return " ".join(reversed(out_words))
    
    def reset(self):
        """Clear history to free memory between trials."""
        self.words.clear()
        self.parents.clear()


class FastNGramLM:
    """
    Optimized KenLM wrapper using Integer State IDs.
    """
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


# =============================================================================
# INLINED & OPTIMIZED Scoring Function
# =============================================================================

def apply_word_ngram_lm_scoring(
    word_lm: FastNGramLM,
    word_history: WordHistory,
    lexicon: "VectorizedLexiconConstraint",
    beam_hyps: "BatchedBeamHyps",
    boundary_token: int,
    next_labels: "torch.Tensor",
    prev_last_labels: "torch.Tensor",
    parent_lexicon_states: "torch.Tensor",
    homophone_prune_threshold: Optional[float] = 10.0,
) -> None:
    """
    Highly optimized scoring loop.
    Inlines function calls and uses local variable caching for max throughput.
    """
    import torch

    if word_lm is None: return

    # --- Pre-fetch logic for speed (avoid dot lookups in loop) ---
    lm_cache = word_lm.transition_cache
    lm_states = word_lm.states
    lm_base_score = word_lm.model.BaseScore
    
    # Constants
    alpha = word_lm.alpha
    beta = word_lm.beta
    unk_offset = word_lm.unk_score_offset
    log_conv = LOG10_TO_LN
    
    # History buffers (Parallel Lists)
    hist_words = word_history.words
    hist_parents = word_history.parents

    # --- PHASE 1: Identification ---
    # Fast boolean mask operations
    valid_mask = beam_hyps.scores != float('-inf')
    at_boundary = (next_labels == boundary_token)
    was_not_boundary = (prev_last_labels != boundary_token)
    completed_word_mask = valid_mask & at_boundary & was_not_boundary

    batch_indices, beam_indices = torch.where(completed_word_mask)
    if len(batch_indices) == 0: return

    batch_indices_list = batch_indices.tolist()
    beam_indices_list = beam_indices.tolist()
    num_homophone_beams = beam_hyps.num_homophone_beams
    
    # Reusable state object for KenLM lookups (avoids allocation on cache miss)
    # We only copy it to the list if we actually add a new state.
    temp_state = kenlm.State()

    # --- PHASE 2: Process Beams ---
    for b, k in zip(batch_indices_list, beam_indices_list):
        
        parent_state = parent_lexicon_states[b, k].item()
        word_indices = lexicon.get_words_at_state(parent_state)
        if not word_indices: continue
        
        # Assumption: lexicon.word_list is already interned in main.py
        candidate_words = [lexicon.word_list[idx] for idx in word_indices]
        
        # Fast Dedupe (only if necessary)
        if len(candidate_words) > 1:
            candidate_words = list(dict.fromkeys(candidate_words))
        
        context_tuples = beam_hyps.context_texts[b][k]

        # Handle Init/Legacy format
        if not context_tuples or (context_tuples and len(context_tuples[0]) == 2):
            context_tuples = [(0.0, 0, -1)] # (score, state_0, hist_-1)

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
                    p_state = lm_states[parent_lm_id]
                    
                    # Compute log10 prob
                    log10 = lm_base_score(p_state, word, temp_state)
                    
                    if word not in word_lm.model:
                        log10 += unk_offset
                    
                    word_score = alpha * (log10 * log_conv) + beta
                    
                    # Register new state (Must copy temp_state)
                    # We can't reuse temp_state for storage, so we alloc new one
                    new_st = kenlm.State()
                    # Re-run Write to ensure safety or just copy (assignment works for KenLM)
                    # KenLM states are copyable via byte copy or assignment
                    lm_base_score(p_state, word, new_st) 
                    
                    lm_states.append(new_st)
                    child_lm_id = len(lm_states) - 1
                    
                    lm_cache[cache_key] = (word_score, child_lm_id)

                # B. Update History (Parallel List Append - Inlined)
                # Returns the index of the newly added item
                hist_words.append(word)
                hist_parents.append(parent_hist_id)
                child_hist_id = len(hist_words) - 1
                
                # C. Accumulate
                total_score = prev_score + word_score
                
                all_candidates.append((total_score, child_lm_id, child_hist_id))

        if not all_candidates: continue

        # --- PHASE 4: Sort and Prune ---
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        if homophone_prune_threshold is not None and all_candidates:
            best_score = all_candidates[0][0]
            # Fast slice-based pruning
            all_candidates = [
                c for c in all_candidates
                if best_score - c[0] <= homophone_prune_threshold
            ]

        new_tuples = all_candidates[:num_homophone_beams]

        # --- PHASE 5: Update Beam ---
        old_best_lm_score = context_tuples[0][0]
        new_best_lm_score = new_tuples[0][0]
        
        beam_hyps.scores[b, k] += (new_best_lm_score - old_best_lm_score)
        beam_hyps.context_texts[b][k] = new_tuples
        
        # Hash the history ID (unique to this text path)
        beam_hyps.context_texts_hash[b][k] = new_tuples[0][2]


def apply_word_ngram_eos_scoring(
    word_lm: FastNGramLM,
    beam_hyps: "BatchedBeamHyps",
) -> None:
    """Apply EOS scoring (Inlined)."""
    if word_lm is None: return

    batch_size = beam_hyps.batch_size
    beam_size = beam_hyps.beam_size
    INACTIVE = float("-inf")
    
    # Pre-fetch
    get_eos = word_lm.get_eos_score

    for b in range(batch_size):
        for k in range(beam_size):
            if beam_hyps.scores[b, k].item() == INACTIVE:
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

            if not updated_tuples: continue

            updated_tuples.sort(key=lambda x: x[0], reverse=True)

            old_best = context_tuples[0][0]
            new_best = updated_tuples[0][0]
            
            beam_hyps.scores[b, k] += (new_best - old_best)
            beam_hyps.context_texts[b][k] = updated_tuples