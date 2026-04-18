# Word-Level N-gram LM Integration Plan

> **Reference Implementation**: This plan draws heavily from [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode), specifically:
> - `pyctcdecode/language_model.py` - KenLM wrapper with state-based scoring
> - `pyctcdecode/decoder.py` - Text-keyed LM score cache pattern
>
> Key files in `/home/ebrahim/pyctcdecode/pyctcdecode/` should be consulted during implementation.

## Executive Summary

This document outlines the implementation of a **fast, state-based word-level N-gram language model** for CTC beam search decoding. The key insight is that naive implementations that rescore the entire context at each word boundary are O(context_length) per word, while state-based implementations using KenLM are O(1) per word.

**Current state**: The codebase has `NeuralLanguageModelFusion` which works with text strings and rescores context. This is acceptable for neural LMs (which need full context anyway), but is too slow for N-gram LMs where we can do incremental scoring.

**Goal**: Implement `WordNGramLMFusion` that maintains KenLM state per beam, enabling O(1) word scoring.

### Key Lessons from pyctcdecode Review

After reviewing `/home/ebrahim/pyctcdecode/pyctcdecode/decoder.py`, the correct approach uses:

| Aspect | Original Plan (Wrong) | pyctcdecode Pattern (Correct) |
|--------|----------------------|------------------------------|
| State storage | `word_lm_states[b][k]` per beam | `cache[(text, is_eos)]` by text |
| Parent lookup | Gather by beam index | Lookup by parent text string |
| Beam recombination | Complex state merging | Automatic (same text = same entry) |
| Cache lifetime | Reset each frame | Persists entire decoding |
| Files to modify | 4 files | 3 files (no BatchedBeamHyps changes) |

The text-keyed cache is simpler and more robust because:
1. **No state/text desync risk** - text IS the key
2. **Natural deduplication** - beams with same text share cache entry
3. **Survives recombination** - no gather operations needed
4. **Easy debugging** - can inspect cache by text

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Architecture Overview](#2-architecture-overview)
3. [Implementation Components](#3-implementation-components)
4. [Detailed Implementation Steps](#4-detailed-implementation-steps)
5. [Integration with Existing Code](#5-integration-with-existing-code)
6. [Performance Considerations](#6-performance-considerations)
7. [Testing Strategy](#7-testing-strategy)
8. [Future Optimizations](#8-future-optimizations)

---

## 1. Problem Analysis

### 1.1 Why Current Approach is Slow

The current `NeuralLanguageModelFusion.score_continuations()` interface:

```python
def score_continuations(
    self,
    contexts: List[str],           # ["I saw my", "picnic with"]
    candidate_words: List[List[str]]  # [["aunt", "ant"], ["friends"]]
) -> List[List[float]]:
```

For N-gram LMs, this forces rescoring the entire context:

```python
# SLOW: O(context_length) per word
full_text = f"{context} {word}"
score = kenlm_model.score(full_text)  # Rescores everything!
```

For a 100-word sentence with beam_size=16 and 3 homophones per word:
- Context rescoring: 100 words × 16 beams × 3 homophones × O(100) = **4.8 million operations**

### 1.2 How pyctcdecode Solves This

pyctcdecode uses **state-based incremental scoring**:

```python
# FAST: O(1) per word
def score(self, prev_state: KenlmState, word: str) -> Tuple[float, KenlmState]:
    end_state = kenlm.State()
    score = self._kenlm_model.BaseScore(prev_state.state, word, end_state)
    return score, KenlmState(end_state)
```

The `kenlm.State` object internally encodes the last N-1 words (for an N-gram model). Scoring is just a hash table lookup.

Same scenario with state-based scoring:
- Incremental scoring: 100 words × 16 beams × 3 homophones × O(1) = **4,800 operations**

**1000x speedup** by avoiding context rescoring.

### 1.3 Key Insight

The challenge is that your architecture currently tracks **text strings** (`context_texts`) per beam for homophone tracking, not LM states. We need to add state tracking alongside text tracking.

---

## 2. Architecture Overview

### 2.1 Key Insight from pyctcdecode: Text-Based Caching

**Critical pattern from pyctcdecode** (`decoder.py:121-126, 383-395`):

Instead of tracking LM state per beam index, pyctcdecode uses a **text-keyed cache**:

```python
# Cache structure (from pyctcdecode)
LMScoreCacheKey = Tuple[str, bool]  # (text, is_eos)
LMScoreCacheValue = Tuple[float, float, AbstractLMState]  # (lm_hw_score, raw_lm_score, end_state)
LMScoreCache = Dict[LMScoreCacheKey, LMScoreCacheValue]
```

**Why this is better than per-beam state tracking:**

1. **Natural deduplication**: Beams with identical text share the same cache entry
2. **Survives beam recombination**: When beams merge, we don't need complex state gathering
3. **Persists across frames**: Cache grows throughout decoding, avoiding redundant scoring
4. **Simple parent lookup**: Parent state is found by `cache[(parent_text, False)]`

### 2.2 New Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        WordNGramLMFusion                                │
├─────────────────────────────────────────────────────────────────────────┤
│  - kenlm_model: kenlm.Model           # The actual N-gram model         │
│  - alpha: float                        # Fusion weight                  │
│  - beta: float                         # Word insertion bonus           │
│  - unigram_set: Set[str]              # Known vocabulary                │
│  - unk_score_offset: float            # OOV penalty                     │
├─────────────────────────────────────────────────────────────────────────┤
│  + get_start_state() -> KenlmState                                      │
│  + score_word(prev_state, word, is_eos) -> (float, KenlmState)         │
│  + get_eos_score(state) -> float                                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│              WordLMScoreCache (NEW - following pyctcdecode)             │
├─────────────────────────────────────────────────────────────────────────┤
│  Type: Dict[Tuple[str, bool], Tuple[float, float, KenlmState]]         │
│        Key: (text, is_eos)                                              │
│        Value: (combined_score, raw_lm_score, end_state)                 │
│                                                                         │
│  Initialization: {("", False): (0.0, 0.0, start_state)}                │
│                                                                         │
│  Usage pattern:                                                         │
│    1. Look up parent: parent_score, parent_state = cache[(parent_text, False)]│
│    2. Score new word: score, new_state = lm.score_word(parent_state, word)    │
│    3. Store result: cache[(new_text, is_eos)] = (combined, raw, new_state)    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     BatchedBeamHyps (Minimal Changes)                   │
├─────────────────────────────────────────────────────────────────────────┤
│  Existing (keep as-is):                                                 │
│  - context_texts[b][k]: List[Tuple[float, str]]  # (lm_score, text)    │
│  - context_texts_hash[b][k]: int                                        │
│                                                                         │
│  The text in context_texts serves as the key into the LM cache!        │
│  No additional state storage needed per beam.                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 State Flow During Decoding (pyctcdecode pattern)

```
Initialization:
  └── cached_lm_scores = {("", False): (0.0, 0.0, start_state)}

Frame 1-N: On word boundary for beam with text "hello world":
  └── 1. Parent lookup: _, _, parent_state = cache[("hello world", False)]
      2. For each homophone candidate (e.g., "there", "their"):
         a. score, new_state = lm.score_word(parent_state, "there")
         b. cache[("hello world there", False)] = (score, raw, new_state)
      3. Update context_texts with new text (cache entry already exists)

Final: EOS scoring
  └── For each beam with text T:
      _, _, state = cache[(T, False)]
      eos_score = lm.score_word(state, "</s>", is_eos=True)
      cache[(T, True)] = (eos_score, raw, _)
```

### 2.5 The Core Pattern: Incremental Scoring with Cache (pyctcdecode decoder.py:383-395)

This is the most important pattern to understand:

```python
# From pyctcdecode decoder.py, _get_lm_beams() method
def _get_lm_beams(self, beams, ..., cached_lm_scores, ...):
    for beam in beams:
        new_text = _merge_tokens(beam.text, beam.next_word)
        cache_key = (new_text, is_eos)

        if cache_key not in cached_lm_scores:
            # KEY INSIGHT: Look up PARENT state by PARENT TEXT
            _, prev_raw_lm_score, start_state = cached_lm_scores[(beam.text, False)]

            # Score ONLY the new word (O(1) via KenLM state)
            score, end_state = language_model.score(
                start_state, beam.next_word, is_last_word=is_eos
            )

            # ACCUMULATE scores
            raw_lm_score = prev_raw_lm_score + score

            # CACHE for future use
            cached_lm_scores[cache_key] = (lm_hw_score, raw_lm_score, end_state)

        # Retrieve from cache (O(1) lookup)
        lm_score, _, _ = cached_lm_scores[cache_key]
```

**Why this is O(1) per word**:
1. Parent state lookup: O(1) hash table lookup by text
2. KenLM scoring: O(1) n-gram lookup
3. Cache storage: O(1) hash table insert
4. Result retrieval: O(1) hash table lookup

**Contrast with naive approach** (what my original plan almost did):
```python
# WRONG - O(context_length) per word
full_text = f"{context} {word}"
score = kenlm.score(full_text)  # Rescans entire string!
```

### 2.6 Key Optimization: History Pruning (from pyctcdecode)

pyctcdecode includes an important optimization (`decoder.py:227-258`):

```python
def _prune_history(beams: List[LMBeam], lm_order: int) -> List[Beam]:
    """Filter out beams that differ only beyond the N-gram context window."""
    min_n_history = max(1, lm_order - 1)

    for lm_beam in beams:
        # Hash based on only the last N-1 words (all that matters for N-gram)
        hash_idx = (
            tuple(lm_beam.text.split()[-min_n_history:]),
            lm_beam.partial_word,
            lm_beam.last_char,
        )
```

**Why this matters**: For an N-gram LM, only the last N-1 words affect future scores. Beams that differ only in earlier words can be pruned to the highest-scoring one, dramatically reducing beam count.

**For your implementation**: Consider adding this pruning after word boundary scoring. It's optional but can significantly speed up decoding.

### 2.6 Relationship to Existing Components

```
                    ┌──────────────────────────┐
                    │  BatchedBeamCTCComputer  │
                    └───────────┬──────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  fusion_models  │  │    word_ngram_lm    │  │      lm_fusion      │
│  (phoneme LM)   │  │  (NEW: word N-gram) │  │   (neural LM)       │
├─────────────────┤  ├─────────────────────┤  ├─────────────────────┤
│ Every frame     │  │ Word boundaries     │  │ Word boundaries     │
│ Token-level     │  │ Word-level          │  │ Word-level          │
│ State: int      │  │ State: kenlm.State  │  │ State: text string  │
│ O(V) per frame  │  │ O(1) per word       │  │ O(context) per word │
│ BEFORE pruning  │  │ BEFORE pruning      │  │ AFTER pruning       │
└─────────────────┘  └─────────────────────┘  └─────────────────────┘
```

### 2.7 Pre-Pruning vs Post-Pruning LM Application

**Key Design Decision**: The word N-gram LM is applied **BEFORE** beam pruning (topk selection), similar to the phoneme LM (`fusion_models`). This is different from the neural LM (`lm_fusion`) which applies after pruning.

**Why pre-pruning is better for word N-gram LM:**
1. **Rescue effect**: Good word-level LM scores can save hypotheses that might be pruned based on acoustic scores alone
2. **Fair competition**: All candidate tokens compete with their full scores (acoustic + LM)
3. **Consistency**: Matches how phoneme LM works, making the architecture more uniform
4. **Better WER**: Prevents good word sequences from being prematurely eliminated

**How it works:**
```
Frame loop:
  1. Expand beams with all vocab tokens → log_probs [B, beam, vocab]
  2. Add phoneme LM scores to log_probs (fusion_models)
  3. For tokens that complete words:
     - Look up parent word context from cache
     - Score completed word with word N-gram LM
     - Add word LM score to log_probs for that (beam, token) position
  4. topk selection → selects beam_size best candidates (NOW INCLUDING WORD LM SCORES!)
  5. add_results_(), recombine_hyps_()
  6. Update word LM cache with newly completed words
```

---

## 3. Implementation Components

### 3.1 WordNGramLMFusion Class

**File**: `src/brainaudio/inference/decoder/word_ngram_lm.py` (new file)

```python
"""Fast word-level N-gram LM using KenLM with state-based incremental scoring."""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
from pathlib import Path

import kenlm


# Constants from pyctcdecode
LOG_BASE_CHANGE_FACTOR = 1.0 / 0.4342944819  # log10 to ln conversion


@dataclass
class KenlmState:
    """Wrapper around kenlm.State for type safety and serialization."""
    _state: "kenlm.State"

    @property
    def state(self) -> "kenlm.State":
        return self._state

    def copy(self) -> "KenlmState":
        """Create a copy of this state."""
        new_state = kenlm.State()
        # KenLM states can be copied by scoring with empty context
        # Actually, we need to track this differently - see implementation notes
        return KenlmState(new_state)


class WordNGramLMFusion:
    """
    Fast word-level N-gram LM for CTC beam search.

    Uses KenLM's state-based scoring for O(1) incremental word scoring.
    Designed to work at word boundaries detected by LexiconConstraint.

    Example:
        >>> lm = WordNGramLMFusion.from_arpa("path/to/model.arpa", alpha=0.5)
        >>> state = lm.get_start_state()
        >>> score, new_state = lm.score_word(state, "hello")
        >>> score2, new_state2 = lm.score_word(new_state, "world")
    """

    def __init__(
        self,
        kenlm_model: "kenlm.Model",
        unigrams: Optional[Set[str]] = None,
        alpha: float = 0.5,
        beta: float = 0.0,
        unk_score_offset: float = -10.0,
        score_boundary: bool = True,
    ):
        """
        Initialize the word N-gram LM.

        Args:
            kenlm_model: Loaded KenLM model instance.
            unigrams: Set of known vocabulary words (for OOV detection).
            alpha: LM weight for shallow fusion. Score = alpha * lm_score + beta.
            beta: Word insertion bonus (added per word).
            unk_score_offset: Penalty for out-of-vocabulary words.
            score_boundary: Whether to score <s> and </s> markers.
        """
        self._kenlm_model = kenlm_model
        self._unigram_set = unigrams or set()
        self.alpha = alpha
        self.beta = beta
        self.unk_score_offset = unk_score_offset
        self.score_boundary = score_boundary

    @classmethod
    def from_arpa(
        cls,
        arpa_path: Union[str, Path],
        alpha: float = 0.5,
        beta: float = 0.0,
        unk_score_offset: float = -10.0,
    ) -> "WordNGramLMFusion":
        """
        Load from ARPA file and extract unigrams.

        Args:
            arpa_path: Path to .arpa or .bin KenLM file.
            alpha: LM weight.
            beta: Word insertion bonus.
            unk_score_offset: OOV penalty.
        """
        arpa_path = str(arpa_path)
        kenlm_model = kenlm.Model(arpa_path)

        # Extract unigrams from ARPA file
        unigrams = cls._load_unigrams_from_arpa(arpa_path)

        return cls(
            kenlm_model=kenlm_model,
            unigrams=unigrams,
            alpha=alpha,
            beta=beta,
            unk_score_offset=unk_score_offset,
        )

    @staticmethod
    def _load_unigrams_from_arpa(arpa_path: str) -> Set[str]:
        """Extract unigram vocabulary from ARPA file."""
        unigrams = set()

        # Handle binary files - can't extract unigrams
        if arpa_path.endswith('.bin') or arpa_path.endswith('.binary'):
            return unigrams

        with open(arpa_path, 'r', encoding='utf-8') as f:
            in_unigrams = False
            for line in f:
                line = line.strip()
                if line == "\\1-grams:":
                    in_unigrams = True
                    continue
                elif line == "\\2-grams:":
                    break
                elif in_unigrams and line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        unigrams.add(parts[1])

        return unigrams

    @property
    def order(self) -> int:
        """Get the N-gram order (e.g., 5 for 5-gram)."""
        return self._kenlm_model.order

    def get_start_state(self) -> KenlmState:
        """
        Get initial LM state for beginning of sentence.

        Returns:
            KenlmState initialized with <s> context if score_boundary=True,
            otherwise null context.
        """
        state = kenlm.State()
        if self.score_boundary:
            self._kenlm_model.BeginSentenceWrite(state)
        else:
            self._kenlm_model.NullContextWrite(state)
        return KenlmState(state)

    def score_word(
        self,
        prev_state: KenlmState,
        word: str,
        is_last_word: bool = False,
    ) -> Tuple[float, KenlmState]:
        """
        Score a single word given the previous LM state.

        This is O(1) - just a hash table lookup in the N-gram model.

        Args:
            prev_state: LM state encoding the previous N-1 words.
            word: The word to score.
            is_last_word: If True, also adds </s> score.

        Returns:
            Tuple of (weighted_score, new_state).
            weighted_score = alpha * log_prob + beta
        """
        end_state = kenlm.State()

        # Core KenLM scoring - O(1) hash lookup
        raw_score = self._kenlm_model.BaseScore(prev_state.state, word, end_state)

        # Handle OOV words
        if self._unigram_set and word not in self._unigram_set:
            raw_score += self.unk_score_offset
        elif word not in self._kenlm_model:
            raw_score += self.unk_score_offset

        # Add EOS score if this is the last word
        if is_last_word and self.score_boundary:
            eos_state = kenlm.State()
            raw_score += self._kenlm_model.BaseScore(end_state, "</s>", eos_state)

        # Apply weighting: alpha * log10_score * ln_conversion + beta
        weighted_score = self.alpha * raw_score * LOG_BASE_CHANGE_FACTOR + self.beta

        return weighted_score, KenlmState(end_state)

    def score_words_batch(
        self,
        prev_states: List[KenlmState],
        words: List[str],
        is_last_word: bool = False,
    ) -> List[Tuple[float, KenlmState]]:
        """
        Score multiple words in batch (still sequential, but cleaner API).

        Args:
            prev_states: List of previous LM states.
            words: List of words to score (one per state).
            is_last_word: Whether these are sentence-final words.

        Returns:
            List of (score, new_state) tuples.
        """
        results = []
        for state, word in zip(prev_states, words):
            score, new_state = self.score_word(state, word, is_last_word)
            results.append((score, new_state))
        return results

    def score_candidates(
        self,
        prev_state: KenlmState,
        candidate_words: List[str],
        is_last_word: bool = False,
    ) -> List[Tuple[float, KenlmState]]:
        """
        Score multiple candidate words from the same state.

        Useful for scoring homophones at a word boundary.

        Args:
            prev_state: The shared previous LM state.
            candidate_words: List of candidate words (e.g., ["aunt", "ant"]).
            is_last_word: Whether this is the last word.

        Returns:
            List of (score, new_state) for each candidate.
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
            state: Current LM state.

        Returns:
            Weighted EOS score.
        """
        if not self.score_boundary:
            return 0.0

        eos_state = kenlm.State()
        raw_score = self._kenlm_model.BaseScore(state.state, "</s>", eos_state)
        return self.alpha * raw_score * LOG_BASE_CHANGE_FACTOR
```

### 3.2 LM Score Cache (Following pyctcdecode Pattern)

**Key insight**: Instead of storing state per beam, use a **text-keyed cache** that persists across the entire decoding process. This is exactly how pyctcdecode does it (`decoder.py:121-126`).

**File**: `src/brainaudio/inference/decoder/word_ngram_lm.py` (add to same file as WordNGramLMFusion)

```python
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Type aliases following pyctcdecode convention
WordLMCacheKey = Tuple[str, bool]  # (text, is_eos)
WordLMCacheValue = Tuple[float, float, "KenlmState"]  # (combined_score, raw_score, state)
WordLMCache = Dict[WordLMCacheKey, WordLMCacheValue]


def create_word_lm_cache(word_lm: "WordNGramLMFusion") -> WordLMCache:
    """
    Initialize LM cache with start state for empty text.

    This mirrors pyctcdecode's initialization (decoder.py:621-625):
        cached_lm_scores = {("", False): (0.0, 0.0, start_state)}
    """
    start_state = word_lm.get_start_state()
    return {("", False): (0.0, 0.0, start_state)}


def score_word_with_cache(
    cache: WordLMCache,
    word_lm: "WordNGramLMFusion",
    parent_text: str,
    new_word: str,
    is_eos: bool = False,
) -> Tuple[float, float, "KenlmState"]:
    """
    Score a word using cached parent state, following pyctcdecode pattern.

    This mirrors pyctcdecode's _get_lm_beams (decoder.py:383-395):
        _, prev_raw_lm_score, start_state = cached_lm_scores[(beam.text, False)]
        score, end_state = language_model.score(start_state, beam.next_word, ...)
        raw_lm_score = prev_raw_lm_score + score
        cached_lm_scores[cache_key] = (lm_hw_score, raw_lm_score, end_state)

    Args:
        cache: The LM score cache (text -> (score, raw_score, state))
        word_lm: The WordNGramLMFusion instance
        parent_text: The text before this word (used to look up parent state)
        new_word: The word to score
        is_eos: Whether this is end-of-sentence scoring

    Returns:
        Tuple of (combined_score, raw_score, new_state)
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
        raise KeyError(f"Parent text '{parent_text}' not in cache. "
                      f"Cache keys: {list(cache.keys())[:5]}...")

    _, prev_raw_score, parent_state = cache[parent_key]

    # Score the new word (O(1) operation!)
    word_score, new_state = word_lm.score_word(parent_state, new_word, is_last_word=is_eos)

    # Accumulate scores
    new_raw_score = prev_raw_score + word_score
    combined_score = new_raw_score  # Can add hotword scores here if needed

    # Cache the result
    cache[cache_key] = (combined_score, new_raw_score, new_state)

    return combined_score, new_raw_score, new_state
```

**No changes needed to BatchedBeamHyps!** The existing `context_texts[b][k]` already stores `(score, text)` tuples. The text serves as the key into the cache.

### 3.3 Pre-Pruning LM Fusion Function (NEW APPROACH)

**File**: `src/brainaudio/inference/decoder/word_ngram_lm.py`

This function applies word N-gram LM scores **BEFORE** topk selection. It modifies `log_probs` in-place to include word LM scores for tokens that complete words.

```python
def apply_word_ngram_lm_pre_pruning(
    word_lm: "WordNGramLMFusion",
    word_lm_cache: "WordLMCache",
    lexicon: "LexiconConstraint",
    lexicon_state: torch.Tensor,  # [B, beam_size] - current lexicon state per beam
    log_probs: torch.Tensor,      # [B, beam_size, vocab_size] - acoustic scores to modify
    beam_hyps: "BatchedBeamHyps",
    blank_index: int,
    prev_last_labels: torch.Tensor,  # [B, beam_size]
) -> torch.Tensor:
    """
    Apply word-level N-gram LM scores BEFORE beam pruning (topk selection).

    This function identifies which (beam, token) combinations would complete a word,
    computes the word-level LM score for those completions, and adds the score to
    log_probs so that topk selection considers word-level LM information.

    Key insight: Unlike post-pruning approaches, this allows good word-level LM
    scores to "rescue" hypotheses that might otherwise be pruned based on
    acoustic scores alone.

    Args:
        word_lm: WordNGramLMFusion instance for scoring.
        word_lm_cache: Text-keyed cache {(text, is_eos): (score, raw, state)}.
        lexicon: LexiconConstraint for word boundary detection.
        lexicon_state: Current lexicon state for each beam.
        log_probs: [B, beam_size, vocab_size] - scores to modify in-place.
        beam_hyps: Beam hypotheses (for context_texts lookup).
        blank_index: CTC blank token index.
        prev_last_labels: [B, beam_size] - previous last non-blank token per beam.

    Returns:
        Modified log_probs tensor with word LM scores added for word-completing tokens.
    """
    if word_lm is None or lexicon is None:
        return log_probs

    batch_size, beam_size, vocab_size = log_probs.shape
    boundary_token = lexicon.word_boundary_token

    # For each beam, check which tokens would complete a word
    # A token completes a word if:
    #   1. The token is the word boundary token (e.g., space)
    #   2. The previous token was NOT the boundary token (otherwise we're between words)

    # We need to find which beams are "mid-word" (could complete a word with boundary token)
    was_not_boundary = (prev_last_labels != boundary_token)  # [B, beam_size]

    # Get the word that would be completed if we emit boundary_token
    # This requires looking up the lexicon state to find which words match the current prefix

    for b in range(batch_size):
        for k in range(beam_size):
            if not was_not_boundary[b, k]:
                continue  # Already at word boundary, no word to complete

            # Get the current lexicon state for this beam
            state = lexicon_state[b, k].item()

            # Check if this state represents a complete word
            # (i.e., emitting boundary_token would complete a valid word)
            word_indices = lexicon.get_words_at_state(state)

            if not word_indices:
                continue  # No valid word at this state

            # Get parent context text for this beam
            context_tuples = beam_hyps.context_texts[b][k]
            if not context_tuples:
                parent_text = ""
            else:
                parent_text = context_tuples[0][1]  # Best homophone's text

            # Score each possible word completion
            best_word_lm_delta = float('-inf')

            for word_idx in word_indices:
                word = lexicon.word_list[word_idx]

                try:
                    # Look up or compute word LM score
                    combined_score, _, _ = score_word_with_cache(
                        cache=word_lm_cache,
                        word_lm=word_lm,
                        parent_text=parent_text,
                        new_word=word,
                        is_eos=False,
                    )

                    # Compute delta from parent score
                    parent_score = context_tuples[0][0] if context_tuples else 0.0
                    word_lm_delta = combined_score - parent_score

                    if word_lm_delta > best_word_lm_delta:
                        best_word_lm_delta = word_lm_delta

                except KeyError:
                    continue  # Parent text not in cache, skip

            # Add the best word LM score to the boundary token position
            if best_word_lm_delta > float('-inf'):
                log_probs[b, k, boundary_token] += best_word_lm_delta

    return log_probs


def update_word_lm_cache_post_selection(
    word_lm: "WordNGramLMFusion",
    word_lm_cache: "WordLMCache",
    lexicon: "LexiconConstraint",
    beam_hyps: "BatchedBeamHyps",
    blank_index: int,
    boundary_token: int,
    next_labels: torch.Tensor,
    prev_last_labels: torch.Tensor,
) -> None:
    """
    Update the word LM cache after beam selection with newly completed words.

    This function runs AFTER topk selection and updates context_texts for beams
    that just completed a word. It also ensures the cache contains entries for
    all current beam texts.

    Args:
        word_lm: WordNGramLMFusion instance.
        word_lm_cache: Text-keyed cache to update.
        lexicon: LexiconConstraint for word lookup.
        beam_hyps: Beam hypotheses to update.
        blank_index: CTC blank token index.
        boundary_token: Word boundary token index.
        next_labels: [B, beam_size] - tokens selected at this frame.
        prev_last_labels: [B, beam_size] - previous last non-blank tokens.
    """
    if word_lm is None or lexicon is None:
        return

    from .beam_helpers import materialize_beam_transcripts_batched, collapse_ctc_sequence

    batch_size, beam_size = beam_hyps.scores.shape

    # Identify beams that just completed a word
    valid_mask = beam_hyps.scores != float('-inf')
    at_boundary = (next_labels == boundary_token)
    was_not_boundary = (prev_last_labels != boundary_token)
    completed_word_mask = valid_mask & at_boundary & was_not_boundary

    batch_indices, beam_indices = torch.where(completed_word_mask)
    if len(batch_indices) == 0:
        return

    # Batch fetch transcripts
    batch_indices_list = batch_indices.tolist()
    beam_indices_list = beam_indices.tolist()
    all_transcripts = materialize_beam_transcripts_batched(
        beam_hyps, batch_indices_list, beam_indices_list
    )

    # Cache for lexicon lookups
    lexicon_cache = {}
    num_homophone_beams = beam_hyps.num_homophone_beams

    for i, (b, k) in enumerate(zip(batch_indices_list, beam_indices_list)):
        seq_raw = all_transcripts[i]
        seq_ctc = collapse_ctc_sequence(seq_raw.tolist(), blank_index)

        seq_key = tuple(seq_ctc)
        if seq_key in lexicon_cache:
            at_boundary_flag, word_indices = lexicon_cache[seq_key]
        else:
            _, at_boundary_flag, word_indices = lexicon.get_valid_next_tokens_with_word_info(seq_ctc)
            lexicon_cache[seq_key] = (at_boundary_flag, word_indices)

        if not at_boundary_flag or not word_indices:
            continue

        # Get current context texts
        parent_tuples = beam_hyps.context_texts[b][k]

        # Get candidate words (homophones)
        base_words = [lexicon.word_list[idx] for idx in word_indices]
        candidate_words = []
        for word in base_words:
            candidate_words.extend(get_capitalization_variants(word))
        candidate_words = list(dict.fromkeys(candidate_words))

        # Score all candidate words and update context_texts
        all_candidates = []

        for prev_lm_score, prev_text in parent_tuples:
            for word in candidate_words:
                try:
                    combined_score, raw_score, _ = score_word_with_cache(
                        cache=word_lm_cache,
                        word_lm=word_lm,
                        parent_text=prev_text,
                        new_word=word,
                        is_eos=False,
                    )
                except KeyError:
                    continue

                new_text = f"{prev_text} {word}".strip() if prev_text else word
                all_candidates.append((combined_score, new_text))

        if not all_candidates:
            continue

        # Deduplicate by lowercase text, keeping best score
        lowercase_to_best = {}
        for lm_score, text in all_candidates:
            text_lower = text.lower()
            if text_lower not in lowercase_to_best or lm_score > lowercase_to_best[text_lower][0]:
                lowercase_to_best[text_lower] = (lm_score, text)
        all_candidates = list(lowercase_to_best.values())

        # Sort and keep top candidates
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = all_candidates[:num_homophone_beams]

        # Update context_texts
        new_tuples = [(score, text) for score, text in top_candidates]
        beam_hyps.context_texts[b][k] = new_tuples
        beam_hyps.context_texts_hash[b][k] = hash(new_tuples[0][1])


def get_capitalization_variants(word: str) -> List[str]:
    """Generate capitalization variants for a word."""
    variants = [word.lower()]
    if word[0].isupper():
        variants.append(word)
    if word.lower() != word.capitalize():
        variants.append(word.capitalize())
    return list(dict.fromkeys(variants))  # Remove duplicates, preserve order
```

### 3.4 Required Lexicon Method

The pre-pruning approach requires a new method on `VectorizedLexiconConstraint`:

**File**: `src/brainaudio/inference/decoder/lexicon_constraint.py`

```python
def get_words_at_state(self, state: int) -> List[int]:
    """
    Get word indices for all valid words that can be completed at this state.

    Args:
        state: Current lexicon FST state.

    Returns:
        List of word indices from self.word_list that are valid completions.
        Empty list if no words complete at this state.
    """
    # The state encodes the current prefix in the lexicon trie/FST
    # We need to find which words have this prefix and are at a word boundary

    # Check if this state has a word boundary arc (can complete a word)
    word_indices = []
    for arc_label, (next_state, word_idx) in self.transitions[state].items():
        if arc_label == self.word_boundary_token and word_idx is not None:
            word_indices.append(word_idx)

    return word_indices
```

**Note**: The exact implementation depends on how `VectorizedLexiconConstraint` stores word indices. You may need to modify this based on the actual data structure. The key is to quickly look up which words are valid completions at a given lexicon state.

### 3.5 Key Differences from Post-Pruning Approach

| Aspect | Post-Pruning (Original) | Pre-Pruning (New) |
|--------|------------------------|-------------------|
| When applied | After topk selection | Before topk selection |
| Modifies | beam_hyps.scores directly | log_probs tensor |
| Rescue effect | No - beams already pruned | Yes - LM can save good hypotheses |
| Complexity | O(beam_size) per frame | O(beam_size) per frame |
| Cache updates | During scoring | Separate post-selection step |

---

## 4. Detailed Implementation Steps

### Step 1: Create WordNGramLMFusion Class

**File to create**: `src/brainaudio/inference/decoder/word_ngram_lm.py`

1. Copy the class definition from Section 3.1
2. Add imports: `import kenlm` and necessary typing imports
3. Test standalone:
   ```python
   lm = WordNGramLMFusion.from_arpa("path/to/model.arpa")
   state = lm.get_start_state()
   score1, state1 = lm.score_word(state, "hello")
   score2, state2 = lm.score_word(state1, "world")
   print(f"hello: {score1}, world: {score2}")
   ```

### Step 2: No Changes to BatchedBeamHyps Needed!

**Key insight from pyctcdecode**: The text-based cache approach means we don't need to add state tracking to `BatchedBeamHyps`. The existing `context_texts[b][k]` already stores `(score, text)` tuples, and the text serves as the key into the LM cache.

This is simpler and more robust because:
- No state gathering needed when beams are reordered
- Cache naturally handles beam merging (same text = same cache entry)
- No risk of state/text getting out of sync

### Step 3: Add apply_word_ngram_fusion_post_selection Function

**File to modify**: `src/brainaudio/inference/decoder/neural_lm_fusion.py`

1. Add the function from Section 3.3
2. Add import at top: `from .word_ngram_lm import WordNGramLMFusion, KenlmState`

### Step 4: Integrate into BatchedBeamCTCComputer (PRE-PRUNING)

**File to modify**: `src/brainaudio/inference/decoder/ctc_batched_beam_decoding.py`

**IMPORTANT**: The word N-gram LM is applied **BEFORE** topk selection (pruning), not after. This allows word-level LM scores to influence which beams survive pruning.

1. Add new parameter to `__init__`:
   ```python
   def __init__(
       self,
       ...,
       word_ngram_lm: Optional["WordNGramLMFusion"] = None,
       word_ngram_lm_weight: Optional[float] = None,  # Override lm.alpha if set
   ):
       self.word_ngram_lm = word_ngram_lm
       if word_ngram_lm_weight is not None and word_ngram_lm is not None:
           word_ngram_lm.alpha = word_ngram_lm_weight
   ```

2. **Create cache at start of `batched_beam_search_torch()`** (following pyctcdecode pattern):
   ```python
   # After creating batched_beam_hyps (around line 369):

   # Initialize word LM cache (following pyctcdecode decoder.py:617-625)
   word_lm_cache = None
   if self.word_ngram_lm is not None:
       from .word_ngram_lm import create_word_lm_cache
       word_lm_cache = create_word_lm_cache(self.word_ngram_lm)
       # Cache starts as: {("", False): (0.0, 0.0, start_state)}
   ```

3. **Apply word N-gram LM scores BEFORE topk** (around line 490, after phoneme LM):
   ```python
   # After fusion_models scoring (phoneme LM), BEFORE topk selection:
   # This is the key difference - we add word LM scores to log_probs before pruning!

   if self.word_ngram_lm is not None and self.lexicon is not None:
       from .word_ngram_lm import apply_word_ngram_lm_pre_pruning

       log_probs = apply_word_ngram_lm_pre_pruning(
           word_lm=self.word_ngram_lm,
           word_lm_cache=word_lm_cache,
           lexicon=self.lexicon,
           lexicon_state=lexicon_state,
           log_probs=log_probs,  # [B, beam_size, vocab_size]
           beam_hyps=batched_beam_hyps,
           blank_index=self._blank_index,
           prev_last_labels=batched_beam_hyps.last_label,
       )

   # NOW do topk selection (with word LM scores already included!)
   next_scores, next_candidates_indices = torch.topk(
       log_probs.view(curr_batch_size, -1), k=self.beam_size, largest=True, sorted=True
   )
   ```

4. **Update cache after beam selection** (after add_results_, around line 668):
   ```python
   # After: batched_beam_hyps.add_results_(next_indices, next_labels, next_scores)
   # Before: batched_beam_hyps.recombine_hyps_(...)

   # Update word LM cache with newly completed words
   if self.word_ngram_lm is not None and self.lexicon is not None:
       from .word_ngram_lm import update_word_lm_cache_post_selection

       update_word_lm_cache_post_selection(
           word_lm=self.word_ngram_lm,
           word_lm_cache=word_lm_cache,
           lexicon=self.lexicon,
           beam_hyps=batched_beam_hyps,
           blank_index=self._blank_index,
           boundary_token=self.lexicon.word_boundary_token,
           next_labels=next_labels,
           prev_last_labels=prev_last_labels,
       )

   batched_beam_hyps.recombine_hyps_(is_last_step= curr_max_time-1 == frame_idx)
   ```

5. **EOS scoring using cache** (around line 698):
   ```python
   # Simpler: use get_eos_score directly on cached state
   if self.word_ngram_lm is not None and word_lm_cache is not None:
       for b in range(curr_batch_size):
           for k in range(self.beam_size):
               if batched_beam_hyps.scores[b, k] != float('-inf'):
                   context_tuples = batched_beam_hyps.context_texts[b][k]
                   if context_tuples:
                       _, text = context_tuples[0]
                       cache_entry = word_lm_cache.get((text, False))
                       if cache_entry:
                           _, _, state = cache_entry
                           eos_score = self.word_ngram_lm.get_eos_score(state)
                           batched_beam_hyps.scores[b, k] += eos_score
   ```

### Step 5: No State Gathering Needed!

With the text-based cache approach, we don't need to modify `recombine_hyps_()`.

**Why it just works**:
- When beams are recombined, their `context_texts` are updated
- The text in `context_texts` serves as the key into `word_lm_cache`
- The cache already has entries for all texts we've scored
- No explicit state copying or gathering required

This is a major simplification compared to the per-beam state tracking approach.

---

## 5. Integration with Existing Code

### 5.1 Interaction with Neural LM

You can use **both** word N-gram and neural LM together:

```python
decoder = BatchedBeamCTCComputer(
    ...,
    lexicon=lexicon,
    word_ngram_lm=WordNGramLMFusion.from_arpa("model.arpa", alpha=0.3),
    lm_fusion=HuggingFaceLMFusion(model, tokenizer, weight=0.2),
)
```

Execution order in the frame loop:
1. **Phoneme N-gram LM** (`fusion_models`) - scores all tokens before pruning
2. **Word N-gram LM** (`word_ngram_lm`) - scores word-completing tokens before pruning
3. **topk selection** (pruning) - selects beam_size best candidates
4. **Neural LM** (`lm_fusion`) - scores completed words after pruning (optional refinement)

The key insight is that both phoneme and word N-gram LMs run **before pruning**, allowing their scores to influence which beams survive. The neural LM runs **after pruning** for computational efficiency (it's much slower).

### 5.2 Interaction with Phoneme N-gram LM

The `fusion_models` (phoneme-level) and `word_ngram_lm` operate at different granularities but both apply **before pruning**:

- `fusion_models`: Every frame, scores all tokens, guides towards likely phoneme sequences
- `word_ngram_lm`: Every frame, scores word-boundary tokens only, guides towards likely word sequences

Both can be used together for complementary benefits. The phoneme LM helps with within-word phoneme selection, while the word LM helps with word-level disambiguation (homophones).

### 5.3 Parameter Tuning

Recommended starting points:

| Parameter | Word N-gram | Neural LM | Phoneme N-gram |
|-----------|-------------|-----------|----------------|
| alpha/weight | 0.3-0.5 | 0.1-0.3 | 0.1-0.2 |
| beta (word bonus) | 0.0-1.0 | 0.0-0.5 | N/A |
| unk_penalty | -10.0 | N/A | N/A |

---

## 6. Performance Considerations

### 6.1 Complexity Analysis

| Operation | Naive (string-based) | State-based |
|-----------|---------------------|-------------|
| Score 1 word | O(context_length) | O(1) |
| Score all homophones | O(context × homophones) | O(homophones) |
| Full sentence (N words) | O(N² × beams × homophones) | O(N × beams × homophones) |

For a 50-word sentence with beam_size=16 and 3 homophones/word:
- Naive: 50 × 50 × 16 × 3 = 120,000 context-words processed
- State-based: 50 × 16 × 3 = 2,400 words processed
- **50x speedup**

### 6.2 Memory Overhead

Each `kenlm.State` is approximately 64 bytes (depends on N-gram order).

Per beam: `num_homophone_beams` states × 64 bytes
Total: `batch_size × beam_size × num_homophone_beams × 64` bytes

Example: 32 batch × 16 beam × 4 homophones × 64 bytes = 128 KB (negligible)

### 6.3 KenLM Optimizations

1. **Use binary format**: Convert ARPA to binary for faster loading:
   ```bash
   build_binary model.arpa model.bin
   ```

2. **Probing vs Trie**: Binary format uses probing hash tables (faster queries)

3. **Memory mapping**: KenLM memory-maps binary files, reducing RAM usage

### 6.4 Python Overhead

The main bottleneck is Python loop overhead, not KenLM scoring. Potential optimizations:

1. **Batch state operations**: Group all score_word calls and use numpy/torch for bookkeeping
2. **Cython wrapper**: Create Cython bindings for hot loop
3. **State caching**: Cache states for common prefixes (e.g., "the", "a", "I")

---

## 7. Testing Strategy

### 7.1 Unit Tests

**Test 1: KenLM State Consistency**
```python
def test_state_based_vs_full_scoring():
    lm = WordNGramLMFusion.from_arpa("test.arpa")

    # Score incrementally
    state = lm.get_start_state()
    total_incremental = 0
    for word in ["hello", "world"]:
        score, state = lm.score_word(state, word)
        total_incremental += score

    # Score full sentence (for comparison)
    import kenlm
    raw_model = kenlm.Model("test.arpa")
    full_score = raw_model.score("hello world")

    # Should be approximately equal (accounting for alpha/beta)
    assert abs(total_incremental - full_score * lm.alpha) < 0.01
```

**Test 2: Homophone Scoring**
```python
def test_homophone_scoring():
    lm = WordNGramLMFusion.from_arpa("test.arpa")
    state = lm.get_start_state()

    # Score "I saw my"
    for word in ["I", "saw", "my"]:
        _, state = lm.score_word(state, word)

    # Score homophones
    results = lm.score_candidates(state, ["aunt", "ant"])

    # "aunt" should score higher after "my" (my aunt vs my ant)
    assert results[0][0] > results[1][0]
```

### 7.2 Integration Tests

**Test 3: Full Decoding Pipeline**
```python
def test_decoding_with_word_ngram():
    decoder = BatchedBeamCTCComputer(
        blank_index=0,
        beam_size=16,
        lexicon=lexicon,
        word_ngram_lm=WordNGramLMFusion.from_arpa("lm.arpa", alpha=0.5),
    )

    # Decode some audio
    logits = torch.randn(1, 100, 41)  # [B, T, V]
    lengths = torch.tensor([100])

    result = decoder(logits, lengths)

    # Check that states were tracked
    assert result.word_lm_states is not None
    assert len(result.word_lm_states[0][0]) > 0
```

### 7.3 Performance Tests

**Test 4: Speed Comparison**
```python
def test_speed_improvement():
    import time

    # Naive approach (string-based)
    naive_lm = NaiveWordLM("lm.arpa")
    start = time.time()
    for _ in range(1000):
        naive_lm.score_full_context("this is a test sentence", "word")
    naive_time = time.time() - start

    # State-based approach
    state_lm = WordNGramLMFusion.from_arpa("lm.arpa")
    state = state_lm.get_start_state()
    for word in "this is a test sentence".split():
        _, state = state_lm.score_word(state, word)

    start = time.time()
    for _ in range(1000):
        state_lm.score_word(state, "word")
    state_time = time.time() - start

    print(f"Naive: {naive_time:.3f}s, State-based: {state_time:.3f}s")
    assert state_time < naive_time / 10  # At least 10x faster
```

---

## 8. Future Optimizations

### 8.1 State Caching for Common Prefixes

Cache LM states for frequent word sequences:

```python
class CachedWordNGramLMFusion(WordNGramLMFusion):
    def __init__(self, ...):
        super().__init__(...)
        self._prefix_cache = {}  # {("the",): state, ("a",): state, ...}
        self._warm_cache()

    def _warm_cache(self):
        # Pre-compute states for top 1000 unigrams
        for word in self.top_unigrams[:1000]:
            state = self.get_start_state()
            _, end_state = self.score_word(state, word)
            self._prefix_cache[(word,)] = end_state
```

### 8.2 Batch Scoring with Numpy

Vectorize state operations:

```python
def score_words_vectorized(self, states, words):
    # Use numpy for bookkeeping, call KenLM in tight loop
    scores = np.zeros(len(words))
    new_states = []

    for i, (state, word) in enumerate(zip(states, words)):
        scores[i], new_state = self.score_word(state, word)
        new_states.append(new_state)

    return scores, new_states
```

### 8.3 CUDA Graphs Compatibility

For CUDA graphs mode, pre-allocate state storage:

```python
class StaticWordLMStates:
    """Pre-allocated state storage for CUDA graphs compatibility."""

    def __init__(self, batch_size, beam_size, num_homophones):
        # States are CPU-only (KenLM), but indices can be on GPU
        self.states = [[
            [kenlm.State() for _ in range(num_homophones)]
            for _ in range(beam_size)
        ] for _ in range(batch_size)]
```

### 8.4 Multi-LM Ensemble

Support multiple N-gram models (e.g., domain-specific + general):

```python
class EnsembleWordNGramLM:
    def __init__(self, models: List[WordNGramLMFusion], weights: List[float]):
        self.models = models
        self.weights = weights

    def score_word(self, states, word):
        total_score = 0
        new_states = []
        for model, state, weight in zip(self.models, states, self.weights):
            score, new_state = model.score_word(state, word)
            total_score += weight * score
            new_states.append(new_state)
        return total_score, new_states
```

---

## Appendix A: File Changes Summary

| File | Changes |
|------|---------|
| `word_ngram_lm.py` | **NEW** - WordNGramLMFusion class, KenlmState wrapper, cache utilities, `apply_word_ngram_lm_pre_pruning()`, `update_word_lm_cache_post_selection()` |
| `batched_beam_decoding_utils.py` | **NO CHANGES** - text-based cache eliminates need for state tracking |
| `ctc_batched_beam_decoding.py` | Add `word_ngram_lm` parameter, create cache, call pre-pruning function before topk, call cache update after add_results_, EOS scoring |

**Key Design Decision**: Word N-gram LM is applied **BEFORE** pruning (topk selection), not after. This allows word-level LM scores to influence which beams survive, preventing good word sequences from being prematurely eliminated based on acoustic scores alone.

## Appendix B: Dependencies

```
kenlm>=0.1.0  # pip install https://github.com/kpu/kenlm/archive/master.zip
```

## Appendix C: Example Usage

```python
from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer,
    VectorizedLexiconConstraint,
    WordNGramLMFusion,
)

# Load components
lexicon = VectorizedLexiconConstraint.from_file_paths(
    "tokens.txt", "lexicon.txt"
)
word_lm = WordNGramLMFusion.from_arpa(
    "word_lm.arpa",
    alpha=0.4,
    beta=0.5,
)

# Create decoder
decoder = BatchedBeamCTCComputer(
    blank_index=0,
    beam_size=16,
    lexicon=lexicon,
    word_ngram_lm=word_lm,
)

# Decode
logits = model(audio)  # [B, T, V]
lengths = get_lengths(audio)
result = decoder(logits, lengths)

# Get best hypothesis
best_text = result.context_texts[0][0][0][1]  # batch 0, beam 0, homophone 0, text
print(best_text)
```
