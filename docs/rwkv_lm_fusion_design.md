# RWKVLMFusion Class Design

## Purpose
Score candidate words given a context string, returning log probabilities.

The main interface is `score_continuations(contexts, candidate_words)` which is called by the beam search decoder at word boundaries.

---

## State Caching Strategy

**Key insight**: Each beam maintains a context that grows incrementally (e.g., "the" → "the cat" → "the cat sat"). Instead of reprocessing the entire context each time, we cache RWKV states.

### State Storage Structure

```
state_cache: Dict[str, Tuple[state, logits]]
    key: context_text (e.g., "the cat")
    value: (rwkv_state, logits_predicting_next_token)
```

**Why store logits too?** After processing "the cat", the logits predict P(next_token | "the cat"). We need these to score the first token of the next word.

### Memory Budget

- **N** = beam size (e.g., 40)
- **K** = homophones per beam (e.g., 3)
- **Total states** = N × K = 120 states max

Each RWKV state is a list of tensors. For a 2.9B model, each state ≈ few hundred MB, so we may need to limit cache size.

---

## Methods Overview

### 1. `__init__(model, pipeline, weight, ...)`
Store model, pipeline, and initialize state cache.

```python
def __init__(self, model, pipeline, weight=0.3, ...):
    self.model = model
    self.pipeline = pipeline
    self.weight = weight
    self.word_insertion_bonus = word_insertion_bonus

    # State cache: context_text → (state, logits)
    self.state_cache: Dict[str, Tuple[Any, torch.Tensor]] = {}
```

### 2. `score_continuations(contexts, candidate_words)` → Main method

**Input:**
- `contexts`: List of context strings, e.g., `["the cat", "I saw my"]`
- `candidate_words`: List of candidate word lists, e.g., `[["sat", "mat"], ["aunt", "ant"]]`

**Output:**
- `scores`: List of score lists, e.g., `[[-0.5, -1.2], [-0.3, -0.8]]`

**What it does:**
```
For each (context, candidates) pair:
    1. Look up cached state for context, OR compute if missing
    2. For each candidate word:
        a. Score word tokens using cached (state, logits)
        b. Store new state for "context + word" in cache
    3. Return weighted log probabilities
```

### 3. `_get_or_compute_state(context)` → State lookup/computation

```python
def _get_or_compute_state(self, context: str) -> Tuple[state, logits]:
    """
    Get cached state or compute it.

    Returns:
        state: RWKV state after processing context
        logits: Logits predicting next token after context
    """
    if context in self.state_cache:
        return self.state_cache[context]

    # Not cached - compute from scratch
    if context:
        tokens = self.pipeline.encode(context)
        logits, state = self.model.forward(tokens, None)
    else:
        # Empty context - use initial state
        logits, state = self.model.forward([0], None)  # BOS or similar

    self.state_cache[context] = (state, logits)
    return state, logits
```

### 4. `_score_word_and_cache(word, context, state, logits)` → Score + cache new state

```python
def _score_word_and_cache(self, word: str, context: str, state, logits) -> float:
    """
    Score a word given context state, and cache the resulting state.

    Args:
        word: Word to score (e.g., "cat")
        context: Current context text (e.g., "the")
        state: RWKV state after context
        logits: Logits predicting first token of word

    Returns:
        Log probability of word given context
    """
    word_tokens = self.pipeline.encode(" " + word)
    total_log_prob = 0.0

    current_state = state
    current_logits = logits

    for token in word_tokens:
        # Score this token
        log_probs = F.log_softmax(current_logits, dim=-1)
        total_log_prob += log_probs[token].item()

        # Forward to get next state/logits
        current_logits, current_state = self.model.forward([token], current_state)

    # Cache the new state for "context + word"
    new_context = f"{context} {word}".strip()
    self.state_cache[new_context] = (current_state, current_logits)

    return total_log_prob
```

### 5. `clear_cache()` → Reset between trials

```python
def clear_cache(self):
    """Clear state cache. Call between decoding different trials."""
    self.state_cache.clear()
```

---

## Flow Diagram

```
Trial starts → clear_cache()
    │
    ▼
Word boundary detected for beam (context="the cat", candidates=["sat", "mat"])
    │
    ▼
score_continuations(["the cat"], [["sat", "mat"]])
    │
    ├─► _get_or_compute_state("the cat")
    │       │
    │       ├─ Cache hit? → return cached (state, logits)
    │       └─ Cache miss? → forward("the cat"), cache it, return
    │
    ├─► _score_word_and_cache("sat", "the cat", state, logits)
    │       │
    │       ├─ Score tokens of " sat"
    │       ├─ Cache state for "the cat sat"
    │       └─ Return log_prob
    │
    └─► _score_word_and_cache("mat", "the cat", state, logits)
            │
            ├─ Score tokens of " mat"
            ├─ Cache state for "the cat mat"
            └─ Return log_prob
```

---

## Summary

| Method | Purpose |
|--------|---------|
| `__init__` | Store model, pipeline, init cache |
| `score_continuations` | Main API - score candidates with caching |
| `_get_or_compute_state` | Lookup/compute state for context |
| `_score_word_and_cache` | Score word + cache new state |
| `clear_cache` | Reset cache between trials |
