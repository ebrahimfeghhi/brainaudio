# KV Cache Implementation Plan for HuggingFaceLMFusion

## Overview

Implement KV (Key-Value) caching in `score_continuations` to avoid redundantly computing attention for the same context prefix when scoring multiple candidate words.

**Current behavior:** For context "He is also a member of the" with candidates ["royal", "real", "reel"], we run 3 full forward passes, each recomputing attention for the entire 8-word context.

**With KV caching:** Compute attention for the context once, cache it, then only compute attention for each candidate word while reusing the cached context attention.

---

## How KV Caching Works

### Background

In transformer models, each layer computes:
- **K (Key)** and **V (Value)** matrices from the input
- These are used in attention: `Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V`

For autoregressive generation:
- When generating token N+1, we need K and V for tokens 1..N
- These don't change as we generate more tokens
- So we cache them and only compute K, V for the new token

### HuggingFace API

```python
# First call - compute and cache
outputs = model(context_ids, use_cache=True)
past_key_values = outputs.past_key_values  # Tuple of (K, V) per layer

# Subsequent calls - reuse cache
outputs = model(new_token_ids, past_key_values=past_key_values, use_cache=True)
```

---

## Current Code Flow (without caching)

```
score_continuations(contexts, candidate_words):

    1. PREPARE: For each (context, word) pair:
       - Build full_text = truecase(context + word)
       - Compute start_idx from truecased context
       - Flatten into flat_texts list

    2. PROCESS CHUNKS: For each chunk of 32 sequences:
       - Tokenize all full texts
       - Run forward pass on ALL tokens (context + word)
       - Extract log-probs for word tokens only
       - Sum to get word score

    3. RETURN: Reshape scores to match input structure
```

**Problem:** If context "He is a member of the" has 5 candidate words, we tokenize and run forward pass on the full context 5 times.

---

## Proposed Code Flow (with caching)

```
score_continuations(contexts, candidate_words):

    1. GROUP BY CONTEXT:
       - Group candidates by their context
       - contexts_to_candidates = {context: [word1, word2, ...]}

    2. FOR EACH UNIQUE CONTEXT:
       a. Tokenize and truecase the context
       b. Run forward pass with use_cache=True
       c. Save past_key_values (the KV cache)
       d. Get last token logits for first word prediction

       e. FOR EACH CANDIDATE WORD:
          - Tokenize just the word
          - Run forward pass with past_key_values
          - Extract log-probs for word tokens
          - Sum to get score

    3. RETURN: Reshape scores to match input structure
```

---

## Implementation Details

### Step 1: Modify `score_continuations` signature (no change needed)

```python
def score_continuations(self, contexts: List[str], candidate_words: List[List[str]]) -> List[List[float]]:
```

### Step 2: Group candidates by context

```python
# Build mapping: context_idx -> list of (word, flat_idx)
context_groups = {}
flat_idx = 0
for ctx_idx, (context, candidates) in enumerate(zip(contexts, candidate_words)):
    # Apply truecase to context for consistency
    truecased_context = truecase.get_true_case(context) if context else ""

    if truecased_context not in context_groups:
        context_groups[truecased_context] = []

    for word in candidates:
        context_groups[truecased_context].append((word, flat_idx, ctx_idx))
        flat_idx += 1

flat_scores = [0.0] * flat_idx
```

### Step 3: Process each unique context with KV caching

```python
for truecased_context, word_list in context_groups.items():
    # Tokenize context once
    if truecased_context:
        context_ids = self.tokenizer.encode(truecased_context,
                                             add_special_tokens=True,
                                             return_tensors="pt").to(self.device)
    else:
        # Empty context - just use BOS token
        context_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)

    # Forward pass on context to build KV cache
    with torch.no_grad():
        context_outputs = self.model(context_ids, use_cache=True)
        past_kv = context_outputs.past_key_values
        # last_logits for scoring first token of word
        last_hidden = context_outputs.logits[:, -1, :]  # [1, vocab_size]

    # Score each candidate word using cached KV
    for word, flat_idx, ctx_idx in word_list:
        score = self._score_word_with_cache(
            word, truecased_context, past_kv, last_hidden
        )
        flat_scores[flat_idx] = score
```

### Step 4: New helper method `_score_word_with_cache`

```python
def _score_word_with_cache(self, word: str, context: str,
                            past_kv, last_logits) -> float:
    """Score a single word using cached context KV."""

    # Build full text to get proper truecasing of the word
    if not context:
        full_text = word
    else:
        full_text = f"{context} {word}"
    full_text_truecased = truecase.get_true_case(full_text)

    # Extract the truecased word (last word)
    truecased_word = full_text_truecased.rsplit(" ", 1)[-1] if " " in full_text_truecased else full_text_truecased

    # Tokenize just the word (with leading space for proper tokenization)
    # Note: We need the space because "royal" vs " royal" tokenize differently
    word_with_space = " " + truecased_word
    word_ids = self.tokenizer.encode(word_with_space, add_special_tokens=False,
                                      return_tensors="pt").to(self.device)

    # Score first token using cached last_logits
    log_probs = F.log_softmax(last_logits, dim=-1)
    first_token_score = log_probs[0, word_ids[0, 0]].item()

    total_score = first_token_score

    # If word has multiple tokens, continue with KV cache
    if word_ids.shape[1] > 1:
        # Forward pass for remaining tokens using cache
        outputs = self.model(word_ids[:, :-1], past_key_values=past_kv, use_cache=True)

        # Extract log-probs for each subsequent token
        word_logits = outputs.logits  # [1, num_word_tokens-1, vocab_size]
        word_log_probs = F.log_softmax(word_logits, dim=-1)

        for i in range(word_ids.shape[1] - 1):
            next_token_id = word_ids[0, i + 1]
            total_score += word_log_probs[0, i, next_token_id].item()

    return self.weight * total_score + self.word_insertion_bonus
```

---

## Memory Considerations

### KV Cache Size per Context

For Gemma 3 4B:
- 32 layers, 8 KV heads per layer, 256 head dim
- Per token: 32 × 8 × 256 × 2 (K and V) × 2 bytes (float16) = 256 KB
- For 100-token context: ~25 MB per context

### Total Memory with Caching

- Model (4-bit): ~2-3 GB
- KV cache for 50 unique contexts × 100 tokens: ~1.25 GB
- Working memory: ~1 GB
- **Total: ~5-6 GB** (fits easily in 32 GB)

### Chunking for Many Contexts

If there are too many unique contexts, process them in chunks:

```python
CONTEXT_CHUNK_SIZE = 32  # Process 32 unique contexts at a time

context_list = list(context_groups.items())
for chunk_start in range(0, len(context_list), CONTEXT_CHUNK_SIZE):
    chunk = context_list[chunk_start:chunk_start + CONTEXT_CHUNK_SIZE]

    for truecased_context, word_list in chunk:
        # Process as described above
        ...

    # Clear GPU cache between context chunks
    torch.cuda.empty_cache()
```

---

## Edge Cases to Handle

1. **Empty context**: Use BOS token only, no KV cache from context
2. **Very long context**: Truncate to `max_context_length` before caching
3. **Truecase changes word**: Handle case where truecasing the full text changes the word itself
4. **Single-token words**: Skip the incremental forward pass
5. **Model doesn't support KV cache**: Fall back to original implementation

---

## Expected Speedup

| Scenario | Without Cache | With Cache | Speedup |
|----------|---------------|------------|---------|
| 1 context, 5 words | 5 full passes | 1 full + 5 small | ~3-4x |
| 10 contexts, 5 words each | 50 full passes | 10 full + 50 small | ~3-4x |
| 100 contexts, 2 words each | 200 full passes | 100 full + 200 small | ~1.5-2x |

The speedup is proportional to the average number of candidates per unique context.

---

## Files to Modify

1. **`/home/ebrahim/brainaudio/src/brainaudio/inference/decoder/neural_lm_fusion.py`**
   - Modify `score_continuations` method
   - Add `_score_word_with_cache` helper method

2. **`/home/ebrahim/brainaudio/tests/test_tokenization_alignment.py`** (optional)
   - Add tests for KV cache correctness

---

## Implementation Steps

1. Add `_score_word_with_cache` helper method
2. Modify `score_continuations` to group by context
3. Implement context KV caching loop
4. Add fallback for models without cache support
5. Test with existing test cases
6. Benchmark speedup

---

## Potential Issues & Mitigations

### 1. Truecase Inconsistency (IMPORTANT)

**Problem:** Truecasing depends on the full sentence. Adding different words might change how the context is capitalized:
- `truecase("he is also")` → `"he is also"` (lowercase)
- `truecase("he is also happy")` → `"He is also happy"` (capitalized!)

If we cache KV from `truecase(context)` alone, it won't match `truecase(context + word)`.

**Mitigation Strategy:**
```python
# For each context, check if all candidate words produce the same truecased context
def get_canonical_context(context, candidates):
    truecased_contexts = set()
    for word in candidates:
        full = f"{context} {word}".strip()
        truecased_full = truecase.get_true_case(full)
        truecased_ctx = truecased_full.rsplit(" ", 1)[0] if " " in truecased_full else ""
        truecased_contexts.add(truecased_ctx)

    if len(truecased_contexts) == 1:
        # All words produce same truecased context - safe to cache
        return truecased_contexts.pop(), True
    else:
        # Different truecasings - fall back to non-cached approach
        return None, False
```

This way we only use KV caching when it's safe (same truecased context for all candidates).

### 2. Tokenization Boundary

**Problem:** `"royal"` vs `" royal"` tokenize differently.

**Mitigation:** Always prepend space when tokenizing words separately:
```python
word_with_space = " " + truecased_word
word_ids = tokenizer.encode(word_with_space, add_special_tokens=False)
```

### 3. Memory Accumulation

**Problem:** KV caches consume GPU memory.

**Mitigation:** Process contexts in chunks and clear cache between chunks:
```python
for chunk in context_chunks:
    # Process chunk
    del past_kv
    torch.cuda.empty_cache()
```
