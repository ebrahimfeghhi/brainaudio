"""Neural Language Model fusion with KV caching for efficient beam search decoding."""

from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F

from .neural_lm_fusion import NeuralLanguageModelFusion


class HuggingFaceLMFusionKV(NeuralLanguageModelFusion):
    """
    Neural LM fusion with KV caching for efficient scoring.

    Key optimization: Beams with identical context texts share the same KV cache.
    When scoring candidate words, we reuse cached KV states instead of
    reprocessing the entire context.

    Memory usage: ~3MB per unique context (for LLaMA 3B with 50 token context).
    With beam_size=1000 but only ~50 unique contexts, that's ~150MB vs ~3GB.
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
        max_cache_size: int = 500,
    ):
        """
        Initialize HuggingFace LM fusion with KV caching.

        Args:
            model: HuggingFace causal LM model (e.g., GPT2LMHeadModel, LlamaForCausalLM)
            tokenizer: Corresponding tokenizer
            weight: LM weight for fusion
            homophone_aggregation: 'max' or 'logsumexp'
            device: Device for inference
            max_context_length: Maximum context length in tokens
            word_insertion_bonus: Bonus added per word
            max_cache_size: Maximum number of KV caches to store (LRU eviction)
        """
        super().__init__(weight, homophone_aggregation, device)
        self.model = model
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = "right"
        self.max_context_length = max_context_length
        self.word_insertion_bonus = word_insertion_bonus
        self.max_cache_size = max_cache_size

        self.device = device if device is not None else next(self.model.parameters()).device

        # KV cache storage: maps context_hash -> (context_text, kv_cache, last_logits)
        # last_logits stores the logits for the next token prediction (for single-token words)
        self.kv_cache: Dict[int, Tuple[str, tuple, torch.Tensor]] = {}
        self.cache_access_order: List[int] = []  # For LRU eviction

        # Counters for tracking
        self.llm_call_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

        print(f"[HuggingFaceLMFusionKV] weight={self.weight}, max_cache_size={max_cache_size}")

        # Move model to device and set to eval mode
        if device is not None:
            current_device = next(self.model.parameters()).device
            if current_device != self.device:
                self.model.to(self.device)
        self.model.eval()

    def reset_call_count(self):
        """Reset counters."""
        self.llm_call_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def get_call_count(self) -> int:
        """Get the current LLM call count."""
        return self.llm_call_count

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "llm_calls": self.llm_call_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.kv_cache),
        }

    def clear_cache(self):
        """Clear all cached KV states."""
        self.kv_cache.clear()
        self.cache_access_order.clear()
        torch.cuda.empty_cache()

    def cleanup_unused_cache(self, active_context_texts: List[str]):
        """
        Remove cache entries for contexts no longer in use.
        Call this after beam pruning to free memory from pruned beams.

        Args:
            active_context_texts: List of context strings still in use by surviving beams.
                                  Pass the best text from each beam's context_texts.
        """
        active_hashes = set(hash(ctx) for ctx in active_context_texts)
        to_remove = [h for h in list(self.kv_cache.keys()) if h not in active_hashes]

        for h in to_remove:
            del self.kv_cache[h]
            if h in self.cache_access_order:
                self.cache_access_order.remove(h)

        if to_remove:
            torch.cuda.empty_cache()

    def _evict_lru(self):
        """Evict least recently used cache entry if over capacity."""
        while len(self.kv_cache) >= self.max_cache_size and self.cache_access_order:
            oldest_hash = self.cache_access_order.pop(0)
            if oldest_hash in self.kv_cache:
                del self.kv_cache[oldest_hash]

    def _update_access_order(self, context_hash: int):
        """Update LRU access order."""
        if context_hash in self.cache_access_order:
            self.cache_access_order.remove(context_hash)
        self.cache_access_order.append(context_hash)

    @torch.no_grad()
    def _get_or_create_kv_cache(self, context: str) -> Tuple[tuple, torch.Tensor]:
        """
        Get cached KV state for context, or create it if not cached.

        Returns:
            Tuple of (kv_cache, last_logits) where:
            - kv_cache: tuple of (key, value) pairs for each layer
            - last_logits: logits for next token prediction [1, vocab_size]
        """
        context_hash = hash(context)

        if context_hash in self.kv_cache:
            self.cache_hits += 1
            self._update_access_order(context_hash)
            _, kv_cache, last_logits = self.kv_cache[context_hash]
            return kv_cache, last_logits

        self.cache_misses += 1

        # Tokenize context
        if context:
            input_ids = self.tokenizer.encode(context, add_special_tokens=True, return_tensors="pt")
            input_ids = input_ids.to(self.device)
        else:
            # Empty context - use BOS token
            bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            input_ids = torch.tensor([[bos_id]], device=self.device)

        # Forward pass to get KV cache
        self.llm_call_count += 1
        outputs = self.model(input_ids, use_cache=True)
        kv_cache = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :]  # [1, vocab_size]

        # Store in cache (with LRU eviction)
        self._evict_lru()
        self.kv_cache[context_hash] = (context, kv_cache, last_logits.clone())
        self._update_access_order(context_hash)

        return kv_cache, last_logits

    @torch.no_grad()
    def _score_word_and_cache_extension(
        self,
        context: str,
        word: str,
        kv_cache: tuple,
        last_logits: torch.Tensor
    ) -> float:
        """
        Score a word using cached KV state AND save extended cache for new context.

        This is the key optimization: after scoring "member" with context "He is a",
        we save the extended KV cache for "He is a member" so future scoring is instant.

        Args:
            context: The context string
            word: The word to score
            kv_cache: Cached KV state from context
            last_logits: Logits for next token after context [1, vocab_size]

        Returns:
            Weighted log probability score for the word
        """
        # Build new context string
        if not context:
            new_context = word
        elif context.endswith(" ") or word.startswith(" "):
            new_context = f"{context}{word}"
        elif word and word[0] in '.,!?;:\'"':
            new_context = f"{context}{word}"
        else:
            new_context = f"{context} {word}"

        # Check if we already have cache for the extended context
        new_context_hash = hash(new_context)
        already_cached = new_context_hash in self.kv_cache

        # Tokenize the word (with leading space if context exists)
        if context:
            word_text = " " + word
        else:
            word_text = word
        word_ids = self.tokenizer.encode(word_text, add_special_tokens=False)

        if not word_ids:
            return self.word_insertion_bonus

        # Get log probs from last_logits
        log_probs = F.log_softmax(last_logits, dim=-1)  # [1, vocab_size]

        # First token score from cached logits
        first_token_id = word_ids[0]
        total_score = log_probs[0, first_token_id].item()

        # Process ALL word tokens to get extended KV cache
        all_word_ids = torch.tensor([word_ids], device=self.device)
        self.llm_call_count += 1
        outputs = self.model(all_word_ids, past_key_values=kv_cache, use_cache=True)

        # Get log probs for subsequent tokens (if multi-token word)
        if len(word_ids) > 1:
            logits = outputs.logits  # [1, N, vocab_size]
            token_log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

            for i in range(len(word_ids) - 1):
                next_token_id = word_ids[i + 1]
                total_score += token_log_probs[0, i, next_token_id].item()

        # Save extended KV cache for the new context (skip if already cached)
        if not already_cached:
            extended_kv = outputs.past_key_values
            new_last_logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            self._evict_lru()
            self.kv_cache[new_context_hash] = (new_context, extended_kv, new_last_logits.clone())
            self._update_access_order(new_context_hash)

        return self.weight * total_score + self.word_insertion_bonus

    @torch.no_grad()
    def score_continuations(
        self,
        contexts: List[str],
        candidate_words: List[List[str]]
    ) -> List[List[float]]:
        """
        Score candidate words given contexts using KV caching.

        Contexts with identical text share the same KV cache, significantly
        reducing computation when many beams have the same history.
        """
        all_scores = []

        # Group by unique context to maximize cache reuse
        unique_contexts = list(set(contexts))

        # Pre-populate cache for all unique contexts
        context_to_cache = {}
        for context in unique_contexts:
            kv_cache, last_logits = self._get_or_create_kv_cache(context)
            context_to_cache[context] = (kv_cache, last_logits)

        # Score each context's candidates
        for context, candidates in zip(contexts, candidate_words):
            kv_cache, last_logits = context_to_cache[context]

            scores = []
            for word in candidates:
                score = self._score_word_and_cache_extension(context, word, kv_cache, last_logits)
                scores.append(score)

            all_scores.append(scores)

        return all_scores

    def to(self, device: torch.device):
        """Move model to specified device and clear cache."""
        self.device = device
        self.model.to(device)
        self.clear_cache()  # Cache tensors are on old device
        return self
