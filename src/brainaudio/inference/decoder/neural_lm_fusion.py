
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
from typing import Any, List, Optional, Tuple, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .batched_beam_decoding_utils import BatchedBeamHyps
    from .lexicon_constraint import LexiconConstraint


class BatchedKVCacheManager:
    """
    Manages batched KV cache storage for efficient cross-beam LM scoring.

    This class enables scoring hundreds of (cache, word) pairs in a single
    forward pass by storing all KV caches in a unified tensor format.

    Storage layout:
        - Caches are stored in a flat array indexed by:
          flat_idx = b * (beam_size * num_homophones) + k * num_homophones + h
        - Each layer's cache: [2, total_slots, num_heads, max_cache_len, head_dim]
          where the "2" dimension is for keys (0) and values (1)
    """

    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        num_homophones: int,
        max_cache_len: int = 256,
        device: torch.device = None,
    ):
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.num_homophones = num_homophones
        self.max_cache_len = max_cache_len
        self.device = device

        self.total_slots = batch_size * beam_size * num_homophones

        # Lazily allocated storage (we don't know model dims until first use)
        self.kv_storage = None  # List of [2, total_slots, num_heads, max_len, head_dim] per layer
        self.last_logprobs = None  # [total_slots, vocab_size]

        # Cache lengths: [batch_size, beam_size, num_homophones]
        self.lengths = torch.zeros(
            (batch_size, beam_size, num_homophones),
            device=device,
            dtype=torch.long
        )

        # Model dimensions (set on first allocation)
        self.num_layers = None
        self.num_heads = None
        self.head_dim = None
        self.vocab_size = None

    def is_allocated(self) -> bool:
        return self.kv_storage is not None

    def flat_index(self, b: int, k: int, h: int) -> int:
        """Convert (batch, beam, homophone) to flat storage index."""
        return b * (self.beam_size * self.num_homophones) + k * self.num_homophones + h

    def flat_indices_tensor(
        self,
        b_indices: torch.Tensor,
        k_indices: torch.Tensor,
        h_indices: torch.Tensor
    ) -> torch.Tensor:
        """Convert batched indices to flat storage indices."""
        return (
            b_indices * (self.beam_size * self.num_homophones)
            + k_indices * self.num_homophones
            + h_indices
        )

    def allocate(self, num_layers: int, num_heads: int, head_dim: int, vocab_size: int):
        """Allocate storage tensors. Called lazily on first cache write."""
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size

        # Allocate KV storage: one tensor per layer
        self.kv_storage = [
            torch.zeros(
                (2, self.total_slots, num_heads, self.max_cache_len, head_dim),
                device=self.device,
                dtype=torch.float16,
            )
            for _ in range(num_layers)
        ]

        # Allocate logprobs storage
        self.last_logprobs = torch.zeros(
            (self.total_slots, vocab_size),
            device=self.device,
            dtype=torch.float32,
        )

    def clear(self):
        """Reset all caches to empty (zero length)."""
        self.lengths.zero_()
        if self.kv_storage is not None:
            for layer_cache in self.kv_storage:
                layer_cache.zero_()
        if self.last_logprobs is not None:
            self.last_logprobs.zero_()

    def get_lengths_for_slots(self, flat_indices: torch.Tensor) -> torch.Tensor:
        """Get cache lengths for given flat indices."""
        b = flat_indices // (self.beam_size * self.num_homophones)
        rem = flat_indices % (self.beam_size * self.num_homophones)
        k = rem // self.num_homophones
        h = rem % self.num_homophones
        return self.lengths[b, k, h]

    def get_caches_for_scoring(
        self,
        flat_indices: torch.Tensor,
    ) -> Tuple[Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[torch.Tensor], torch.Tensor]:
        """
        Extract KV caches for given flat indices, ready for batched forward pass.

        Args:
            flat_indices: [N] flat slot indices to extract

        Returns:
            past_key_values: List of (keys, values) per layer, each [N, heads, max_len, dim]
            cache_attention_mask: [N, max_len] mask for valid cache positions
            actual_lengths: [N] actual sequence lengths
        """
        if not self.is_allocated():
            return None, None, torch.zeros(len(flat_indices), device=self.device, dtype=torch.long)

        actual_lengths = self.get_lengths_for_slots(flat_indices)
        max_len = actual_lengths.max().item()

        if max_len == 0:
            return None, None, actual_lengths

        n_slots = flat_indices.shape[0]

        # Extract caches: [N, heads, max_len, dim]
        past_key_values = []
        for layer_cache in self.kv_storage:
            keys = layer_cache[0, flat_indices, :, :max_len, :]
            values = layer_cache[1, flat_indices, :, :max_len, :]
            past_key_values.append((keys.clone(), values.clone()))

        # Build attention mask: [N, max_len]
        positions = torch.arange(max_len, device=self.device).unsqueeze(0)
        cache_attention_mask = (positions < actual_lengths.unsqueeze(1)).to(torch.long)

        return past_key_values, cache_attention_mask, actual_lengths

    def get_logprobs_for_slots(self, flat_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """Get last logprobs for given flat indices. Returns [N, vocab_size]."""
        if self.last_logprobs is None:
            return None
        return self.last_logprobs[flat_indices]

    def set_cache_single(
        self,
        flat_index: int,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        seq_len: int,
        last_logprobs: Optional[torch.Tensor] = None,
    ):
        """Write a single cache to storage."""
        if past_key_values is None:
            return

        # Lazy allocation
        if not self.is_allocated():
            num_layers = len(past_key_values)
            _, num_heads, _, head_dim = past_key_values[0][0].shape
            vocab_size = last_logprobs.shape[-1] if last_logprobs is not None else 32000
            self.allocate(num_layers, num_heads, head_dim, vocab_size)

        write_len = min(seq_len, self.max_cache_len)

        # Write KV data
        for layer_idx, (key, value) in enumerate(past_key_values):
            # Handle both [1, heads, seq, dim] and [heads, seq, dim] shapes
            if key.dim() == 4:
                key = key.squeeze(0)
                value = value.squeeze(0)
            self.kv_storage[layer_idx][0, flat_index, :, :write_len, :] = key[:, :write_len, :]
            self.kv_storage[layer_idx][1, flat_index, :, :write_len, :] = value[:, :write_len, :]

        # Update length
        b = flat_index // (self.beam_size * self.num_homophones)
        rem = flat_index % (self.beam_size * self.num_homophones)
        k = rem // self.num_homophones
        h = rem % self.num_homophones
        self.lengths[b, k, h] = write_len

        # Write logprobs
        if last_logprobs is not None:
            if last_logprobs.dim() > 1:
                last_logprobs = last_logprobs.squeeze(0)
            self.last_logprobs[flat_index] = last_logprobs

    def set_caches_batched(
        self,
        flat_indices: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        seq_lens: torch.Tensor,
        last_logprobs: Optional[torch.Tensor] = None,
    ):
        """
        Write multiple caches to storage in a batched operation.

        Args:
            flat_indices: [N] flat slot indices
            past_key_values: List of (keys, values) per layer, each [N, heads, seq_len, dim]
            seq_lens: [N] sequence lengths
            last_logprobs: [N, vocab_size] last position logprobs
        """
        if past_key_values is None:
            return

        # Lazy allocation
        if not self.is_allocated():
            num_layers = len(past_key_values)
            _, num_heads, _, head_dim = past_key_values[0][0].shape
            vocab_size = last_logprobs.shape[-1] if last_logprobs is not None else 32000
            self.allocate(num_layers, num_heads, head_dim, vocab_size)

        max_len = min(seq_lens.max().item(), self.max_cache_len)

        # Write KV data for all slots
        for layer_idx, (keys, values) in enumerate(past_key_values):
            self.kv_storage[layer_idx][0, flat_indices, :, :max_len, :] = keys[:, :, :max_len, :]
            self.kv_storage[layer_idx][1, flat_indices, :, :max_len, :] = values[:, :, :max_len, :]

        # Update lengths
        b = flat_indices // (self.beam_size * self.num_homophones)
        rem = flat_indices % (self.beam_size * self.num_homophones)
        k = rem // self.num_homophones
        h = rem % self.num_homophones
        self.lengths[b, k, h] = torch.clamp(seq_lens, max=self.max_cache_len)

        # Write logprobs
        if last_logprobs is not None:
            self.last_logprobs[flat_indices] = last_logprobs

    def reorder_for_beam_selection(self, next_indices: torch.Tensor):
        """
        Reorder caches when beams are selected during beam search.

        Args:
            next_indices: [batch_size, beam_size] where next_indices[b, k] is the
                         source beam index for the new beam k in batch b
        """
        if not self.is_allocated():
            return

        # Build source flat indices: for each (b, k, h), source is (b, next_indices[b,k], h)
        b_idx = torch.arange(self.batch_size, device=self.device)[:, None, None]
        h_idx = torch.arange(self.num_homophones, device=self.device)[None, None, :]
        src_k = next_indices[:, :, None].expand(-1, -1, self.num_homophones)

        src_flat = (
            b_idx * (self.beam_size * self.num_homophones)
            + src_k * self.num_homophones
            + h_idx
        ).reshape(-1)

        # Reorder KV storage
        for layer_idx in range(self.num_layers):
            old_cache = self.kv_storage[layer_idx].clone()
            self.kv_storage[layer_idx] = old_cache[:, src_flat, :, :, :]

        # Reorder logprobs
        old_logprobs = self.last_logprobs.clone()
        self.last_logprobs = old_logprobs[src_flat]

        # Reorder lengths
        old_lengths = self.lengths.clone()
        src_k_expanded = next_indices[:, :, None].expand(-1, -1, self.num_homophones)
        self.lengths = torch.gather(old_lengths, dim=1, index=src_k_expanded)


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

        # Batched KV cache manager (initialized per decoding session)
        self.cache_manager: Optional[BatchedKVCacheManager] = None

    def init_cache_manager(
        self,
        batch_size: int,
        beam_size: int,
        num_homophones: int,
        max_cache_len: int = 256,
    ):
        """
        Initialize the batched KV cache manager for a decoding session.

        Call this at the start of each decoding batch to set up cache storage.

        Args:
            batch_size: Number of utterances in the batch
            beam_size: Beam width for beam search
            num_homophones: Number of homophone hypotheses to track per beam
            max_cache_len: Maximum cache sequence length (default 256)
        """
        self.cache_manager = BatchedKVCacheManager(
            batch_size=batch_size,
            beam_size=beam_size,
            num_homophones=num_homophones,
            max_cache_len=max_cache_len,
            device=self.device,
        )

    def reset_cache_manager(self):
        """Clear all cached states for a new decoding session."""
        if self.cache_manager is not None:
            self.cache_manager.clear()

    def reset_call_count(self):
        """Reset the LLM call counter to 0."""
        self.llm_call_count = 0

    def get_call_count(self) -> int:
        """Get the current LLM call count."""
        return self.llm_call_count

    @torch.no_grad()
    def score_batch_with_cache(
        self,
        flat_indices: torch.Tensor,
        words: List[str],
        context_texts: List[str],
        batch_size: int = 512,
        update_caches: bool = False,
    ) -> torch.Tensor:
        """
        Score multiple (cache_slot, word) pairs in batched forward passes.

        This is the key method for fast cross-beam scoring. It:
        1. Extracts KV caches for all specified slots from the cache manager
        2. Tokenizes all words
        3. Runs batched forward passes (up to batch_size at a time)
        4. Returns scores (does NOT update caches by default)

        Args:
            flat_indices: [N] tensor of flat cache slot indices to READ from
            words: [N] list of words to score (one per slot)
            context_texts: [N] list of context texts (for space handling)
            batch_size: Maximum items per forward pass (default 512)
            update_caches: If True, update caches after scoring (default False)

        Returns:
            scores: [N] tensor of LM scores (weighted log-probs + bonus)
        """
        if self.cache_manager is None:
            raise RuntimeError("Cache manager not initialized. Call init_cache_manager first.")

        n_items = len(words)
        if n_items == 0:
            return torch.tensor([], device=self.device)

        all_scores = []

        # Process in batches
        for batch_start in range(0, n_items, batch_size):
            batch_end = min(batch_start + batch_size, n_items)
            batch_indices = flat_indices[batch_start:batch_end]
            batch_words = words[batch_start:batch_end]
            batch_contexts = context_texts[batch_start:batch_end]

            scores = self._score_batch_read_only(
                batch_indices, batch_words, batch_contexts
            )
            all_scores.append(scores)

        return torch.cat(all_scores)

    @torch.no_grad()
    def update_caches_for_selected(
        self,
        src_flat_indices: torch.Tensor,
        dst_flat_indices: torch.Tensor,
        words: List[str],
        context_texts: List[str],
        batch_size: int = 512,
    ):
        """
        Update caches for selected (slot, word) pairs after top-K selection.

        This runs forward passes to compute new caches for the selected items
        and writes them to the destination slots.

        Args:
            src_flat_indices: [N] source slot indices (where to read cache from)
            dst_flat_indices: [N] destination slot indices (where to write new cache)
            words: [N] list of selected words
            context_texts: [N] list of context texts
            batch_size: Maximum items per forward pass
        """
        if self.cache_manager is None:
            return

        n_items = len(words)
        if n_items == 0:
            return

        for batch_start in range(0, n_items, batch_size):
            batch_end = min(batch_start + batch_size, n_items)
            src_indices = src_flat_indices[batch_start:batch_end]
            dst_indices = dst_flat_indices[batch_start:batch_end]
            batch_words = words[batch_start:batch_end]
            batch_contexts = context_texts[batch_start:batch_end]

            self._update_caches_batch(
                src_indices, dst_indices, batch_words, batch_contexts
            )

    @torch.no_grad()
    def _score_batch_read_only(
        self,
        flat_indices: torch.Tensor,
        words: List[str],
        context_texts: List[str],
    ) -> torch.Tensor:
        """
        Score a batch of (slot, word) pairs WITHOUT updating caches.

        This is read-only - it reads from cache manager but doesn't write back.
        Used during the scoring phase before top-K selection.
        """
        n_items = len(words)
        device = self.device

        # Get cached states from manager (read-only)
        past_kv, cache_attn_mask, cache_lengths = self.cache_manager.get_caches_for_scoring(flat_indices)
        cached_logprobs = self.cache_manager.get_logprobs_for_slots(flat_indices)

        # Tokenize all words (with proper spacing)
        word_token_lists = []
        for word, context in zip(words, context_texts):
            if context and not context.endswith(" ") and not word.startswith(" "):
                full_word = " " + word
            else:
                full_word = word
            tokens = self.tokenizer.encode(full_word, add_special_tokens=False)
            word_token_lists.append(tokens if tokens else [self.tokenizer.eos_token_id])

        # Handle case where we have no cached state (bootstrapping)
        if past_kv is None or cache_lengths.max().item() == 0:
            # No cache - score with full context (read-only)
            return self._bootstrap_score_only(words, context_texts, word_token_lists)

        # Pad word tokens to same length
        max_word_len = max(len(t) for t in word_token_lists)
        padded_word_ids = []
        word_attention_masks = []
        for tokens in word_token_lists:
            pad_len = max_word_len - len(tokens)
            padded_word_ids.append(tokens + [self.tokenizer.pad_token_id] * pad_len)
            word_attention_masks.append([1] * len(tokens) + [0] * pad_len)

        word_input_ids = torch.tensor(padded_word_ids, device=device)
        word_attn_mask = torch.tensor(word_attention_masks, device=device)

        # Build position IDs for new tokens
        position_ids = cache_lengths.unsqueeze(1) + torch.arange(max_word_len, device=device).unsqueeze(0)

        # Build full attention mask
        full_attn_mask = torch.cat([cache_attn_mask, word_attn_mask], dim=1)

        # Forward pass (don't need use_cache=True since we're not using the output cache)
        self.llm_call_count += 1
        outputs = self.model(
            input_ids=word_input_ids,
            attention_mask=full_attn_mask,
            past_key_values=past_kv,
            position_ids=position_ids,
            use_cache=False,  # Don't compute output cache for scoring only
        )

        # Compute scores
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        scores = torch.zeros(n_items, device=device)
        for i, tokens in enumerate(word_token_lists):
            word_len = len(tokens)

            # First token: use cached logprobs (O(1) lookup)
            if cached_logprobs is not None and cache_lengths[i] > 0:
                first_token_score = cached_logprobs[i, tokens[0]].item()
            else:
                first_token_score = 0.0

            # Remaining tokens: from forward pass output
            remaining_score = 0.0
            for j in range(word_len - 1):
                next_token_id = tokens[j + 1]
                remaining_score += log_probs[i, j, next_token_id].item()

            total_score = first_token_score + remaining_score
            scores[i] = total_score * self.weight + self.word_insertion_bonus

        del outputs, logits, log_probs
        torch.cuda.empty_cache()

        return scores

    @torch.no_grad()
    def _bootstrap_score_only(
        self,
        words: List[str],
        context_texts: List[str],
        word_token_lists: List[List[int]],
    ) -> torch.Tensor:
        """Score words without cache (bootstrap case), read-only."""
        n_items = len(words)
        device = self.device

        # Build full texts: context + word
        full_texts = []
        word_start_indices = []
        for word, context in zip(words, context_texts):
            if context and not context.endswith(" ") and not word.startswith(" "):
                full_text = context + " " + word
            else:
                full_text = (context or "") + word
            full_texts.append(full_text)

            if context:
                context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
                word_start_indices.append(len(context_tokens) - 1)
            else:
                word_start_indices.append(0)

        # Tokenize with padding
        inputs = self.tokenizer(
            full_texts, return_tensors="pt", padding=True, add_special_tokens=True
        ).to(device)

        # Forward pass
        self.llm_call_count += 1
        outputs = self.model(inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False)

        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        scores = torch.zeros(n_items, device=device)
        for i, (start_idx, attn) in enumerate(zip(word_start_indices, inputs.attention_mask)):
            seq_len = attn.sum().item()
            word_tokens = inputs.input_ids[i, :int(seq_len)].tolist()

            total_score = 0.0
            for j in range(start_idx, int(seq_len) - 1):
                next_token_id = word_tokens[j + 1]
                total_score += log_probs[i, j, next_token_id].item()

            scores[i] = total_score * self.weight + self.word_insertion_bonus

        del outputs, logits, log_probs, inputs
        torch.cuda.empty_cache()

        return scores

    @torch.no_grad()
    def _update_caches_batch(
        self,
        src_flat_indices: torch.Tensor,
        dst_flat_indices: torch.Tensor,
        words: List[str],
        context_texts: List[str],
    ):
        """
        Update caches for selected items by running forward passes.

        Reads cache from src_flat_indices, computes new cache with word,
        writes to dst_flat_indices.
        """
        n_items = len(words)
        device = self.device

        # Get source caches
        past_kv, cache_attn_mask, cache_lengths = self.cache_manager.get_caches_for_scoring(src_flat_indices)

        # Tokenize words
        word_token_lists = []
        for word, context in zip(words, context_texts):
            if context and not context.endswith(" ") and not word.startswith(" "):
                full_word = " " + word
            else:
                full_word = word
            tokens = self.tokenizer.encode(full_word, add_special_tokens=False)
            word_token_lists.append(tokens if tokens else [self.tokenizer.eos_token_id])

        # Handle bootstrap case
        if past_kv is None or cache_lengths.max().item() == 0:
            self._bootstrap_and_update_caches(dst_flat_indices, words, context_texts)
            return

        # Pad word tokens
        max_word_len = max(len(t) for t in word_token_lists)
        padded_word_ids = []
        word_attention_masks = []
        for tokens in word_token_lists:
            pad_len = max_word_len - len(tokens)
            padded_word_ids.append(tokens + [self.tokenizer.pad_token_id] * pad_len)
            word_attention_masks.append([1] * len(tokens) + [0] * pad_len)

        word_input_ids = torch.tensor(padded_word_ids, device=device)
        word_attn_mask = torch.tensor(word_attention_masks, device=device)

        # Position IDs and attention mask
        position_ids = cache_lengths.unsqueeze(1) + torch.arange(max_word_len, device=device).unsqueeze(0)
        full_attn_mask = torch.cat([cache_attn_mask, word_attn_mask], dim=1)

        # Forward pass WITH cache output
        self.llm_call_count += 1
        outputs = self.model(
            input_ids=word_input_ids,
            attention_mask=full_attn_mask,
            past_key_values=past_kv,
            position_ids=position_ids,
            use_cache=True,
        )

        # Compute new cache lengths and logprobs
        new_cache_lengths = torch.zeros(n_items, device=device, dtype=torch.long)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        new_logprobs = torch.zeros(n_items, log_probs.shape[-1], device=device)

        for i, tokens in enumerate(word_token_lists):
            word_len = len(tokens)
            new_cache_lengths[i] = cache_lengths[i] + word_len
            new_logprobs[i] = log_probs[i, word_len - 1]

        # Write to destination slots
        self.cache_manager.set_caches_batched(
            dst_flat_indices,
            outputs.past_key_values,
            new_cache_lengths,
            new_logprobs,
        )

        del outputs, log_probs
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _bootstrap_and_update_caches(
        self,
        dst_flat_indices: torch.Tensor,
        words: List[str],
        context_texts: List[str],
    ):
        """Bootstrap and update caches when no prior cache exists."""
        n_items = len(words)
        device = self.device

        # Build full texts
        full_texts = []
        word_start_indices = []
        for word, context in zip(words, context_texts):
            if context and not context.endswith(" ") and not word.startswith(" "):
                full_text = context + " " + word
            else:
                full_text = (context or "") + word
            full_texts.append(full_text)

            if context:
                context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
                word_start_indices.append(len(context_tokens) - 1)
            else:
                word_start_indices.append(0)

        # Tokenize
        inputs = self.tokenizer(
            full_texts, return_tensors="pt", padding=True, add_special_tokens=True
        ).to(device)

        # Forward pass
        self.llm_call_count += 1
        outputs = self.model(inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=True)

        log_probs = F.log_softmax(outputs.logits, dim=-1)

        new_cache_lengths = torch.zeros(n_items, device=device, dtype=torch.long)
        new_logprobs = torch.zeros(n_items, log_probs.shape[-1], device=device)

        for i, attn in enumerate(inputs.attention_mask):
            seq_len = int(attn.sum().item())
            new_cache_lengths[i] = seq_len
            new_logprobs[i] = log_probs[i, seq_len - 1]

        # Write caches
        self.cache_manager.set_caches_batched(
            dst_flat_indices,
            outputs.past_key_values,
            new_cache_lengths,
            new_logprobs,
        )

        del outputs, log_probs, inputs
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _bootstrap_and_score_batch(
        self,
        flat_indices: torch.Tensor,
        words: List[str],
        context_texts: List[str],
        word_token_lists: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bootstrap caches and score words when no prior cache exists.

        This handles the first word of each beam, running a full forward pass
        on context + word and initializing the cache.
        """
        n_items = len(words)
        device = self.device

        # Build full texts: context + word
        full_texts = []
        word_start_indices = []
        for word, context in zip(words, context_texts):
            if context and not context.endswith(" ") and not word.startswith(" "):
                full_text = context + " " + word
            else:
                full_text = (context or "") + word

            full_texts.append(full_text)

            # Find where word starts in token space
            if context:
                context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
                word_start_indices.append(len(context_tokens) - 1)
            else:
                word_start_indices.append(0)

        # Tokenize all full texts with padding
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Forward pass
        self.llm_call_count += 1
        outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=True)

        # Get logprobs
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute scores and extract cache info
        scores = torch.zeros(n_items, device=device)
        new_cache_lengths = torch.zeros(n_items, device=device, dtype=torch.long)
        new_logprobs = torch.zeros(n_items, log_probs.shape[-1], device=device)

        for i, (start_idx, attn) in enumerate(zip(word_start_indices, attention_mask)):
            seq_len = attn.sum().item()
            word_tokens = input_ids[i, :seq_len].tolist()

            # Score the word tokens
            total_score = 0.0
            for j in range(start_idx, seq_len - 1):
                next_token_id = word_tokens[j + 1]
                total_score += log_probs[i, j, next_token_id].item()

            scores[i] = total_score * self.weight + self.word_insertion_bonus
            new_cache_lengths[i] = seq_len
            new_logprobs[i] = log_probs[i, seq_len - 1]

        # Update caches in manager
        self.cache_manager.set_caches_batched(
            flat_indices,
            outputs.past_key_values,
            new_cache_lengths,
            new_logprobs,
        )

        # Cleanup
        del outputs, logits, log_probs, inputs
        torch.cuda.empty_cache()

        return scores, new_cache_lengths

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
            outputs = self.model(input_ids, attention_mask=attention_mask)
            

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

    @torch.no_grad()
    def score_word_with_cache(
        self,
        kv_cache: Optional[Any],
        last_logprobs: Optional[torch.Tensor],
        context_text: str,
        word: str,
    ) -> Tuple[float, Any, torch.Tensor]:
        """
        Score a word using cached KV state for O(1) first-token lookup.

        This method provides massive speedup over score_continuations by:
        1. First token: O(1) lookup in last_logprobs (FREE - no forward pass!)
        2. Remaining tokens: Incremental forward pass using KV cache (O(k) not O(n))

        Args:
            kv_cache: KV cache from previous words (DynamicCache), or None for first word
            last_logprobs: Log probs [vocab_size] from previous forward pass, or None
            context_text: Text context (only used if cache is None for bootstrapping)
            word: The word to score (will be prepended with space if needed)

        Returns:
            Tuple of (score, new_kv_cache, new_last_logprobs)
        """
        # Handle space prepending for word boundaries
        if context_text and not context_text.endswith(" ") and not word.startswith(" "):
            full_word = " " + word
        else:
            full_word = word

        # Tokenize the word only (not the full context!)
        word_tokens = self.tokenizer.encode(full_word, add_special_tokens=False)

        if not word_tokens:
            # Empty word - return unchanged state
            return 0.0, kv_cache, last_logprobs

        # Case 1: No cache (first word) - need to bootstrap with full forward pass
        if kv_cache is None or last_logprobs is None:
            return self._bootstrap_cache_and_score(context_text, word)

        # Case 2 & 3: Have cache - use it for fast scoring
        total_score = 0.0

        # First token is FREE - direct lookup in cached logprobs
        first_token_id = word_tokens[0]
        total_score += last_logprobs[first_token_id].item()

        # Forward pass to get updated cache and new logprobs
        # We process ALL tokens (including first) to update the KV cache
        input_ids = torch.tensor([word_tokens], device=self.device)
        self.llm_call_count += 1
        outputs = self.model(input_ids, past_key_values=kv_cache, use_cache=True)
        new_cache = outputs.past_key_values

        # Get logprobs for all positions
        all_logprobs = F.log_softmax(outputs.logits[0], dim=-1)

        # Score remaining tokens (tokens 1 to N-1)
        # Position i in output predicts token i+1, but we already have token i in input
        # So position i gives us P(token[i+1] | context + token[0:i+1])
        # But wait - we need P(token[i] | context + token[0:i]) for scoring
        # The model output at position i is P(next | prefix including token[i])
        # So for scoring token[i+1], we look at position i
        for i in range(len(word_tokens) - 1):
            next_token_id = word_tokens[i + 1]
            # Position i predicts position i+1
            total_score += all_logprobs[i, next_token_id].item()

        # Last position gives us logprobs for the NEXT word
        new_last_logprobs = all_logprobs[-1]

        return total_score * self.weight + self.word_insertion_bonus, new_cache, new_last_logprobs

    @torch.no_grad()
    def _bootstrap_cache_and_score(
        self,
        context_text: str,
        word: str,
    ) -> Tuple[float, Any, torch.Tensor]:
        """
        Bootstrap the KV cache with context + word when no cache exists.

        Called for the first word or when cache is missing.
        Returns the score for the word along with initialized cache and logprobs.
        """
        # Build full text with proper spacing
        if context_text and not context_text.endswith(" ") and not word.startswith(" "):
            full_text = context_text + " " + word
        else:
            full_text = context_text + word

        # Tokenize full text
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)

        if not full_tokens:
            return 0.0, None, None

        # Figure out where the word starts
        if context_text:
            context_tokens = self.tokenizer.encode(context_text, add_special_tokens=True)
            word_start_idx = len(context_tokens) - 1  # -1 because we score from previous position
        else:
            word_start_idx = 0

        # Forward pass on full sequence
        input_ids = torch.tensor([full_tokens], device=self.device)
        self.llm_call_count += 1
        outputs = self.model(input_ids, use_cache=True)
        new_cache = outputs.past_key_values

        # Get logprobs
        all_logprobs = F.log_softmax(outputs.logits[0], dim=-1)

        # Score the word tokens
        total_score = 0.0
        seq_len = len(full_tokens)
        for i in range(word_start_idx, seq_len - 1):
            next_token_id = full_tokens[i + 1]
            total_score += all_logprobs[i, next_token_id].item()

        # Last position gives logprobs for next word
        new_last_logprobs = all_logprobs[-1]

        return total_score * self.weight + self.word_insertion_bonus, new_cache, new_last_logprobs

    @torch.no_grad()
    def score_words_with_cache(
        self,
        kv_cache: Optional[Any],
        last_logprobs: Optional[torch.Tensor],
        context_text: str,
        words: List[str],
        batch_size: int = 512,
    ) -> Tuple[List[float], List[Any], List[torch.Tensor]]:
        """
        Score multiple words using cached KV state with batched processing.

        This method batches the forward passes for efficiency while still
        leveraging the KV cache. Words are processed in batches of `batch_size`.

        Args:
            kv_cache: KV cache from previous words (DynamicCache), or None for first word
            last_logprobs: Log probs [vocab_size] from previous forward pass, or None
            context_text: Text context (only used if cache is None for bootstrapping)
            words: List of words to score
            batch_size: Maximum number of words to process in a single forward pass

        Returns:
            Tuple of (scores, new_kv_caches, new_last_logprobs) where each is a list
            with one entry per word
        """
        if not words:
            return [], [], []

        # Case 1: No cache - need to bootstrap
        if kv_cache is None or last_logprobs is None:
            return self._bootstrap_cache_and_score_batch(context_text, words, batch_size)

        # Case 2: Have cache - use batched scoring
        all_scores = []
        all_caches = []
        all_logprobs = []

        # Process words in batches
        for batch_start in range(0, len(words), batch_size):
            batch_end = min(batch_start + batch_size, len(words))
            batch_words = words[batch_start:batch_end]

            scores, caches, logprobs = self._score_batch_with_cache(
                kv_cache, last_logprobs, context_text, batch_words
            )
            all_scores.extend(scores)
            all_caches.extend(caches)
            all_logprobs.extend(logprobs)

        return all_scores, all_caches, all_logprobs

    @torch.no_grad()
    def _score_batch_with_cache(
        self,
        kv_cache: Any,
        last_logprobs: torch.Tensor,
        context_text: str,
        words: List[str],
    ) -> Tuple[List[float], List[Any], List[torch.Tensor]]:
        """
        Score a batch of words against the same KV cache state.

        Expands the cache to batch size, runs a single forward pass, then
        extracts per-word results.
        """
        # Tokenize all words (with space prepending for word boundaries)
        word_token_lists = []
        for word in words:
            if context_text and not context_text.endswith(" ") and not word.startswith(" "):
                full_word = " " + word
            else:
                full_word = word
            tokens = self.tokenizer.encode(full_word, add_special_tokens=False)
            word_token_lists.append(tokens if tokens else [self.tokenizer.eos_token_id])

        num_words = len(words)
        max_len = max(len(t) for t in word_token_lists)

        # Pad token lists to max length
        padded_input_ids = []
        attention_masks = []
        for tokens in word_token_lists:
            pad_len = max_len - len(tokens)
            padded_input_ids.append(tokens + [self.tokenizer.pad_token_id] * pad_len)
            attention_masks.append([1] * len(tokens) + [0] * pad_len)

        input_ids = torch.tensor(padded_input_ids, device=self.device)
        attention_mask = torch.tensor(attention_masks, device=self.device)

        # Expand KV cache to batch size
        expanded_cache = self._expand_cache(kv_cache, num_words)

        # Build position_ids for the new tokens (continuing from cache length)
        cache_seq_len = kv_cache[0][0].shape[2] if kv_cache else 0
        position_ids = torch.arange(
            cache_seq_len, cache_seq_len + max_len, device=self.device
        ).unsqueeze(0).expand(num_words, -1)

        # Build cache attention mask (all 1s for cached positions + new attention mask)
        if cache_seq_len > 0:
            cache_attn = torch.ones(num_words, cache_seq_len, device=self.device, dtype=attention_mask.dtype)
            full_attention_mask = torch.cat([cache_attn, attention_mask], dim=1)
        else:
            full_attention_mask = attention_mask

        # Forward pass
        self.llm_call_count += 1
        outputs = self.model(
            input_ids,
            attention_mask=full_attention_mask,
            past_key_values=expanded_cache,
            position_ids=position_ids,
            use_cache=True,
        )
        new_cache_batched = outputs.past_key_values
        all_logits = outputs.logits  # [num_words, max_len, vocab_size]
        all_logprobs_out = F.log_softmax(all_logits, dim=-1)

        # Compute scores and extract per-word results
        scores = []
        new_caches = []
        new_last_logprobs = []

        for i, tokens in enumerate(word_token_lists):
            seq_len = len(tokens)

            # First token score from cached logprobs (O(1) lookup)
            first_token_score = last_logprobs[tokens[0]].item()

            # Remaining token scores from forward pass output
            remaining_score = 0.0
            for j in range(seq_len - 1):
                next_token_id = tokens[j + 1]
                remaining_score += all_logprobs_out[i, j, next_token_id].item()

            total_score = first_token_score + remaining_score
            scores.append(total_score * self.weight + self.word_insertion_bonus)

            # Extract per-word cache (slice to actual sequence length)
            word_cache = self._slice_cache(new_cache_batched, i, cache_seq_len + seq_len)
            new_caches.append(word_cache)

            # Last logprobs for this word (at position seq_len-1)
            new_last_logprobs.append(all_logprobs_out[i, seq_len - 1].clone())

        # Free memory
        del outputs, all_logits, all_logprobs_out, expanded_cache
        torch.cuda.empty_cache()

        return scores, new_caches, new_last_logprobs

    def _expand_cache(self, kv_cache: Any, batch_size: int) -> Any:
        """Expand a batch-1 KV cache to the given batch size."""
        if kv_cache is None:
            return None
        expanded = tuple(
            (k.expand(batch_size, -1, -1, -1), v.expand(batch_size, -1, -1, -1))
            for k, v in kv_cache
        )
        return expanded

    def _slice_cache(self, batched_cache: Any, batch_idx: int, seq_len: int) -> Any:
        """Extract a single-sequence cache from a batched cache."""
        if batched_cache is None:
            return None
        sliced = tuple(
            (k[batch_idx:batch_idx+1, :, :seq_len, :].clone(),
             v[batch_idx:batch_idx+1, :, :seq_len, :].clone())
            for k, v in batched_cache
        )
        return sliced

    @torch.no_grad()
    def _bootstrap_cache_and_score_batch(
        self,
        context_text: str,
        words: List[str],
        batch_size: int = 512,
    ) -> Tuple[List[float], List[Any], List[torch.Tensor]]:
        """
        Bootstrap KV cache and score multiple words when no cache exists.

        Processes words in batches for efficiency.
        """
        all_scores = []
        all_caches = []
        all_logprobs = []

        for batch_start in range(0, len(words), batch_size):
            batch_end = min(batch_start + batch_size, len(words))
            batch_words = words[batch_start:batch_end]

            # Build full texts for each word
            full_texts = []
            word_start_indices = []
            for word in batch_words:
                if context_text and not context_text.endswith(" ") and not word.startswith(" "):
                    full_text = context_text + " " + word
                else:
                    full_text = context_text + word

                full_texts.append(full_text)

                # Compute where the word starts in token space
                if context_text:
                    context_tokens = self.tokenizer.encode(context_text, add_special_tokens=True)
                    word_start_indices.append(len(context_tokens) - 1)
                else:
                    word_start_indices.append(0)

            # Tokenize all texts with padding
            inputs = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True
            ).to(self.device)

            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            # Forward pass
            self.llm_call_count += 1
            outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=True)
            batched_cache = outputs.past_key_values
            all_logits = outputs.logits
            all_logprobs_out = F.log_softmax(all_logits, dim=-1)

            # Extract scores and caches for each word
            for i, (word, start_idx) in enumerate(zip(batch_words, word_start_indices)):
                seq_len = attention_mask[i].sum().item()
                word_tokens = input_ids[i, :int(seq_len)].tolist()

                # Score the word tokens
                total_score = 0.0
                for j in range(start_idx, int(seq_len) - 1):
                    next_token_id = word_tokens[j + 1]
                    total_score += all_logprobs_out[i, j, next_token_id].item()

                all_scores.append(total_score * self.weight + self.word_insertion_bonus)

                # Extract cache for this sequence
                word_cache = self._slice_cache(batched_cache, i, int(seq_len))
                all_caches.append(word_cache)

                # Last logprobs
                all_logprobs.append(all_logprobs_out[i, int(seq_len) - 1].clone())

            # Free memory
            del outputs, all_logits, all_logprobs_out, batched_cache
            del inputs, input_ids, attention_mask
            torch.cuda.empty_cache()

        return all_scores, all_caches, all_logprobs

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
        # and the number of tuples = num_homophone_beams
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

    num_homophone_beams = beam_hyps.num_homophone_beams

    # Check if we can use the new batched cache manager
    use_cache_manager = (
        hasattr(lm_fusion, 'cache_manager') and
        lm_fusion.cache_manager is not None
    )

    if use_cache_manager:
        # === FAST PATH: Batched scoring across all beams using cache manager ===
        _apply_lm_fusion_with_cache_manager(
            lm_fusion, beam_hyps, to_score, num_homophone_beams, homophone_prune_threshold
        )
    else:
        # === FALLBACK PATH: Original per-beam scoring ===
        _apply_lm_fusion_legacy(
            lm_fusion, beam_hyps, to_score, num_homophone_beams, homophone_prune_threshold
        )


def _apply_lm_fusion_with_cache_manager(
    lm_fusion: 'HuggingFaceLMFusion',
    beam_hyps: 'BatchedBeamHyps',
    to_score: list,
    num_homophone_beams: int,
    homophone_prune_threshold: float | None,
) -> None:
    """
    Fast path: Score all beams using batched cache manager.

    Collects ALL (slot, word) pairs across all beams and scores them
    in batched forward passes for maximum efficiency.

    Two phases:
    1. Score all candidates (read-only from cache)
    2. After selection, update caches only for selected items
    """
    cache_mgr = lm_fusion.cache_manager
    device = lm_fusion.device

    # --- PHASE 2a: Collect ALL scoring requests across all beams ---
    all_flat_indices = []
    all_words = []
    all_contexts = []
    all_metadata = []  # (to_score_idx, tuple_idx, prev_lm_score, word, prev_text)

    for to_score_idx, (b, k, context_text_tuples, candidate_words) in enumerate(to_score):
        for tuple_idx, (prev_lm_score, prev_text) in enumerate(context_text_tuples):
            flat_idx = cache_mgr.flat_index(b, k, tuple_idx)
            for word in candidate_words:
                all_flat_indices.append(flat_idx)
                all_words.append(word)
                all_contexts.append(prev_text)
                all_metadata.append((to_score_idx, tuple_idx, prev_lm_score, word, prev_text))

    if not all_flat_indices:
        return

    # --- PHASE 2b: Batch score everything (READ-ONLY, no cache updates) ---
    flat_indices_tensor = torch.tensor(all_flat_indices, device=device, dtype=torch.long)
    scores_tensor = lm_fusion.score_batch_with_cache(
        flat_indices_tensor, all_words, all_contexts, batch_size=512
    )

    # --- PHASE 3: Organize results by beam and select top candidates ---
    scores_by_beam = {}  # to_score_idx -> list of (total_score, text, tuple_idx, word, prev_text)
    for i, (to_score_idx, tuple_idx, prev_lm_score, word, prev_text) in enumerate(all_metadata):
        word_score = scores_tensor[i].item()
        total_score = prev_lm_score + word_score

        b, k, context_text_tuples, _ = to_score[to_score_idx]
        new_text = f"{prev_text} {word}".strip()

        if to_score_idx not in scores_by_beam:
            scores_by_beam[to_score_idx] = []
        scores_by_beam[to_score_idx].append((total_score, new_text, tuple_idx, word, prev_text))

    # Collect updates and selected items for cache update
    update_batch_indices = []
    update_beam_indices = []
    update_scores = []
    update_context_texts = []
    update_hashes = []

    # For cache updates: src_idx -> dst_idx mapping
    all_cache_updates = []  # (b, k, new_h, src_tuple_idx, word, prev_text)

    for to_score_idx, (b, k, context_text_tuples, candidate_words) in enumerate(to_score):
        base_score = beam_hyps.scores[b, k].item()
        candidates = scores_by_beam.get(to_score_idx, [])

        # Deduplicate by lowercase text
        lowercase_to_best = {}
        for total_score, text, tuple_idx, word, prev_text in candidates:
            text_lower = text.lower()
            if text_lower not in lowercase_to_best or total_score > lowercase_to_best[text_lower][0]:
                lowercase_to_best[text_lower] = (total_score, text, tuple_idx, word, prev_text)
        candidates = list(lowercase_to_best.values())

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Prune
        if homophone_prune_threshold is not None and candidates:
            best = candidates[0][0]
            candidates = [c for c in candidates if best - c[0] <= homophone_prune_threshold]

        # Keep top K
        top_k = candidates[:num_homophone_beams]

        if not top_k:
            continue

        # Build new tuples (score, text)
        new_tuples = [(score, text) for score, text, _, _, _ in top_k]

        # Track which items were selected for cache update
        for new_h, (_, _, src_tuple_idx, word, prev_text) in enumerate(top_k):
            all_cache_updates.append((b, k, new_h, src_tuple_idx, word, prev_text))

        # Compute updated beam score
        old_best_lm_score = context_text_tuples[0][0] if context_text_tuples else 0.0
        new_best_lm_score = new_tuples[0][0]
        acoustic_score = base_score - old_best_lm_score
        new_score = acoustic_score + new_best_lm_score

        update_batch_indices.append(b)
        update_beam_indices.append(k)
        update_scores.append(new_score)
        update_context_texts.append((b, k, new_tuples))
        update_hashes.append((b, k, hash(new_tuples[0][1])))

    # --- PHASE 4: Update caches for selected items only ---
    if all_cache_updates:
        src_indices = []
        dst_indices = []
        words = []
        contexts = []

        for b, k, new_h, src_tuple_idx, word, prev_text in all_cache_updates:
            src_indices.append(cache_mgr.flat_index(b, k, src_tuple_idx))
            dst_indices.append(cache_mgr.flat_index(b, k, new_h))
            words.append(word)
            contexts.append(prev_text)

        src_tensor = torch.tensor(src_indices, device=device, dtype=torch.long)
        dst_tensor = torch.tensor(dst_indices, device=device, dtype=torch.long)
        lm_fusion.update_caches_for_selected(src_tensor, dst_tensor, words, contexts, batch_size=512)

    # Apply updates to beam_hyps
    if update_batch_indices:
        beam_hyps.scores[update_batch_indices, update_beam_indices] = torch.tensor(
            update_scores, device=beam_hyps.scores.device, dtype=beam_hyps.scores.dtype
        )

    for b, k, new_tuples in update_context_texts:
        beam_hyps.context_texts[b][k] = new_tuples

    for b, k, hash_value in update_hashes:
        beam_hyps.context_texts_hash[b][k] = hash_value


def _apply_lm_fusion_legacy(
    lm_fusion: NeuralLanguageModelFusion,
    beam_hyps: 'BatchedBeamHyps',
    to_score: list,
    num_homophone_beams: int,
    homophone_prune_threshold: float | None,
) -> None:
    """
    Legacy path: Per-beam scoring without cache manager.

    Falls back to score_continuations for simple batched scoring.
    """
    # Collect all (context, candidates) across all beams for batched scoring
    flat_contexts = []
    flat_candidates = []
    flat_mapping = []  # (to_score_idx, tuple_idx, prev_lm_score)

    for to_score_idx, (b, k, context_text_tuples, candidate_words) in enumerate(to_score):
        for tuple_idx, (prev_lm_score, prev_text) in enumerate(context_text_tuples):
            flat_contexts.append(prev_text)
            flat_candidates.append(candidate_words)
            flat_mapping.append((to_score_idx, tuple_idx, prev_lm_score))

    # Single batched LM call
    all_scores = lm_fusion.score_continuations(flat_contexts, flat_candidates)

    # Organize results by beam
    scores_by_beam = {}
    for flat_idx, (to_score_idx, tuple_idx, prev_lm_score) in enumerate(flat_mapping):
        b, k, context_text_tuples, candidate_words = to_score[to_score_idx]
        prev_text = context_text_tuples[tuple_idx][1]
        word_scores = all_scores[flat_idx]

        if to_score_idx not in scores_by_beam:
            scores_by_beam[to_score_idx] = []

        for word, word_score in zip(candidate_words, word_scores):
            total_score = prev_lm_score + word_score
            new_text = f"{prev_text} {word}".strip()
            scores_by_beam[to_score_idx].append((total_score, new_text))

    # Collect updates
    update_batch_indices = []
    update_beam_indices = []
    update_scores = []
    update_context_texts = []
    update_hashes = []

    for to_score_idx, (b, k, context_text_tuples, candidate_words) in enumerate(to_score):
        base_score = beam_hyps.scores[b, k].item()
        candidates = scores_by_beam.get(to_score_idx, [])

        # Deduplicate
        lowercase_to_best = {}
        for total_score, text in candidates:
            text_lower = text.lower()
            if text_lower not in lowercase_to_best or total_score > lowercase_to_best[text_lower][0]:
                lowercase_to_best[text_lower] = (total_score, text)
        candidates = list(lowercase_to_best.values())

        # Sort and prune
        candidates.sort(key=lambda x: x[0], reverse=True)
        if homophone_prune_threshold is not None and candidates:
            best = candidates[0][0]
            candidates = [c for c in candidates if best - c[0] <= homophone_prune_threshold]

        top_k = candidates[:num_homophone_beams]
        if not top_k:
            continue

        new_tuples = [(score, text) for score, text in top_k]

        old_best_lm_score = context_text_tuples[0][0] if context_text_tuples else 0.0
        new_best_lm_score = new_tuples[0][0]
        acoustic_score = base_score - old_best_lm_score
        new_score = acoustic_score + new_best_lm_score

        update_batch_indices.append(b)
        update_beam_indices.append(k)
        update_scores.append(new_score)
        update_context_texts.append((b, k, new_tuples))
        update_hashes.append((b, k, hash(new_tuples[0][1])))

    # Apply updates
    if update_batch_indices:
        beam_hyps.scores[update_batch_indices, update_beam_indices] = torch.tensor(
            update_scores, device=beam_hyps.scores.device, dtype=beam_hyps.scores.dtype
        )

    for b, k, new_tuples in update_context_texts:
        beam_hyps.context_texts[b][k] = new_tuples

    for b, k, hash_value in update_hashes:
        beam_hyps.context_texts_hash[b][k] = hash_value


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
        beam_hyps.context_texts_hash[b][k] = hash(updated_tuples[0][1])
