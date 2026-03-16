"""
Utilities for dynamic chunked attention in transformer models.

Provides configuration management and mask generation for streaming inference
with left-context constraints.
"""

import torch
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ChunkConfig:
    
    """Configuration for chunked attention.
    
    Attributes:
        chunk_size: Number of patches (tokens) per chunk. None means full context.
        context_chunks: Number of left context chunks to attend to. None means all previous chunks.
    """
    chunk_size: Optional[int]
    context_chunks: Optional[int] 

    def is_full_context(self) -> bool:
        return self.chunk_size is None


class ChunkConfigSampler:
    
    def __init__(
        self,
        *,
        chunk_size_min: int,
        chunk_size_max: int,
        context_sec_range: Tuple[float, float],
        timestep_duration_sec: float,
        chunkwise_prob: float = 1.0,
        left_constrain_prob: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:

        """Initialize demi-chunk config sampler for training-time augmentation.

        Chunk size is a binary choice: chunk_size_min with probability chunkwise_prob,
        chunk_size_max with probability 1 - chunkwise_prob. Always chunked during training.

        Args:
            chunk_size_min: Chunk size selected with probability chunkwise_prob.
            chunk_size_max: Chunk size selected with probability 1 - chunkwise_prob.
            context_sec_range: Range (min, max) from which to sample context duration in seconds.
            timestep_duration_sec: Duration of each patch in seconds.
            chunkwise_prob: Probability of using chunk_size_min (vs chunk_size_max).
            left_constrain_prob: Probability of restricting left context when chunking is enabled.
            seed: Random seed for reproducibility.
        """

        if chunk_size_max < chunk_size_min:
            raise ValueError(f"Chunk size range fault: max size is {chunk_size_max} and min size is {chunk_size_min}")

        context_sec_min, context_sec_max = context_sec_range
        if context_sec_max < context_sec_min:
            raise ValueError(f"Context sec range fault: max size is {context_sec_max} and min size is {context_sec_min}")

        self.chunk_size_min = max(1, int(chunk_size_min))
        if chunk_size_max != float('inf'):
            self.chunk_size_max = max(1, int(chunk_size_max))
        else:
            self.chunk_size_max = None
        self.context_sec_range = context_sec_range
        
        if timestep_duration_sec <= 0:
            raise ValueError("timestep_duration_sec must be positive.")
        self.timestep_duration_sec = timestep_duration_sec

        self.left_constrain_prob = max(0.0, min(1.0, float(left_constrain_prob)))
        self.chunkwise_prob = max(0.0, min(1.0, float(chunkwise_prob)))
        self._rng = random.Random(seed)

    def _sample_range(self, range_values: Optional[Tuple[float, float]], dtype: str) -> Optional[float]:
        
        """Sample a value from a range, casting to the specified dtype ('int' or 'float')."""
        
        if range_values is None:
            return None
        low, high = range_values

        if low == float('inf') or high == float('inf'):
            return float('inf')

        if dtype == 'int':
            low = max(0, int(low))
            high = max(low, int(high))
            if low == high:
                return low
            return self._rng.randint(low, high)
        
        elif dtype == 'float':
            low = max(0.0, float(low))
            high = max(low, float(high))
            if low == high:
                return low
            return self._rng.uniform(low, high)
        
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def sample(self) -> ChunkConfig:

        """Sample a chunk configuration for one training step.

        Chunk size: chunk_size_min with probability chunkwise_prob,
                    chunk_size_max with probability 1 - chunkwise_prob.
        Left context sampling: unchanged.
        """

        # Binary chunk size choice
        if self._rng.random() < self.chunkwise_prob:
            chunk_size = self.chunk_size_min
        else:
            chunk_size = self.chunk_size_max

        # Left context sampling (unchanged)
        if self.left_constrain_prob < 1.0 and self._rng.random() > self.left_constrain_prob:
            context_chunks = None
        else:

            context_sec = self._sample_range(self.context_sec_range, dtype='float')

            if context_sec is None or chunk_size is None or context_sec == float('inf'):
                context_chunks = None
            else:
                total_context_timesteps = context_sec / self.timestep_duration_sec
                context_chunks = math.ceil(total_context_timesteps / chunk_size)

        return ChunkConfig(chunk_size=chunk_size, context_chunks=context_chunks)


def create_dynamic_chunk_mask(seq_len: int, config: ChunkConfig, device=None):
    """Create block-causal attention mask for dynamic chunked attention.
    
    Args:
        seq_len: Total sequence length T (number of patches/timesteps).
        config: ChunkConfig with chunk_size and context_chunks.
        device: Device for the mask tensor.
    
    Returns:
        Attention mask of shape (1, 1, T, T) or None for full context.
    """
    if config.is_full_context():
        return None

    chunk_size = max(1, min(int(config.chunk_size), seq_len))
    chunk_ids = torch.arange(seq_len, device=device) // chunk_size
    query_chunk_ids = chunk_ids.unsqueeze(1)
    key_chunk_ids = chunk_ids.unsqueeze(0)
    
    if config.context_chunks is None:
        lower_bound = torch.zeros_like(query_chunk_ids)
        upper_bound = query_chunk_ids
    else:
        context_chunks = max(0, int(config.context_chunks))
        lower_bound = (query_chunk_ids - context_chunks).clamp(min=0)
        upper_bound = query_chunk_ids

    mask = (key_chunk_ids >= lower_bound) & (key_chunk_ids <= upper_bound)
    return mask.unsqueeze(0).unsqueeze(0)
