import torch 
import torch.nn as nn
from torch import Tensor
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from .base_model import BaseTimeMaskedModel

'''
Code adapted from Francois Porcher: https://github.com/FrancoisPorcher/vit-pytorch
Dynamic Chunking credits to SpeechBrain
'''

def pad_to_multiple(tensor, multiple, dim=1, value=0):
    
    """
    Pads `tensor` along `dim` so that its size is divisible by `multiple`.
    """
    
    size = tensor.size(dim)
    padding_needed = (multiple - size % multiple) % multiple
    if padding_needed == 0:
        return tensor
    pad_dims = [0] * (2 * tensor.dim())
    pad_dims[-2 * dim - 1] = padding_needed  # padding at the end
    return F.pad(tensor, pad_dims, value=value)


@dataclass
class ChunkConfig:
    """Configuration for chunked attention.
    
    chunk_size: the number of patches (tokens) per chunk. If None, use full context.
    context_chunks: the number of left context chunks to attend to. If None, attend to all previous chunks.
    
    """
    chunk_size: Optional[Union[int, float]]
    context_chunks: Optional[int] 

    def is_full_context(self) -> bool:
        return self.chunk_size is None
    
    def is_causal_attention(self) -> bool:
        """
        Returns True if this config is for full-context CAUSAL attention.
        This is distinct from chunk_size=inf, which is full-context BIDIRECTIONAL.
        """
        return self.chunk_size is None
    
class ChunkConfigSampler:
    def __init__(
        self,
        *,
        chunk_size_range: Tuple[Union[int, float], Union[int, float]],
        context_chunks_range: Tuple[int, int],
        chunkwise_prob: float = 1.0,
        left_constrain_prob: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        
        """
        chunk_size_range: Tuple (min_chunk_size, max_chunk_size). Use float('inf') for infinite chunk size.
        context_chunks_range: Tuple (min_context_chunks, max_context_chunks).
        chunkwise_prob: Probability of using chunked attention. (0.0 = no chunking, 1.0 = always chunking)
        left_constrain_prob: Probability of using left-constrained context. (0.0 = no left constraint, 1.0 = always left constraint)
        seed: Optional random seed for reproducibility.
        """
        
        
        chunk_size_min, chunk_size_max = chunk_size_range
        if chunk_size_max < chunk_size_min:
            raise ValueError(f"Chunk size range fault: max size is {chunk_size_max} and min size is {chunk_size_min}")
        context_chunks_min, context_chunks_max = context_chunks_range
        if context_chunks_max < context_chunks_min:
            raise ValueError(f"Context chunks range fault: max size is {context_chunks_max} and min size is {context_chunks_min}")
        # Store the new chunk size range
        self.chunk_size_range = chunk_size_range
        # Store the new context range
        self.context_chunks_range = context_chunks_range

        self.left_constrain_prob = max(0.0, min(1.0, float(left_constrain_prob)))
        self.chunkwise_prob = max(0.0, min(1.0, float(chunkwise_prob)))
        self._rng = random.Random(seed)

    def _sample_range(self, range_values: Optional[Tuple[Union[int, float], Union[int, float]]]) -> Optional[int]:
        if range_values is None:
            return None
        low, high = range_values
        
        # Handle the infinite case
        if low == float('inf') or high == float('inf'):
            # Assume if low is inf, high is also inf
            return float('inf')

        # Handle finite (int) case: sanity guard for defined range
        low = max(0, int(low))
        high = max(low, int(high))

        # Static chunking case
        if low == high:
            return low

        return self._rng.randint(low, high)

    def sample(self) -> ChunkConfig:
        if self.chunkwise_prob < 1.0 and self._rng.random() > self.chunkwise_prob:
            # Case for no chunking. run in full context mode
            return ChunkConfig(chunk_size=None, context_chunks=None)

        # Sample the single chunk size value
        chunk_size = self._sample_range(self.chunk_size_range)
        # Sample the single context_chunks value
        if self.left_constrain_prob < 1.0 and self._rng.random() > self.left_constrain_prob:
            # Case 1: for no left-constrained chunking context
            context_chunks = None   # “no limit” case
        else:
            # Case 2: for left-constrained chunking context
            context_chunks = self._sample_range(self.context_chunks_range)
        
        return ChunkConfig(chunk_size=chunk_size, context_chunks=context_chunks)


def create_dynamic_chunk_mask(seq_len: int, config: ChunkConfig, device=None):
    
    """
    seq_len: sequence length T after padding
    config: ChunkConfig object defining chunk_size and context_chunks   
    """
    
    if config.is_full_context():
        return None

    chunk_size = max(1, min(int(config.chunk_size), seq_len))
    chunk_ids = torch.arange(seq_len, device=device) // chunk_size

    query_chunk_ids = chunk_ids.unsqueeze(1)  # (T, 1)
    key_chunk_ids = chunk_ids.unsqueeze(0)    # (1, T)
    
    if config.context_chunks is None:
        lower_bound = torch.zeros_like(query_chunk_ids)
        upper_bound = query_chunk_ids
    else:
        context_chunks = max(0, int(config.context_chunks))
        lower_bound = (query_chunk_ids - context_chunks).clamp(min=0)
        upper_bound = query_chunk_ids

    mask = (key_chunk_ids >= lower_bound) & (key_chunk_ids <= upper_bound)
    return mask.unsqueeze(0).unsqueeze(0)

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def get_sinusoidal_pos_emb(seq_len, dim, device=None):
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def create_temporal_mask(seq_len, device=None):
    """
    Build a boolean mask of shape [1, 1, seq_len, seq_len] that allows each
    timestep t to attend to positions ≤ t

    Args:
        seq_len (int): sequence length T


    Returns:
        torch.Tensor: Boolean mask of shape [1, 1, T, T]
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)  # [T, 1] (query index)
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T] (key index)
    mask = j <= i                           # [T, T], True = allowed
    return mask.unsqueeze(0).unsqueeze(0)                  # [1, 1, T, T]

class Attention(nn.Module):
    
    def __init__(self, dim, heads, dim_head, dropout, max_rel_dist=200, use_relative_bias=True, cache_trimming=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # T5-style relative position bias
        self.max_rel_dist = max_rel_dist
        self.use_relative_bias = use_relative_bias
        if self.use_relative_bias:
            self.rel_pos_bias = nn.Embedding(2 * max_rel_dist - 1, 1)

        self.cache_trimming = cache_trimming 
      
    def forward(self, x, temporal_mask=None, past_key_value=None, use_cache:bool=False, max_cache_len:Optional[int]=None):
        """
        Args:
            past_key_value: optional tuple (past_keys, past_values) with shape [B, H, T_past, D]
            use_cache: if True, return updated (keys, values) alongside the output
            max_cache_len: keep only the most recent this-many time steps from the cache
        """
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k is not None and past_v is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
        if use_cache:
            cached_k, cached_v = k, v
            if max_cache_len is not None and self.cache_trimming:
                cached_k = cached_k[:, :, -max_cache_len:, :]
                cached_v = cached_v[:, :, -max_cache_len:, :]
        else:
            cached_k = cached_v = None

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b, h, n, n)

        # Add relative positional bias if enabled
        if self.use_relative_bias:
            seq_len = x.size(1)
            i = torch.arange(seq_len, device=x.device).unsqueeze(1)
            j = torch.arange(seq_len, device=x.device).unsqueeze(0)
            rel_pos = (i - j).clamp(-self.max_rel_dist + 1, self.max_rel_dist - 1) + self.max_rel_dist - 1
            rel_bias = self.rel_pos_bias(rel_pos).squeeze(-1).unsqueeze(0).unsqueeze(0) # shap seq_len x seq_len
            dots = dots + rel_bias

        if temporal_mask is not None:
            dots = dots.masked_fill(temporal_mask == 0, float('-inf'))
            
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), (cached_k, cached_v)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim_ratio, 
                 dropout=0., use_relative_bias=True):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        mlp_dim = mlp_dim_ratio * dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, 
                          dropout=dropout, use_relative_bias=use_relative_bias),
                FFN(dim=dim, hidden_dim=mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, mask=None, past_key_values: Optional[List[Optional[Tuple[Tensor, Tensor]]]] = None,
                use_cache:bool=False, max_cache_len:Optional[int]=None):

        next_key_values = [] if use_cache else None

        for layer_idx, (attn, ffn) in enumerate(self.layers):
            pkv = None
            if past_key_values is not None and len(past_key_values) > layer_idx:
                pkv = past_key_values[layer_idx]
            attn_out, new_pkv = attn(x, temporal_mask=mask, past_key_value=pkv, use_cache=use_cache, max_cache_len=max_cache_len)
            x = attn_out + x
            x = ffn(x) + x

            if use_cache:
                next_key_values.append(new_pkv)

        x = self.norm(x)
        return x, next_key_values
    

class TransformerModel(BaseTimeMaskedModel):
    
    def __init__(self, *, samples_per_patch, features_list, dim, depth, heads, mlp_dim_ratio,
                 dim_head, dropout, input_dropout,
                 nClasses, max_mask_pct, num_masks, num_participants, 
                 return_final_layer, chunked_attention=None):
   
        super().__init__(max_mask_pct=max_mask_pct, num_masks=num_masks)

        self.samples_per_patch = samples_per_patch
        self.features_list = features_list
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim_ratio = mlp_dim_ratio
        self.dim_head = dim_head
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.nClasses = nClasses
        self.num_participants = num_participants
        self.return_final_layer = return_final_layer
        self._train_sampler: Optional[ChunkConfigSampler] = None
        self._eval_config: Optional[ChunkConfig] = None
        self._last_chunk_config: Optional[ChunkConfig] = None
        if isinstance(chunked_attention, dict):
            self._setup_chunked_attention(chunked_attention)
        
        # self.mask_token = nn.Parameter(torch.randn(self.patch_dim))  
        self.patch_embedders = nn.ModuleList([])
        
        for pid in range(self.num_participants):
        
            feature_size = self.features_list[pid]
                    
            patch_dim = self.samples_per_patch*feature_size
            
            self.patch_embedders.append(
            nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                        p1=self.samples_per_patch, p2=feature_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, self.dim),
                nn.LayerNorm(self.dim)
            ))
            
            
        self.dropout_layer = nn.Dropout(self.input_dropout)
  
        self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim_ratio, 
                                    self.dropout, use_relative_bias=True)
    
        self.projection = nn.Linear(self.dim, nClasses+1)

    def compute_length(self, X_len):
        
        # computing ceiling because X is padded to be divisible by path_height
        return torch.ceil(X_len / self.samples_per_patch).to(dtype=torch.int32)
        
    # To separately build a chunkconfig for evaluation
    def _build_chunk_config(self, data) -> Optional[ChunkConfig]:
        chunk_size = data.get("chunk_size")
        context_chunks = data.get("context_chunks")

        return ChunkConfig(
            chunk_size=chunk_size,
            context_chunks=context_chunks,
        )

    # one-time initialization of 1. train config sampler and 2. build eval config 
    def _setup_chunked_attention(self, config_dict) -> None:

        min_chunk_size = config_dict["chunk_size_min"]
        max_chunk_size = config_dict["chunk_size_max"]
        min_context_range = config_dict["context_chunks_min"]
        max_context_range = config_dict["context_chunks_max"]
        
        left_constrain_prob = config_dict.get("left_constrain_prob", 0.0)
        chunkwise_prob = config_dict.get("chunkwise_prob", 0.0)
        sampler = ChunkConfigSampler(
        chunk_size_range=(min_chunk_size, max_chunk_size),
        context_chunks_range=(min_context_range, max_context_range),
        chunkwise_prob=chunkwise_prob,
        left_constrain_prob=left_constrain_prob,
        seed=config_dict.get("seed"),
        )

        self._train_sampler = sampler
        self._eval_config = self._build_chunk_config(config_dict.get("eval"))


    def _sample_chunk_config(self) -> Optional[ChunkConfig]:
        # Sample train config
        if self.training:
            return self._train_sampler.sample()
        # Return eval config
        return ChunkConfig(
            chunk_size=self._eval_config.chunk_size,
            context_chunks=self._eval_config.context_chunks,
        ) 

    @property
    def last_chunk_config(self) -> Optional[ChunkConfig]:
        return self._last_chunk_config

    def forward(self, neuralInput, X_len, participant_idx=None, day_idx=None, 
                past_key_values: Optional[List[Optional[Tuple[Tensor, Tensor]]]]=None,use_cache:bool=False):
        """
        Args:
            neuralInput: Tensor of shape (B, T, F)
            X_len: Tensor of shape (B, )
            participant_idx: integer ID, all data for a given batch must come from the same participant 
            dayIdx: Not used for Transformer 
            past_key_values: cached past kv matrices
            use_cache: flag for kv-cache usage

        Returns:
            Tensor: (B, num_patches, dim)
        """
        
        neuralInput = pad_to_multiple(neuralInput, multiple=self.samples_per_patch)
        
        neuralInput = neuralInput.unsqueeze(1)
            
        # add time masking
        if self.training and self.max_mask_pct > 0:
            patchify = self.patch_embedders[participant_idx][0]
            post_patchify = self.patch_embedders[participant_idx][1:]

            x = patchify(neuralInput)
            x, _ = self.apply_time_masking(x, X_len, mask_value=0)    
            x = post_patchify(x)

        else:
            x = self.patch_embedders[participant_idx](neuralInput)

        # apply input level dropout. 
        x = self.dropout_layer(x)
        
        b, seq_len, _ = x.shape

        temporal_mask = None

        chunk_config = self._sample_chunk_config()
        self._last_chunk_config = chunk_config

        max_cache_len = None

        applied_config: Optional[ChunkConfig] = None
        if chunk_config.is_causal_attention():
            # --- Case 1: Full Unidirectional CAUSAL ---
            # chunk_size is None (e.g., from chunkwise_prob: 0.0)
            temporal_mask = create_temporal_mask(seq_len, device=x.device)
            if use_cache:
                # Standard causal masking is compatible with a non-trimming cache
                max_cache_len = None
        
        else:
            # --- Case 2: Chunked / Bidirectional ---
            is_bidirectional = (
                chunk_config.chunk_size == float('inf') or
                chunk_config.chunk_size >= seq_len
            )

            if is_bidirectional:
                # --- Case 2a: Full-context BIDIRECTIONAL ---
                # (chunk_size=inf OR chunk_size >= seq_len)
                temporal_mask = None  # No mask = full bidirectional attention
                if use_cache:
                    raise RuntimeError("Bidirectional Models (chunk_size >= seq_len or inf) are not compatible with KV-cached inference")
            else:
                # --- Case 2b: Chunked Unidirectional (Streaming) ---
                # (chunk_size is a finite int < seq_len)
                applied_config = chunk_config
                temporal_mask = create_dynamic_chunk_mask(
                    seq_len, applied_config, device=x.device
                )
                
                # Setup KV cache trimming (safe to cast to int now)
                chunk_len = int(chunk_config.chunk_size)
                context_chunks = int(chunk_config.context_chunks or 0)
                max_cache_len = chunk_len * (context_chunks + 1)

        transformer_out, next_key_values = self.transformer(
            x,
            mask=temporal_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            max_cache_len=max_cache_len,
        )
        
        out = self.projection(transformer_out)
        
        if self.return_final_layer:
            return out, transformer_out

        if use_cache:
            return out, next_key_values
        return out
    
    
    