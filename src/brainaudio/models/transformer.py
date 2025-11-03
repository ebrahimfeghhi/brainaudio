import torch 
import torch.nn as nn
from torch import Tensor
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from .base_model import BaseTimeMaskedModel

'''
Code adapted from Francois Porcher: https://github.com/FrancoisPorcher/vit-pytorch
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
    """Configuration for chunked attention."""
    chunk_size: Optional[int]
    context_chunks: Optional[int] = None

    def is_full_context(self) -> bool:
        return self.chunk_size is None or self.chunk_size <= 0


class ChunkConfigSampler:
    def __init__(
        self,
        *,
        chunk_size_range: Tuple[int, int],
        context_chunks_range: Optional[Tuple[int, int]] = None,
        chunkwise_prob: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.chunk_size_min, self.chunk_size_max = chunk_size_range
        if self.chunk_size_max < self.chunk_size_min:
            self.chunk_size_max = self.chunk_size_min
        
        # Store the new context range
        self.context_chunks_range = context_chunks_range
        
        self.chunkwise_prob = max(0.0, min(1.0, float(chunkwise_prob)))
        self._rng = random.Random(seed)

    def _sample_range(self, range_values: Optional[Tuple[int, int]]) -> Optional[int]:
        if range_values is None:
            return None
        low, high = range_values
        low = max(0, int(low))
        high = max(low, int(high))

        return self._rng.randint(low, high)

    def sample(self) -> ChunkConfig:
        if self.chunkwise_prob < 1.0 and self._rng.random() > self.chunkwise_prob:
            # Return config for full context
            return ChunkConfig(chunk_size=None, context_chunks=None)

        chunk_size = self._rng.randint(self.chunk_size_min, self.chunk_size_max)
        # Sample the single context_chunks value
        context_chunks = self._sample_range(self.context_chunks_range)
        
        return ChunkConfig(chunk_size=chunk_size, context_chunks=context_chunks)


def create_dynamic_chunk_mask(seq_len: int, config: ChunkConfig, is_causal: bool, device=None):
    if config is None or config.is_full_context() or seq_len <= 0:
        return None

    chunk_size = max(1, min(int(config.chunk_size), seq_len))
    chunk_ids = torch.arange(seq_len, device=device) // chunk_size

    query_chunk_ids = chunk_ids.unsqueeze(1)  # (T, 1)
    key_chunk_ids = chunk_ids.unsqueeze(0)    # (1, T)
    
    max_chunk_id = int(chunk_ids.max().item())

    if config.context_chunks is None:
        lower_bound = torch.zeros_like(query_chunk_ids)
        if is_causal:
            upper_bound = query_chunk_ids
        else:
            upper_bound = torch.full_like(query_chunk_ids, max_chunk_id)
    else:
        context_chunks = max(0, int(config.context_chunks))
        lower_bound = (query_chunk_ids - context_chunks).clamp(min=0)
        if is_causal:
            upper_bound = query_chunk_ids
        else:
            upper_bound = (query_chunk_ids + context_chunks).clamp(max=max_chunk_id)

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
                 return_final_layer, bidirectional, chunked_attention=None):
   
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
        self.bidirectional = bidirectional
        self._train_sampler: Optional[ChunkConfigSampler] = None
        self._static_train_config: Optional[ChunkConfig] = None
        self._eval_config: Optional[ChunkConfig] = None
        self._last_chunk_config: Optional[ChunkConfig] = None
        self._chunking_active = False
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
        
    @staticmethod
    def _coerce_positive(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        int_value = int(value)
        return int_value if int_value > 0 else None

    @staticmethod
    def _coerce_nonnegative(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        int_value = int(value)
        return max(0, int_value)

    def _parse_range(self, config_dict, key_prefix: str) -> Optional[Tuple[int, int]]:
        min_key = f"{key_prefix}_min"
        max_key = f"{key_prefix}_max"
        if min_key not in config_dict and max_key not in config_dict:
            return None

        min_val = config_dict.get(min_key)
        max_val = config_dict.get(max_key, min_val)
        if min_val is None or max_val is None:
            return None

        min_val = max(0, int(min_val))
        max_val = max(min_val, int(max_val))
        return (min_val, max_val)

    def _build_chunk_config(self, data) -> Optional[ChunkConfig]:
        if not data:
            return None

        chunk_size = self._coerce_positive(data.get("chunk_size"))
        context_chunks = self._coerce_nonnegative(data.get("context_chunks"))

        if chunk_size is None and context_chunks is None:
            return None

        return ChunkConfig(
            chunk_size=chunk_size,
            context_chunks=context_chunks,
        )

    def _setup_chunked_attention(self, config_dict) -> None:
        enable_dynamic = bool(config_dict.get("enable_dynamic", False))


        eval_config = self._build_chunk_config(config_dict.get("eval"))
        self._eval_config = eval_config

        if enable_dynamic:

            chunk_size_min = max(1, int(config_dict["chunk_size_min"]))
            chunk_size_max = max(
                chunk_size_min,
                int(config_dict.get("chunk_size_max", chunk_size_min))
            )

            context_chunks_range = self._parse_range(config_dict, "context_chunks")
            chunkwise_prob = float(config_dict.get("chunkwise_prob", 1.0))
            seed = config_dict.get("seed")
            if seed is not None:
                seed = int(seed)

            self._train_sampler = ChunkConfigSampler(
                chunk_size_range=(chunk_size_min, chunk_size_max),
                context_chunks_range=context_chunks_range,
                chunkwise_prob=chunkwise_prob,
                seed=seed,
            )

        elif not enable_dynamic and eval_config is not None:
            # Static chunking: reuse evaluation config during training
            self._static_train_config = eval_config

        self._chunking_active = any(
            cfg is not None for cfg in (self._train_sampler, self._static_train_config, self._eval_config)
        )

    def _sample_chunk_config(self) -> Optional[ChunkConfig]:
        if not self._chunking_active:
            return None

        if self.training:
            if self._train_sampler is not None:
                return self._train_sampler.sample()
            if self._static_train_config is not None:
                return ChunkConfig(
                    chunk_size=self._static_train_config.chunk_size,
                    context_chunks=self._static_train_config.context_chunks,
                )
            return None

        if self._eval_config is None:
            return None

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
        if self.bidirectional and use_cache:
            raise RuntimeError("Bidirectional Models are not compatible with KV-cached inference")
            
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

        max_cache_len = None

        if chunk_config is not None and not chunk_config.is_full_context():
            chunk_len = int(chunk_config.chunk_size)
            context_chunks = int(chunk_config.context_chunks or 0)
            max_cache_len = chunk_len * (context_chunks + 1)  # left context + current chunk

        applied_config: Optional[ChunkConfig] = None

        if self.bidirectional:
            # In chunking case, we use chunked non-causal masking
            if chunk_config is not None and not chunk_config.is_full_context():
                temporal_mask = create_dynamic_chunk_mask(
                    seq_len, chunk_config, is_causal=False, device=x.device
                )
                applied_config = chunk_config
            # No temporal masking if bidirectional when not chunked
            else:
                temporal_mask = None
        else: 
            
            if chunk_config is None or chunk_config.is_full_context():
                temporal_mask = create_temporal_mask(seq_len, device=x.device)
            # In chunking case, we use chunked causal masking
            else:
                # In chunking case, sometimes we chunk, so we use chunked causal masking
                applied_config = chunk_config
                temporal_mask = create_dynamic_chunk_mask(
                    seq_len, applied_config, is_causal=True, device=x.device
                )
                # In chunking case, some times we don't chunk, so we use full context causal masking
                if temporal_mask is None:
                    temporal_mask = create_temporal_mask(seq_len, device=x.device)

        if applied_config is not None:
            self._last_chunk_config = applied_config
        elif chunk_config is not None and chunk_config.is_full_context():
            self._last_chunk_config = chunk_config
        else:
            self._last_chunk_config = None

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
    
    def compute_length(self, X_len):
        
        # computing ceiling because X is padded to be divisible by path_height
        return torch.ceil(X_len / self.samples_per_patch).to(dtype=torch.int32)
    