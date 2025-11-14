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

@dataclass
class ChunkConfig:
    """Configuration for chunked attention.
    
    chunk_size: the number of patches (tokens) per chunk. If None, use full context.
    context_chunks: the number of left context chunks to attend to. If None, attend to all previous chunks.
    
    """
    chunk_size: Optional[int]
    context_chunks: Optional[int] 

    def is_full_context(self) -> bool:
        return self.chunk_size is None
    
class ChunkConfigSampler:
    def __init__(
        self,
        *,
        chunk_size_range: Tuple[int, int],
        context_sec_range: Tuple[float, float],
        timestep_duration_sec: float,
        chunkwise_prob: float = 1.0,
        left_constrain_prob: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        
        
        """
        chunk_size_range: Tuple[int, int], the range form which to sample chunk sizes from
        context_sec_range: Tuple[float, float], the range from which to sample context seconds
        timestep_duration_sec: float, the duration of each patch in seconds     
        chunkwise_prob: float, the probability of using chunkwise attention
        left_constrain_prob: float, the probability of applying left-constrained context given chunkwise attention
        seed: Optional[int]
        """
        
        chunk_size_min, chunk_size_max = chunk_size_range
        if chunk_size_max < chunk_size_min:
            raise ValueError(f"Chunk size range fault: max size is {chunk_size_max} and min size is {chunk_size_min}")
        
        context_sec_min, context_sec_max = context_sec_range
        if context_sec_max < context_sec_min:
            raise ValueError(f"Context sec range fault: max size is {context_sec_max} and min size is {context_sec_min}")
        
        self.chunk_size_range = chunk_size_range
        self.context_sec_range = context_sec_range
        
        if timestep_duration_sec <= 0:
            raise ValueError("timestep_duration_sec must be positive.")
        self.timestep_duration_sec = timestep_duration_sec

        self.left_constrain_prob = max(0.0, min(1.0, float(left_constrain_prob)))
        self.chunkwise_prob = max(0.0, min(1.0, float(chunkwise_prob)))
        self._rng = random.Random(seed)

    # --- MODIFICATION: Combined into one function ---
    def _sample_range(self, range_values: Optional[Tuple[float, float]], dtype: str) -> Optional[float]:
        """
        Samples a value from a given range, casting to the specified dtype.
        dtype: 'int' or 'float'
        """
        if range_values is None:
            return None
        low, high = range_values

        if low == float('inf') or high == float('inf'):
            return float('inf')

        # Handle type-specific logic
        if dtype == 'int':
            low = max(0, int(low))
            high = max(low, int(high))
            if low == high:
                return low
            return self._rng.randint(low, high) # Sample int
        
        elif dtype == 'float':
            low = max(0.0, float(low))
            high = max(low, float(high))
            if low == high:
                return low
            return self._rng.uniform(low, high) # Sample float
        
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    # --- MODIFICATION: Updated sample() to use the new function ---
    def sample(self) -> ChunkConfig:
        if self._rng.random() > self.chunkwise_prob:
            # Case for no chunking
            return ChunkConfig(chunk_size=None, context_chunks=None)

        # 1. Sample chunk size (as int)
        chunk_size = self._sample_range(self.chunk_size_range, dtype='int')
        
        if chunk_size == float('inf'):
            return ChunkConfig(chunk_size=None, context_chunks=None)
        
        chunk_size = max(1, int(chunk_size))

        # 2. Determine context
        if self.left_constrain_prob < 1.0 and self._rng.random() > self.left_constrain_prob:
            # Case: No left-constrained context
            context_chunks = None
        else:
            # Case: Left-constrained context
            # Sample context_sec (as float)
            context_sec = self._sample_range(self.context_sec_range, dtype='float')
        
            if context_sec is None or context_sec == float('inf'):
                context_chunks = None
            else:
                # The key calculation
                total_context_timesteps = context_sec / self.timestep_duration_sec # number of patches in left context size
                context_chunks = math.ceil(total_context_timesteps / chunk_size) # number of left context chunks
        
        return ChunkConfig(chunk_size=chunk_size, context_chunks=context_chunks)
    
    
def create_dynamic_chunk_mask(seq_len: int, config: ChunkConfig, device=None):
    
    """
    Creates a (1, 1, T, T) block-causal attention mask for dynamic chunked attention.
    
    seq_len: Total sequence length T (number of patches/timesteps).
    config: A ChunkConfig object containing:
        - chunk_size: The number of timesteps in a single chunk.
        - context_chunks: The number of *past* chunks a query chunk can attend to.
    """
    
    # 1. Check for Full Context
    # If chunk_size is None, it means we're in "full context" mode (non-streaming).
    # The Transformer should use its default full attention, so we return None.
    if config.is_full_context():
        return None

    # 2. Prepare Chunk IDs
    
    # Ensure chunk_size is at least 1 and not larger than the sequence itself.
    chunk_size = max(1, min(int(config.chunk_size), seq_len))
    
    # Assign an ID to each timestep (patch) based on which chunk it belongs to.
    # E.g., if seq_len=10 and chunk_size=4:
    #   - timesteps [0, 1, 2, 3] are in chunk 0
    #   - timesteps [4, 5, 6, 7] are in chunk 1
    #   - timesteps [8, 9]       are in chunk 2
    # chunk_ids will be: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    chunk_ids = torch.arange(seq_len, device=device) // chunk_size

    # 3. Create Query-Key Chunk ID Matrices
    
    # query_chunk_ids: (T, 1) matrix. Represents the chunk ID for each "query" timestep (rows).
    #   E.g., [[0], [0], [0], [0], [1], [1], [1], [1], [2], [2]]
    query_chunk_ids = chunk_ids.unsqueeze(1)
    
    # key_chunk_ids: (1, T) matrix. Represents the chunk ID for each "key" timestep (columns).
    #   E.g., [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2]]
    key_chunk_ids = chunk_ids.unsqueeze(0)
    
    # By broadcasting these, we can create a (T, T) matrix where mask[i, j]
    # compares the chunk ID of query 'i' with the chunk ID of key 'j'.

    # 4. Define the Attention "Vision Cone" (Bounds)
    
    if config.context_chunks is None:
        # --- UNLIMITED CONTEXT (Full Causal) ---
        # The query can see all past chunks and its own chunk.
        # lower_bound = 0 (can see back to the very first chunk)
        # upper_bound = query_chunk_ids (can see up to its own chunk)
        lower_bound = torch.zeros_like(query_chunk_ids)
        upper_bound = query_chunk_ids
    else:
        # --- LIMITED CONTEXT (Sliding Window) ---
        # The query can only see 'context_chunks' in the past.
        context_chunks = max(0, int(config.context_chunks))
        
        # lower_bound: The *earliest* chunk ID a query can see.
        # E.g., if query is in chunk 5 and context_chunks=2:
        # lower_bound = 5 - 2 = 3. (It can see chunks 3, 4, 5).
        # .clamp(min=0) ensures we don't go below chunk 0.
        lower_bound = (query_chunk_ids - context_chunks).clamp(min=0)
        
        # upper_bound: The *latest* chunk ID a query can see (its own chunk).
        # E.g., if query is in chunk 5, upper_bound = 5.
        upper_bound = query_chunk_ids

    # 5. Create the Mask
    
    # This is the core logic.
    # A query 'i' can attend to a key 'j' IF AND ONLY IF:
    # 1. The key's chunk ID is GREATER than or equal to the lower_bound
    #    (key_chunk_ids >= lower_bound)
    # 2. The key's chunk ID is LESS than or equal to the upper_bound (causality)
    #    (key_chunk_ids <= upper_bound)
    #
    # The result is a (T, T) boolean matrix.
    mask = (key_chunk_ids >= lower_bound) & (key_chunk_ids <= upper_bound)

    # 6. Final Formatting
    
    # PyTorch attention mechanisms expect a mask shape of (Batch, Heads, T, T)
    # or (Batch, 1, T, T).
    # .unsqueeze(0).unsqueeze(0) adds the 'Batch' and 'Heads' dimensions.
    # The mask will be broadcast across all batches and heads.
    # Final shape: (1, 1, T, T)
    return mask.unsqueeze(0).unsqueeze(0)


class Attention(nn.Module):
    
    def __init__(self, dim, heads, dim_head, dropout, max_rel_dist=200, use_relative_bias=True):
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
      
    def forward(self, x, temporal_mask=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

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

        return self.to_out(out)

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

    def forward(self, x, mask=None):
        for attn, ffn in self.layers:
            x = attn(x, temporal_mask=mask) + x
            x = ffn(x) + x
        return self.norm(x)
    
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
        
    # --- MODIFICATION: Updated to use time-based config ---
    def _build_chunk_config(self, data, timestep_duration_sec: float) -> Optional[ChunkConfig]:
        """Builds a deterministic ChunkConfig for evaluation from a config dict."""
        if not data:
            return ChunkConfig(None, None) # Default to full context

        chunk_size = data.get("chunk_size")
        context_sec = data.get("context_sec") # <-- Read context_sec

        # If either is missing, use full context
        if chunk_size is None or context_sec is None:
            return ChunkConfig(None, None)

        # --- This is the key calculation, same as in the sampler ---
        chunk_size = max(1, int(chunk_size))
        total_context_timesteps = context_sec / timestep_duration_sec
        context_chunks = math.ceil(total_context_timesteps / chunk_size)
        # ---
        
        return ChunkConfig(
            chunk_size=chunk_size,
            context_chunks=context_chunks,
        )

    # --- MODIFICATION: Updated to use time-based config ---
    def _setup_chunked_attention(self, config_dict) -> None:

        min_chunk_size = config_dict["chunk_size_min"]
        max_chunk_size = config_dict["chunk_size_max"]
        
        # --- Read new time-based keys ---
        min_context_sec = config_dict["context_sec_min"]
        max_context_sec = config_dict["context_sec_max"]
        timestep_duration_sec = config_dict["timestep_duration_sec"]
        # ---
        
        left_constrain_prob = config_dict.get("left_constrain_prob", 0.0)
        chunkwise_prob = config_dict.get("chunkwise_prob", 0.0)
        
        sampler = ChunkConfigSampler(
            chunk_size_range=(min_chunk_size, max_chunk_size),
            # --- Pass new time-based args ---
            context_sec_range=(min_context_sec, max_context_sec),
            timestep_duration_sec=timestep_duration_sec,
            # ---
            chunkwise_prob=chunkwise_prob,
            left_constrain_prob=left_constrain_prob,
            seed=None,
        )

        self._train_sampler = sampler
        # --- Pass timestep_duration_sec to the eval builder ---
        self._eval_config = self._build_chunk_config(
            config_dict.get("eval"), 
            timestep_duration_sec
        )


    def _sample_chunk_config(self) -> Optional[ChunkConfig]:
        # --- MODIFICATION: Simplified (self._eval_config is now a full ChunkConfig object) ---
        if self.training:
            return self._train_sampler.sample()
        # Return eval config
        return self._eval_config


    @property
    def last_chunk_config(self) -> Optional[ChunkConfig]:
        return self._last_chunk_config

    def forward(self, neuralInput, X_len, participant_idx=None, day_idx=None):
        """
        Args:
            neuralInput: Tensor of shape (B, T, F)
            X_len: Tensor of shape (B, )
            participant_idx: integer ID, all data for a given batch must come from the same participant 
            dayIdx: Not used for Transformer 

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

        chunk_config = self._sample_chunk_config()
        
        # --- Store the config that was used ---
        self._last_chunk_config = chunk_config

        temporal_mask = create_dynamic_chunk_mask(
            seq_len, chunk_config, device=x.device
        )
        
        transformer_out = self.transformer(
            x,
            mask=temporal_mask
        )
        
        out = self.projection(transformer_out)
        
        if self.return_final_layer:
            return out, transformer_out
        
        return out