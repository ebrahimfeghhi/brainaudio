import torch 
import torch.nn as nn
import math
from typing import Optional
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from .base_model import BaseTimeMaskedModel
from .chunking_utils import ChunkConfig, ChunkConfigSampler, create_dynamic_chunk_mask

"""
Transformer model with dynamic chunked left-context attention for streaming inference.

Code adapted from Francois Porcher: https://github.com/FrancoisPorcher/vit-pytorch
Dynamic chunking credits to SpeechBrain.
"""

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
    
def get_sinusoidal_pos_emb(seq_len, dim, device=None):
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


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
        if isinstance(chunked_attention, dict):
            self._setup_chunked_attention(chunked_attention)
        
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

    def _setup_chunked_attention(self, config_dict) -> None:
        
        """Set up chunk config sampler and eval config from training config dict."""

        min_chunk_size = config_dict["chunk_size_min"]
        max_chunk_size = config_dict["chunk_size_max"]
        min_context_sec = config_dict["context_sec_min"]
        max_context_sec = config_dict["context_sec_max"]
        timestep_duration_sec = config_dict["timestep_duration_sec"]
        
        left_constrain_prob = config_dict.get("left_constrain_prob", 0.0)
        chunkwise_prob = config_dict.get("chunkwise_prob", 0.0)
        
        sampler = ChunkConfigSampler(
            chunk_size_range=(min_chunk_size, max_chunk_size),
            context_sec_range=(min_context_sec, max_context_sec),
            timestep_duration_sec=timestep_duration_sec,
            chunkwise_prob=chunkwise_prob,
            left_constrain_prob=left_constrain_prob,
            seed=None,
        )

        self._train_sampler = sampler
        
        eval_chunk_size = config_dict["eval"]["chunk_size"]
        eval_context_secs = config_dict["eval"]["context_sec"]
        if eval_context_secs is not None and eval_chunk_size is not None:
            total_context_timesteps = eval_context_secs / config_dict["timestep_duration_sec"]
            eval_context_chunks = math.ceil(total_context_timesteps / eval_chunk_size)
        else:
            eval_context_chunks = None
        
        
        
        self._eval_config = ChunkConfig(
            chunk_size=eval_chunk_size,
            context_chunks=eval_context_chunks,
        )

    def _sample_chunk_config(self) -> Optional[ChunkConfig]:
        
        """Sample chunk config for current forward pass (train sampler or eval config).""" 
        if self.training:
            return self._train_sampler.sample()
        return self._eval_config


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