import torch 
import torch.nn as nn
from torch import Tensor
import math
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
    
    def __init__(self, *, samples_per_patch, features_list, dim, depth, heads, mlp_dim_ratio, embed_mlp_ratio,
                 dim_head, dropout, input_dropout,
                 nClasses, max_mask_pct, num_masks, gaussianSmoothWidth, kernel_size, num_participants):
   
        super().__init__(max_mask_pct=max_mask_pct, num_masks=num_masks)

        self.samples_per_patch = samples_per_patch
        self.features_list = features_list
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim_ratio = mlp_dim_ratio
        self.embed_mlp_ratio = embed_mlp_ratio #! new
        self.dim_head = dim_head
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.nClasses = nClasses
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.kernel_size = kernel_size
        self.num_participants = num_participants
        
        # self.mask_token = nn.Parameter(torch.randn(self.patch_dim))  
        self.patch_embedders = nn.ModuleList([])
        
        for pid in range(self.num_participants):
        
            feature_size = self.features_list[pid]
                    
            patch_dim = self.samples_per_patch*feature_size
            embed_intermediate_dim = int(patch_dim * self.embed_mlp_ratio) #! new
            
            self.patch_embedders.append(
            nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                        p1=self.samples_per_patch, p2=feature_size),
                nn.LayerNorm(patch_dim),
                # nn.Linear(patch_dim, self.dim),
                #! new
                nn.Linear(patch_dim, embed_intermediate_dim),
                nn.GELU(), # A non-linear activation function
                nn.Linear(embed_intermediate_dim, self.dim),
                #!
                nn.LayerNorm(self.dim)
            ))

            # nn.Sequential(
            #         # --- NEW CONVOLUTIONAL STEM ---
            #         # Input is (B, 1, T, F)
            #         nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding='same'),
            #         nn.GELU(),
            #         nn.LayerNorm([16, neural_T, feature_size]), # You'll need to pass T or use adaptive norm
                    
            #         nn.Conv2d(in_channels=16, out_channels=conv_out_channels, kernel_size=(3, 3), stride=1, padding='same'),
            #         nn.GELU(),
            #         # The output shape is now (B, conv_out_channels, T, F)
                    
            #         # --- Patchify the CONVOLVED output ---
            #         # Note: 'c' is now conv_out_channels, not 1. And p2 is the number of features.
            #         Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
            #                   p1=self.samples_per_patch, p2=feature_size),
                    
            #         # --- Standard Projection ---
            #         nn.LayerNorm(patch_dim),
            #         nn.Linear(patch_dim, self.dim),
            #         nn.LayerNorm(self.dim)
            #     )
                

        self.dropout_layer = nn.Dropout(self.input_dropout)
  
        self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim_ratio, 
                                    self.dropout, use_relative_bias=True)
    
        self.projection = nn.Linear(self.dim, nClasses+1)
        
    def forward(self, neuralInput, X_len, participant_idx=None, day_idx=None):
        """
        Args:
            neuralInput: Tensor of shape (B, T, F)
            X_len: Tensor of shape 
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

        # Create temporal mask
        temporal_mask = create_temporal_mask(seq_len, device=x.device)

        x = self.transformer(x, mask=temporal_mask)
        
        out = self.projection(x)
    
        return out
    
    def compute_length(self, X_len):
        
        # computing ceiling because I pad X to be divisible by path_height
        return torch.ceil(X_len / self.samples_per_patch).to(dtype=torch.int32)
    