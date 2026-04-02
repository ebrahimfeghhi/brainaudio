import torch 
from torch import nn
import torch.nn.functional as F
from ..base_model import BaseTimeMaskedModel
from einops.layers.torch import Rearrange

class GRU_25_patch(BaseTimeMaskedModel):
    '''
    Defines the GRU decoder

    This class combines patch embedder layers, a GRU, and an output classification layer
    '''
    def __init__(self,
                 *,
                 neural_dim,
                 n_classes,
                 hidden_dim,
                 layer_dim, 
                 features_list,
                 num_participants,
                 dropout = 0.0,
                 input_dropout = 0.0,
                 strideLen = 0,
                 kernelLen = 0,
                 bidirectional = False, 
                 max_mask_pct = 0, 
                 num_masks = 0,
                 ):
        
        """GRU-based speech encoder.
        
        Parameters
        ----------
        neural_dim : int
            Number of neural input channels.
        n_classes : int
            Number of output classes (excluding the CTC blank).
        hidden_dim : int
            Hidden state dimensionality of the GRU.
        layer_dim : int
            Number of stacked GRU layers.
        num_participants: int
            Number of participants.
        features_list: [int]
            List of feature numbers that correspond to different participants.
        dropout : float
            Dropout probability within the GRU.
        input_dropout : float
            Dropout probability applied to inputs after patch embedding.
        strideLen : int
            Stride for the unfolding operation (temporal down-sampling).
        kernelLen : int
            Kernel length for the unfolding operation.
        bidirectional : bool
            If ``True``, use a bidirectional GRU.
        max_mask_pct : float
            Maximum proportion of the sequence to mask during SpecAugment-style masking.
        num_masks : int
            Number of temporal masks to apply per sample when training.
        """
        
        super().__init__(max_mask_pct=max_mask_pct, num_masks=num_masks)
        
        self.neural_dim = neural_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.layer_dim = layer_dim 
        self.num_participants = num_participants
        self.features_list = features_list

        self.dropout = dropout
        self.input_dropout = input_dropout
        
        self.kernelLen = kernelLen
        self.strideLen = strideLen

        self.dropout_layer = nn.Dropout(self.input_dropout)
        self.input_size = self.neural_dim

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
                nn.Linear(patch_dim, self.input_size),
                nn.LayerNorm(self.input_size)
            ))
        

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * kernelLen
        if self.kernelLen > 0:
            self.input_size *= self.kernelLen

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.hidden_dim,
            num_layers = self.layer_dim,
            dropout = self.dropout, 
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = bidirectional,
        )
                
        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.hidden_dim, self.n_classes+1)
        
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.hidden_dim)))
        
        
    def forward(self, x, x_len, participant_idx=None):
        
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        participant_idx: integer ID, all data for a given batch must come from the same participant
        '''
        
        # --- SpecAugment‑style time masking (training only) ---
        if self.training and self.max_mask_pct > 0:
            patchify = self.patch_embedders[participant_idx][0]
            post_patchify = self.patch_embedders[participant_idx][1:]

            x = patchify(x)
            x, _ = self.apply_time_masking(x, x_len, mask_value=0)
            x = post_patchify(x)
        else:
            x = self.patch_embedders[participant_idx](x)
        
        x = self.dropout_layer(x)


        # (Optionally) Perform input concat operation
        if self.kernelLen > 0: 
  
            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]
            
            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.kernelLen, self.strideLen)  # [batches, feature_dim, 1, num_patches, kernelLen]
            
            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, kernelLen]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, kernelLen, feature_dim]

            # Flatten last two dimensions (kernelLen and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
        # Determine initial hidden states
        states = self.h0.expand(self.layer_dim, x.shape[0], self.hidden_dim).contiguous()

        # Pass input through RNN 
        output, hidden_states = self.gru(x, states)

        # Compute logits
        logits = self.out(output)
        
        return logits
    
    def compute_length(self, X_len):
        
        return ((X_len - self.kernelLen) / self.strideLen).to(torch.int32)
    