import torch 
from torch import nn
from ..base_model import BaseTimeMaskedModel

class GRU_25(BaseTimeMaskedModel):
    '''
    Defines the GRU decoder

    This class combines day-specific input layers, a GRU, and an output classification layer
    '''
    def __init__(self,
                 *,
                 neural_dim,
                 n_classes,
                 hidden_dim,
                 layer_dim, 
                 nDays,
                 dropout = 0.0,
                 input_dropout = 0.0,
                 strideLen = 0,
                 kernelLen = 0,
                 bidirectional = False, 
                 max_mask_pct = 0, 
                 num_masks = 0
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
        nDays : int
            Number of distinct recording sessions / days (used for day‑specific affine transforms).
        dropout : float
            Dropout probability within the GRU.
        input_dropout : float
            Dropout probability applied to inputs after the day‑specific transform.
        strideLen : int
            Stride for the unfolding operation (temporal down‑sampling).
        kernelLen : int
            Kernel length for the unfolding operation.
        bidirectional : bool
            If ``True``, use a bidirectional GRU.
        max_mask_pct : float
            Maximum proportion of the sequence to mask during SpecAugment‑style masking.
        num_masks : int
            Number of temporal masks to apply per sample when training.
        """
        
        super().__init__(max_mask_pct=max_mask_pct, num_masks=num_masks)
        
        self.neural_dim = neural_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.layer_dim = layer_dim 
        self.nDays = nDays

        self.dropout = dropout
        self.input_dropout = input_dropout
        
        self.kernelLen = kernelLen
        self.strideLen = strideLen

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.nDays)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.nDays)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        self.input_size = self.neural_dim

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
        
        
    def forward(self, x, x_len, day_idx):
        
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        '''
        
        # --- SpecAugment‑style time masking (training only) ---
        if self.training and self.max_mask_pct > 0:
            x, _ = self.apply_time_masking(x, x_len, mask_value=0)
        
        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

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
        
        return  ((X_len - self.kernelLen) / self.strideLen).to(torch.int32)
    