import torch
from torch import nn
from .base_model import BaseTimeMaskedModel

class GRU_24(BaseTimeMaskedModel):
    
    """GRU‑based speech encoder.
    
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

    def __init__(
        self,
        *,
        neural_dim: int,
        n_classes: int,
        hidden_dim: int,
        layer_dim: int,
        nDays: int,
        dropout: float,
        input_dropout: float,
        strideLen: int,
        kernelLen: int,
        bidirectional: bool,
        max_mask_pct: float,
        num_masks: int
    ) -> None:
        
        super().__init__(max_mask_pct=max_mask_pct, num_masks=num_masks)

        # Store constructor args
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.bidirectional = bidirectional

        # === Input processing layers ===
        self.inputLayerNonlinearity = nn.Softsign()
        self.unfolder = nn.Unfold((self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen)
        self.dayWeights = nn.Parameter(torch.randn(self.nDays, self.neural_dim, self.neural_dim))
        self.dayBias = nn.Parameter(torch.zeros(self.nDays, 1, self.neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x].copy_(torch.eye(self.neural_dim))

        for x in range(nDays):
            setattr(self, f"inpLayer{x}", nn.Linear(self.neural_dim, self.neural_dim))
            inp_layer: nn.Linear = getattr(self, f"inpLayer{x}")
            inp_layer.weight.data.add_(torch.eye(self.neural_dim))

        self.inputDropoutLayer = nn.Dropout(p=self.input_dropout)

        # === GRU ===
        self.gru_decoder = nn.GRU(
            self.neural_dim * self.kernelLen,
            self.hidden_dim,
            self.layer_dim,
            batch_first=True,
            dropout=self.dropout if layer_dim > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # === Optional post‑RNN block ===
        rnn_out_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
    
        self.post_rnn_block = nn.Identity()

        # === Final linear projection ===
        self.fc_decoder_out = nn.Linear(rnn_out_dim, n_classes + 1)  # +1 for CTC blank

        
    def forward(self, neuralInput: torch.Tensor, X_len: torch.Tensor, participant_id: torch.Tensor, 
                dayIdx: torch.Tensor) -> torch.Tensor:
        
        
        """Parameters
        ----------
        neuralInput : torch.Tensor, shape (batch, time, neural_dim)
        X_len       : torch.Tensor, shape (batch,), lengths before padding
        participant_id : torch.Tensor, shape (batch, ), index specifying the participant
        dayIdx      : torch.Tensor, shape (batch,), index specifying the session/day of each sample
        """
        

        
        # --- SpecAugment‑style time masking (training only) ---
        if self.training and self.max_mask_pct > 0:
            neuralInput, _ = self.apply_time_masking(neuralInput, X_len, mask_value=0)

        # --- Day‑specific affine transform ---
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)  # (B, C, C)
        transformedNeural = torch.einsum("btd,bdk->btk", neuralInput, dayWeights) + torch.index_select(
            self.dayBias, 0, dayIdx
        )
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        
        neuralInput = self.inputDropoutLayer(neuralInput)

        # --- Temporal unfolding (stride / kernel) ---
        stridedInputs = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)),
            (0, 2, 1),
        )
        
        # --- GRU encoding ---
        h0_dim = self.layer_dim * 2 if self.bidirectional else self.layer_dim
        h0 = torch.zeros(h0_dim, transformedNeural.size(0), self.hidden_dim, device=neuralInput.device)
        hid, _ = self.gru_decoder(stridedInputs, h0.detach())  # (B, T', H[*2])

        # --- Optional post‑RNN refinement ---
        hid = self.post_rnn_block(hid)

        # --- Projection to token logits ---
        seq_out = self.fc_decoder_out(hid)  # (B, T', n_classes+1)
        return seq_out
    
    def compute_length(self, X_len):
        
        return  ((X_len - self.kernelLen) / self.strideLen).to(torch.int32)
    
   