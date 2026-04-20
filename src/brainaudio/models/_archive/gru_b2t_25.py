import torch
from torch import nn
from .base_model import BaseTimeMaskedModel


class GRU_25(BaseTimeMaskedModel):
    """GRU-based speech encoder for B2T '25.

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
        Number of distinct recording sessions / days (used for day-specific affine transforms).
    dropout : float
        Dropout probability within the GRU.
    input_dropout : float
        Dropout probability applied to inputs after the day-specific transform.
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
    samples_per_patch : int
        Temporal resolution of one patch, passed to the base class time masker.
    shared_input : bool
        If ``True``, all samples share a single input transform (day index is ignored).
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
        num_masks: int,
        samples_per_patch: int = 1,
        shared_input: bool = False,
    ):
        super().__init__(max_mask_pct=max_mask_pct, num_masks=num_masks, samples_per_patch=samples_per_patch)

        self.neural_dim = neural_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.layer_dim = layer_dim
        self.nDays = nDays
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.kernelLen = kernelLen
        self.strideLen = strideLen
        self.bidirectional = bidirectional
        self.shared_input = shared_input

        self.day_layer_activation = nn.Softsign()

        n_day_weights = 1 if shared_input else nDays
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(n_day_weights)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(n_day_weights)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)

        input_size = self.neural_dim * self.kernelLen if self.kernelLen > 0 else self.neural_dim

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.layer_dim,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        self.out = nn.Linear(self.hidden_dim, self.n_classes + 1)
        nn.init.xavier_uniform_(self.out.weight)

        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, x, x_len, day_idx):
        """
        Parameters
        ----------
        x       : torch.Tensor, shape (batch, time, neural_dim)
        x_len   : torch.Tensor, shape (batch,), lengths before padding
        day_idx : torch.Tensor, shape (batch,), index specifying the session/day of each sample
        """

        if self.training and self.max_mask_pct > 0:
            x, _ = self.apply_time_masking(x, x_len, mask_value=0)

        idx = torch.zeros_like(day_idx) if self.shared_input else day_idx
        day_weights = torch.stack([self.day_weights[i] for i in idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        if self.kernelLen > 0:
            x = x.unsqueeze(1).permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.kernelLen, self.strideLen)
            x_unfold = x_unfold.squeeze(2).permute(0, 2, 3, 1)
            x = x_unfold.reshape(x_unfold.size(0), x_unfold.size(1), -1)

        states = self.h0.expand(self.layer_dim, x.shape[0], self.hidden_dim).contiguous()
        output, _ = self.gru(x, states)
        return self.out(output)

    def compute_length(self, X_len):
        return ((X_len - self.kernelLen) / self.strideLen + 1).to(torch.int32)
