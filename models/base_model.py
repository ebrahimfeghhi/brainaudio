import torch
from torch import nn

class BaseTimeMaskedModel(nn.Module):
    
    def __init__(self,
                 *,
                 max_mask_pct, 
                 num_masks):
        
        super().__init__()
        
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        
    def compute_length(self, X_len):
    
        raise NotImplementedError("You must implement a method to compute the output length.")

        
    def apply_time_masking(self, X, X_len, mask_value):
        
        """
        Fully vectorized time masking (no loops at all).
        
        Args:
            X: (B, P, D) input tensor
            X_len: (B,) valid lengths in timepoints (for Transformer, pass in X_len // self.patch_height)
            mask_value: value to fill masked entriesw with

        Returns:
            X_masked: (B, P, D) with masked patches
            mask: (B, P) boolean mask of where values were masked
            masked_indices: list of 1D LongTensors, each with indices of masked patches per batch
            unmasked_indices: list of 1D LongTensors, each with indices of unmasked patches per batch
        """
        
        B, P, D = X.shape
        device = X.device

        valid_lens = X_len
            
        max_mask_lens = (self.max_mask_pct * valid_lens).long()  # (B,)

        # Repeat B num_masks times to simulate multiple masks per sample
        B_rep = B * self.num_masks

        # Expand inputs for vectorized masking
        # repeat_interleave works like tile, so values corresponding to the same batch are next to each other
        valid_lens_rep = valid_lens.repeat_interleave(self.num_masks)            # (B * num_masks,)
        max_mask_lens_rep = max_mask_lens.repeat_interleave(self.num_masks)      # (B * num_masks,)
       
        t = (torch.rand(B_rep, device=device) * (max_mask_lens_rep + 1).float()).floor().long()  # (B * num_masks,)
            
        max_start = (valid_lens_rep - t + 1).clamp(min=1)
        
        t0 = (torch.rand(B_rep, device=device) * max_start.float()).floor().long()               # (B * num_masks,)

        # Build the global mask (B, P)
        arange = torch.arange(P, device=device).unsqueeze(0)       # (1, P)
        t0_exp = t0.unsqueeze(1)                                   # (B_rep, 1)
        t1_exp = (t0 + t).unsqueeze(1)                             # (B_rep, 1)
        mask_chunks = (arange >= t0_exp) & (arange < t1_exp)       # (B_rep, P)
        
        # Get index of sample in batch for each mask chunk
        batch_idx = torch.arange(B, device=device).repeat_interleave(self.num_masks)  # (B * num_masks,)

        # Now scatter all the masks into the full mask (B, P)
        patch_idx = mask_chunks.nonzero(as_tuple=False)  # (N_masked, 2)
        b_indices = batch_idx[patch_idx[:, 0]]           # (N_masked,)
        p_indices = patch_idx[:, 1]                      # (N_masked,)

        mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        mask[b_indices, p_indices] = True
        
        # mask: (B, P) boolean, True for masked
        #B, P = mask.shape
        # Apply the mask
        X_masked = X.clone()
        X_masked[mask] = mask_value

        return X_masked, mask
