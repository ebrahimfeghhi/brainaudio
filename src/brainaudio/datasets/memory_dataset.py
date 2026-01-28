import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class MemorySpeechDataset(Dataset):
    """
    Adapter that makes the in-memory 'extended' dictionary list look 
    exactly like the disk-based LazySpeechDataset to the model.
    """
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list (list): The list of day-dictionaries loaded from your pickle
                              (e.g. val_extended_25['val_2x_b2t25']).
        """
        self.transform = transform
        self.samples = []

        # Flatten the hierarchical [Day -> List of Trials] structure into a flat list
        for day_idx, day_data in enumerate(data_list):
            if day_data is None:
                continue
            
            # We assume day_data contains lists for 'sentenceDat', 'text', etc.
            n_trials = len(day_data['sentenceDat'])
            
            for i in range(n_trials):
                self.samples.append({
                    'neural': day_data['sentenceDat'][i],
                    'text': day_data['text'][i],
                    'transcript': day_data['transcriptions'][i],
                    # In concatenated data, we usually treat 'day' as the day index 
                    # from the original list order
                    'day': day_idx 
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        neural_feats = sample['neural']
        text_seq = sample['text']
        
        # Calculate lengths
        neural_time_bins = neural_feats.shape[0]
        text_seq_len = len(text_seq)

        # Convert to Tensors
        neural_feats_tensor = torch.tensor(neural_feats, dtype=torch.float32)
        if self.transform:
            neural_feats_tensor = self.transform(neural_feats_tensor)

        # Match the exact tuple structure of LazySpeechDataset
        # (X, y, X_len, y_len, day, transcript)
        return (
            neural_feats_tensor,
            torch.tensor(text_seq, dtype=torch.int32),
            torch.tensor(neural_time_bins, dtype=torch.int32),
            torch.tensor(text_seq_len, dtype=torch.int32),
            torch.tensor(sample['day'], dtype=torch.int64),
            sample['transcript']
        )