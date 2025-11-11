import json
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, ConcatDataset

class LazySpeechDataset(Dataset):
    """
    Defines a "lazy" Pytorch dataset object.
    It is initialized with a list of file paths to .npz files.
    Data is loaded only when __getitem__ is called.
    """

    def __init__(self, file_paths: list[str], transform=None, 
                 return_transcript=False, return_alignments=False):
        
        self.file_paths = file_paths
        self.transform = transform
        self.return_transcript = return_transcript


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        
        # 1. Load *only* the single trial's .npz file
        #    allow_pickle=True is needed to load string arrays
        with np.load(self.file_paths[idx], allow_pickle=True) as trial_data:
            
            # Load neural data (always present)
            neural_feats = trial_data['sentenceDat']
            
            # Load text, using a dummy array if it's a test file
            # This prevents crashes in test mode and ensures collate_fn works
            text_seq = trial_data.get('text')
            
            # Load metadata, .item() extracts the scalar value
            day = trial_data['day'].item()
            
            transcript = trial_data['transcription'].item()


        # 2. Calculate lengths on-the-fly (no need to save them)
        neural_time_bins = neural_feats.shape[0]
        text_seq_len = len(text_seq)

        # 3. Convert to Tensors
        neural_feats_tensor = torch.tensor(neural_feats, dtype=torch.float32)
        
        if self.transform:
            neural_feats_tensor = self.transform(neural_feats_tensor)

        # 4. Build the output tuple
        items = [
            neural_feats_tensor,
            torch.tensor(text_seq, dtype=torch.int32),
            torch.tensor(neural_time_bins, dtype=torch.int32),
            torch.tensor(text_seq_len, dtype=torch.int32),
            torch.tensor(day, dtype=torch.int64),
        ]

        if self.return_transcript:
            items.append(transcript)

        return tuple(items)

def getDatasetLoaders(
    manifest_paths: list[str],
    batch_size: int,
    return_transcript=False,
    shuffle_train=True,
    num_workers=8 # Increased default, tune this for your machine
):
    
    '''
    Args:
        manifest_paths ([str]): list of paths to participant manifest.json files
        batch_size (int): batch size used for training
        ... (rest of args) ...
    '''

    def _padding(batch):
        """
        Pads a batch of data. This function is unchanged,
        but it now works robustly with test data because
        __getitem__ returns a dummy tensor instead of None.
        """

        if return_transcript:
            X, y, X_lens, y_lens, days, transcripts = zip(*batch)
        else:
            X, y, X_lens, y_lens, days = zip(*batch)
            
        padding_value = 0
        X_padded = pad_sequence(X, batch_first=True, padding_value=padding_value)
        y_padded = pad_sequence(y, batch_first=True, padding_value=padding_value)
        
        items = [X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days)]
        
        if return_transcript:
            items.append(transcripts)
            
        return tuple(items)
    
    train_data_loaders = []
    val_data_loaders = []
    test_data_loaders = []
    
    # Loop over each participant's manifest file
    for manifest_path in manifest_paths:
        
        # Load the manifest
        with open(manifest_path, "r") as handle:
            manifest = json.load(handle)
        
        # manifest is expected to be {'train': [...], 'val': [...], 'test': [...]}
        train_files = manifest.get('train', [])
        val_files = manifest.get('val', [])
        test_files = manifest.get('test', [])
        
        # Create Train loader
        if train_files:
            train_ds = LazySpeechDataset(
                train_files, 
                transform=None, # As in your original code
                return_transcript=return_transcript
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True, # <-- FASTER
                collate_fn=_padding,
            )
            train_data_loaders.append(train_loader)

        # Create Val loader
        if val_files:
            val_ds = LazySpeechDataset(
                val_files, 
                return_transcript=return_transcript
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=1, # As in your original code
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True, # <-- FASTER
                collate_fn=_padding,
            )
            val_data_loaders.append(val_loader)
        
        # Create Test loader
        if test_files:
            test_ds = LazySpeechDataset(
                test_files,
                return_transcript=return_transcript
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=1, # As in your original code
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True, # <-- FASTER
                collate_fn=_padding,
            )
            test_data_loaders.append(test_loader)

    return train_data_loaders, val_data_loaders, test_data_loaders