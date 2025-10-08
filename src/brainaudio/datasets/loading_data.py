import pickle 
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, ConcatDataset


class SpeechDataset(Dataset):

    def __init__(self, data, transform=None, return_transcript=False, pid=None):
        """
        Defines a Pytorch dataset object which returns neural data, output text labels, 
        neural data length, output text labels length, day idx, and optionally the ground
        truth transcript. 
        
        Args: 

            data (dict): dictionary containing neural data as well as associated text and meta data
            transform (function): if a function is passed, applies it to neural data.
            Set to None to ignore.
            return_transcript: returns the ground-truth transcript if True.
            
        """
        self.data = data
        self.transform = transform
        self.return_transcript = return_transcript

        self.n_days = len(data)

        self.neural_feats = []
        self.text_seqs = []
        self.neural_time_bins = []
        self.text_seq_lens = []
        self.days = []
        self.transcriptions = []
        self.participant_id = []
        
 
        for day in range(self.n_days):
            
            # check to see if data exists for that day
            if data[day] == None:
                continue

            n_trials = len(data[day]["sentenceDat"])
            
            for trial in range(n_trials):
                feats = data[day]["sentenceDat"][trial]
                self.neural_feats.append(feats)

                self.text_seqs.append(data[day]["text"][trial])
                self.neural_time_bins.append(feats.shape[0])
                self.text_seq_lens.append(data[day]["textLens"][trial])
                self.transcriptions.append(data[day]['transcriptions'][trial])
                self.days.append(day)
                self.participant_id.append(pid)


        self.n_trials = len(self.days)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)
        if self.transform:
            neural_feats = self.transform(neural_feats)

        items = [
            neural_feats,
            torch.tensor(self.text_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.text_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        ]

        if self.return_transcript:
            items.append(self.transcriptions[idx])

        return tuple(items)

def getDatasetLoaders(
    data_paths:list[str],
    batch_size:int,
):
    
    '''
    Args:
        datasetName ([str]): list of paths to dataset
        batchSize (int): batch size used for training
    
    Returns train and val datasets as Pytorch data loaders, 
    as well as full dataset as a list of all datasets.
    '''

    def _padding(batch):
    
        X, y, X_lens, y_lens, days = zip(*batch)
        padding_value = 0
        X_padded = pad_sequence(X, batch_first=True, padding_value=padding_value)
        y_padded = pad_sequence(y, batch_first=True, padding_value=padding_value)
        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )
    
    train_data_loaders = []
    val_data_loaders = []
    loadedData = []
    for i in range(len(data_paths)):
        
        datasetName = data_paths[i]
        with open(datasetName, "rb") as handle:
            ds = pickle.load(handle)
        loadedData.append(ds)
        train_ds = SpeechDataset(ds['train'], transform=None, pid=i)
        val_ds = SpeechDataset(ds['val'], pid=i)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=_padding,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_padding,
        )

        train_data_loaders.append(train_loader)
        val_data_loaders.append(val_loader)

    return train_data_loaders, val_data_loaders, loadedData