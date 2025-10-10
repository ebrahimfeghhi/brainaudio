
import torch
import pickle
import os
from typing import Dict, Any, List, Tuple
import numpy as np
from edit_distance import SequenceMatcher
import re
from tqdm import tqdm
from itertools import zip_longest
from torchaudio.functional import forced_align
from torch.nn.functional import log_softmax
import torchaudio.functional as F
import pickle

from brainaudio.models.transformer import TransformerModel
from brainaudio.models.gru_b2t_25 import GRU_25
from brainaudio.datasets.loading_data import getDatasetLoaders
from brainaudio.training.utils.augmentations import gauss_smooth

def load_model(folder: str, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    """
    Load a pre-trained model from a folder containing 'args' and 'modelWeights'.

    Args:
        folder (str): Path to folder containing 'args' (pickle) and 'modelWeights' (torch).
        device (torch.device): Device to map the model onto.

    Returns:
        torch.nn.Module: The loader model in eval mode, and the args file.
    """
    
    # Load args
    args_path = os.path.join(folder, "args")
    with open(args_path, "rb") as handle:
        config = pickle.load(handle)
        
    modelType = config['modelType']
    model_args = config['model'][modelType]
    
    if modelType == 'transformer':
        
        model = TransformerModel(features_list=model_args['features_list'], samples_per_patch=model_args['samples_per_patch'], dim=model_args['d_model'], depth=model_args['depth'], 
                        heads=model_args['n_heads'], mlp_dim_ratio=model_args['mlp_dim_ratio'],  dim_head=model_args['dim_head'], 
                        dropout=config['dropout'], input_dropout=config['input_dropout'], nClasses=config['nClasses'], 
                        max_mask_pct=config['max_mask_pct'], num_masks=config['num_masks'], gaussianSmoothWidth=config['gaussianSmoothWidth'], 
                        kernel_size=config['smooth_kernel_size'], num_participants=len(model_args['features_list']))


    elif modelType == 'gru':
        
        model = GRU_25(neural_dim=model_args['nInputFeatures'], n_classes=config['nClasses'], hidden_dim=model_args['nUnits'], 
            layer_dim=model_args['nLayers'], nDays=model_args['nDays'], dropout=config['dropout'], input_dropout=config['input_dropout'],
            strideLen=model_args['strideLen'], kernelLen=model_args['kernelLen'], gaussianSmoothWidth=config['gaussianSmoothWidth'], 
            kernel_size=config['smooth_kernel_size'], bidirectional=model_args['bidirectional'], max_mask_pct=config['max_mask_pct'], 
            num_masks=config['num_masks'])


    # Load weights
    ckpt_path = os.path.join(folder, "modelWeights")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    return model, config


def obtain_word_level_timespans(alignments, scores, ground_truth_sequence, transcript,
                                silence_token_id=40):
    
    """Computes word level start and end times.
    
    Parameters
    ----------
    alignments: torch array (T)
        Provides the frame-level alignments using CTC for each output frame, 
        where T is the number of output frames. Each entry is an integer which 
        corresponds to a token of the acoustic model.
        
    scores: torch array (T)
        Provides the log probability for the token in "alignments" at that frame. 
        
    ground_truth_sequence: torch array (N)
        The ground-truth sentence displayed to the participant. 
        Each entry is an integer corresponding to a phoneme or silence token. 
        
    transcript: str
        Ground truth word-level transcript.
        
    silence_token_id: int
        The integer corresponding to the silence token, which denotes word boundaries. 
        
    Returns
    -------
    list
        Entry i in the list contains the start and end times for each token of the ith
        word in the transcript. 
    """
    
    # Compute the length (number of phonemes) in each word
    word_lens = []
    current_word_length = 0
    for tok in ground_truth_sequence:
        if tok == silence_token_id:
            if current_word_length > 0:
                word_lens.append(current_word_length)
            current_word_length = 0 
            word_lens.append(1)
        else:
            current_word_length += 1
        
            
    def unflatten(list_, lengths):
        
        assert len(list_) == sum(lengths)
        i = 0
        ret = []
        # gather all tokens corresponding to a word
        for l in lengths:
            ret.append(list_[i : i + l])
            i += l
        return ret
    
    # apply CTC merging function
    token_spans = F.merge_tokens(alignments, scores)
    word_spans = unflatten(token_spans, word_lens)
    
    word_span_information = []
    
    for word_span, word in zip(word_spans, transcript):
        
        try:
            word_span_information.append([word_span[0].start, word_span[-1].end, word])
        except:
            breakpoint()
                
    return word_span_information


def generate_log_probs(model, args, partition, device):
    
    datasetPaths = ['/data2/brain2text/b2t_25/brain2text25.pkl', 
                    '/data2/brain2text/b2t_24/brain2text24']
    
    #datasetPaths = ['/data2/brain2text/b2t_24/brain2text24']
    #savePaths = ['/data2/brain2text/b2t_24/']
    
    participant_ids = [0, 1]
    
    savePaths =  ['/data2/brain2text/b2t_25/', 
                  '/data2/brain2text/b2t_24/']
    
    trainLoaders, valLoaders, loadedData = getDatasetLoaders(
        datasetPaths,
        args["batchSize"], 
        return_transcript=True, 
        shuffle_train=False
    )
    
    if partition == 'train':
        
        dataLoaders = trainLoaders
    
    elif partition == "val":
        
        dataLoaders = valLoaders
        
    model.eval()
    
    with torch.no_grad():
        
        # loop through data for each participant 
        for participant_id, dataLoader in zip(participant_ids, dataLoaders):
            
            if participant_id == 0:
                continue
            
            word_spans_dict = {}
        
            for batch in tqdm(dataLoader):
                
                X, y, X_len, y_len, dayIdxs, transcripts = batch

                # Move data to the specified device
                X = X.to(device)
                y = y.to(device)
                X_len = X_len.to(device)
                y_len = y_len.to(device)
                dayIdxs = dayIdxs.to(device)
                
                # replace padding value with -1 so it doesn't get interpreted as blank token                        
                X = gauss_smooth(X, device=device, smooth_kernel_size=args['smooth_kernel_size'], smooth_kernel_std=args['gaussianSmoothWidth'])
                adjusted_lens = model.compute_length(X_len)
                
                # B x T x C 
                logits = model.forward(X, X_len, participant_id, dayIdxs)
                
                current_batch_size = logits.shape[0]
                
                log_probs = log_softmax(logits)
                
                # B x T         
                for i in range(current_batch_size):        
                    
                    fa_labels, fa_probs = forced_align(
                        log_probs=log_probs[i:i+1, :adjusted_lens[i]], 
                        targets=y[i:i+1, :y_len[i]], 
                        blank=0
                    )
                    
                    dayIdx = int(dayIdxs[i])
                    
                    transcript = transcripts[i]
                    
                    words = transcript.split(' ')
                    transcript_list = [item for word in words for item in (word, 'SIL')]

                    word_spans = obtain_word_level_timespans(fa_labels[0], fa_probs[0], y[i:i+1, :y_len[i]].cpu().numpy().squeeze(), 
                                                             transcript=transcript_list, silence_token_id=40)
                    
                    if dayIdx in word_spans_dict.keys():
                        word_spans_dict[dayIdx].append(word_spans)
                    else:
                        word_spans_dict[dayIdx] = []
                        word_spans_dict[dayIdx].append(word_spans)
                        
                        
            savePath = f"{savePaths[participant_id]}word_alignments.pkl"
            
            with open(savePath, 'wb') as handle:
                pickle.dump(word_spans_dict, handle)
                        
                    
load_model_folder = "/data2/brain2text/b2t_combined/outputs_tm_transformer_b2t_24+25_large_wide/"  
device = "cuda:0"    

model, args = load_model(load_model_folder, device)
generate_log_probs(model, args, partition="train", device=device)
                

        
                    
                
            
            
    
        
    
    
    