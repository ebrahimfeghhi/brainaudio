
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
        
        if 'return_final_layer' not in config:
            config['return_final_layer'] = False
        
        model = TransformerModel(features_list=model_args['features_list'], samples_per_patch=model_args['samples_per_patch'], dim=model_args['d_model'], depth=model_args['depth'], 
                        heads=model_args['n_heads'], mlp_dim_ratio=model_args['mlp_dim_ratio'],  dim_head=model_args['dim_head'], 
                        dropout=config['dropout'], input_dropout=config['input_dropout'], nClasses=config['nClasses'], 
                        max_mask_pct=config['max_mask_pct'], num_masks=config['num_masks'], gaussianSmoothWidth=config['gaussianSmoothWidth'], 
                        kernel_size=config['smooth_kernel_size'], num_participants=len(model_args['features_list']), return_final_layer=config['return_final_layer'])


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
        
        word_span_information.append([word_span[0].start, word_span[-1].end, word])
                
    return word_span_information


def generate_and_save_logits(model, args, partition, device, 
                             dataset_paths, save_paths, participant_ids):
    
    """
    Runs the model forward pass and saves the output logits to a file.
    All dependencies are passed as arguments.

    Args:
        model (torch.nn.Module): The neural network model.
        args (dict): Dictionary of arguments.
        partition (str): The data partition to process ('train' or 'val').
        device (torch.device): The device to run the model on.
        dataset_paths (list): List of paths to the dataset files.
        save_paths (list): List of paths to save the output files.
        participant_ids (list): List of participant IDs.
    """
    
    print(f"--- Starting: Generating logits for '{partition}' partition ---")

    trainLoaders, valLoaders, _ = getDatasetLoaders(
        dataset_paths,
        args["batchSize"], 
        return_transcript=True, 
        shuffle_train=False
    )
    
    dataLoaders = trainLoaders if partition == 'train' else valLoaders
        
    model.eval()
    with torch.no_grad():
        for participant_id, dataLoader in zip(participant_ids, dataLoaders):
            print(f"Processing participant {participant_id}...")
            
            logits_data_by_day = {}
        
            for batch in tqdm(dataLoader, desc=f"P{participant_id} Logits"):
                X, y, X_len, y_len, dayIdxs, transcripts = batch

                X, y, X_len, y_len, dayIdxs = (
                    X.to(device), y.to(device), X_len.to(device), 
                    y_len.to(device), dayIdxs.to(device)
                )
                
                X = gauss_smooth(X, device=device, smooth_kernel_size=args['smooth_kernel_size'], smooth_kernel_std=args['gaussianSmoothWidth'])
                adjusted_lens = model.compute_length(X_len)
                logits = model.forward(X, X_len, participant_id, dayIdxs)
                log_probs = log_softmax(logits, dim=-1)
                
                for i in range(log_probs.shape[0]):
                    dayIdx = int(dayIdxs[i].item())
                    
                    sample_data = {
                        'log_probs': log_probs[i, :adjusted_lens[i]].cpu(),
                        'targets': y[i, :y_len[i]].cpu(),
                        'transcript': transcripts[i]
                    }
                    
                    if dayIdx not in logits_data_by_day:
                        logits_data_by_day[dayIdx] = []
                        
                    logits_data_by_day[dayIdx].append(sample_data)

            save_path = f"{save_paths[participant_id]}/logits_{partition}.pkl"
            print(f"Saving logits for participant {participant_id} to {save_path}")
            with open(save_path, 'wb') as handle:
                pickle.dump(logits_data_by_day, handle)
    
    print("--- Finished: Logit generation complete. ---")

# ==============================================================================
# FUNCTION 2: Compute Forced Alignments (Self-Contained)
# ==============================================================================
def compute_forced_alignments(partition, save_paths, participant_ids, 
                              silence_token_id=40, blank_id=0):
    """
    Loads pre-computed logits and computes forced alignments.
    All dependencies are passed as arguments.

    Args:
        partition (str): The data partition to process.
        save_paths (list): List of paths where logits are and alignments will be saved.
        participant_ids (list): List of participant IDs.
        silence_token_id (int): The ID for the silence token.
        blank_id (int): The ID for the CTC blank token.
    """
    print(f"--- Starting: Computing forced alignments for '{partition}' partition ---")

    for participant_id in participant_ids:
        print(f"Processing participant {participant_id}...")
        
        logits_path = f"{save_paths[participant_id]}logits_{partition}.pkl"
        alignments_save_path = f"{save_paths[participant_id]}word_alignments_{partition}.pkl"
        
        print(f"Loading pre-computed logits from {logits_path}")
        with open(logits_path, 'rb') as handle:
            logits_data_by_day = pickle.load(handle)
            
        word_spans_dict = {}

        for dayIdx, daily_data in tqdm(logits_data_by_day.items(), desc=f"P{participant_id} Alignments"):
            word_spans_dict[dayIdx] = []
            
            for sample_data in daily_data:
                log_probs = sample_data['log_probs'].unsqueeze(0)
                targets = sample_data['targets'].unsqueeze(0)
                transcript = sample_data['transcript']
                
                fa_labels, fa_probs = forced_align(
                    log_probs=log_probs, 
                    targets=targets, 
                    blank=blank_id
                )
                
                words = transcript.split(' ')
                transcript_list = [item for word in words for item in (word, 'SIL')]
                
                word_spans = obtain_word_level_timespans(
                    fa_labels[0], 
                    fa_probs[0], 
                    targets.numpy().squeeze(),
                    transcript=transcript_list, 
                    silence_token_id=silence_token_id
                )
                
                word_spans_dict[dayIdx].append(word_spans)

        print(f"Saving word alignments for participant {participant_id} to {alignments_save_path}")
        with open(alignments_save_path, 'wb') as handle:
            pickle.dump(word_spans_dict, handle)
            
    print("--- Finished: Forced alignment complete. ---")