
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
#from ctc_forced_aligner.alignment_utils import forced_align
from torch.nn.functional import log_softmax

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


def generate_log_probs(model, args, partition, device, run_forced_alignment=False):
    
    datasetPaths = ['/data2/brain2text/b2t_25/brain2text25.pkl', 
                    '/data2/brain2text/b2t_24/brain2text24']
    
    trainLoaders, valLoaders, loadedData = getDatasetLoaders(
        datasetPaths,
        args["batchSize"]
    )
    
    if partition == 'train':
        
        dataLoaders = trainLoaders
    
    elif partition == "val":
        
        dataLoaders = valLoaders
        
    
    model.eval()
    
    with torch.no_grad():
        
        # loop through data for each participant 
        for participant_id, dataLoader in enumerate(dataLoaders):
        
            for batch in tqdm(dataLoader):
                
                X, y, X_len, y_len, testDayIdx = batch

                # Move data to the specified device
                X = X.to(device)
                y = y.to(device)
                X_len = X_len.to(device)
                y_len = y_len.to(device)
                testDayIdx = testDayIdx.to(device)
                
                # replace padding value with -1 so it doesn't get interpreted as blank token
                y[y==0] = -1
                        
                X = gauss_smooth(X, device=device, smooth_kernel_size=args['smooth_kernel_size'], smooth_kernel_std=args['gaussianSmoothWidth'])
                adjusted_lens = model.compute_length(X_len)
                
                # B x T x C 
                logits = model.forward(X, X_len, participant_id, testDayIdx)
                
                if run_forced_alignment:
                    
                    log_probs = log_softmax(logits)
                    
                    # B x T         
                    for i in range(args['batchSize']):        
                           
                        fa_labels, fa_probs = forced_align(
                            log_probs=log_probs[i:i+1, :adjusted_lens[i]], 
                            targets=y[i:i+1, :y_len[i]], 
                            blank=0
                        )
                    
            breakpoint()    
                    
load_model_folder = "/data2/brain2text/b2t_combined/outputs_tm_transformer_b2t_24+25_large_wide/"  
device = "cuda:0"    

model, args = load_model(load_model_folder, device)
generate_log_probs(model, args, partition="train", device=device, run_forced_alignment=False)
                

        
                    
                
            
            
    
        
    
    
    