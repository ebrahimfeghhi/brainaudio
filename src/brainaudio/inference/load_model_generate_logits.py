
import torch
import pickle
import os
from typing import Dict, Any, List, Tuple
import numpy as np
import re
from tqdm import tqdm
from itertools import zip_longest
from torch.nn.functional import log_softmax

import pickle
from typing import Optional
import yaml
import re

from brainaudio.models.transformer_chunking import TransformerModel
from brainaudio.datasets.lazy_data_loading import getDatasetLoaders
from brainaudio.training.utils.augmentations import gauss_smooth
from brainaudio.inference.eval_metrics import compute_per

def load_transformer_model(
    folder: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    modelWeightsFile: Optional[str] = "modelWeights",
    eval_chunk_config: Optional[Dict[str, Optional[int]]] = None,
):
    """
    Load a pre-trained Transformer model from a folder.

    config:
        folder (str): Path to folder containing 'modelWeights'.
        device (torch.device): Device to map the model onto.
        modelWeightsFile (str|None): Filename of the model weights.
        eval_chunk_config (dict|None): Optional chunking overrides for evaluation.

    Returns:
        torch.nn.Module: The loaded model in eval mode, and the config dictionary.
    """
    
    # --- MODIFIED SECTION ---
    # Determine the path for the config file

    config_path = os.path.join(folder, "args")
    print(f"Loading default pickle config from: {config_path}")
    with open(config_path, "rb") as handle:
        config = pickle.load(handle)
    # ----------------------
        
    # Load config                
    modelType = config['modelType']
    model_config = config['model'][modelType]
    

    if 'return_final_layer' not in config:
        config['return_final_layer'] = False

    chunked_attention_cfg = model_config.get('chunked_attention')
    if eval_chunk_config is not None and isinstance(chunked_attention_cfg, dict):
        chunked_attention_cfg['eval'] = eval_chunk_config
        print(f"Overriding eval chunk config with: {eval_chunk_config}")

    model = TransformerModel(
        features_list=model_config['features_list'],
        samples_per_patch=model_config['samples_per_patch'],
        dim=model_config['d_model'],
        depth=model_config['depth'],
        heads=model_config['n_heads'],
        mlp_dim_ratio=model_config['mlp_dim_ratio'],
        dim_head=model_config['dim_head'],
        dropout=config['dropout'],
        input_dropout=config['input_dropout'],
        nClasses=config['nClasses'],
        max_mask_pct=config['max_mask_pct'],
        num_masks=config['num_masks'],
        num_participants=len(model_config['features_list']),
        return_final_layer=config['return_final_layer'],
        chunked_attention=chunked_attention_cfg,
    )

    # Load weights
    ckpt_path = os.path.join(folder, modelWeightsFile)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    return model, config
            
       
def generate_and_save_logits(model, config, partition, device, 
                             manifest_paths, save_paths, participant_ids,
                             chunk_config: Optional[Dict[str, Optional[int]]] = None):
    
    """
    Runs the model forward pass and saves the output logits to a file.

    config:
        model (torch.nn.Module): The neural network model.
        config (dict): Dictionary of arguments.
        partition (str): The data partition to process ('train' or 'val').
        device (torch.device): The device to run the model on.
        manifest_paths (list): List of manifest.json paths per participant.
        save_paths (list): List of paths to save the output files.
        participant_ids (list): List of integer participant ids.
        chunk_config (dict|None): Optional chunking overrides for evaluation.
    """
    
    print(f"--- Starting: Generating logits for '{partition}' partition ---")

    def _format_chunk_tag(cfg: Optional[Dict[str, Optional[int]]]) -> Optional[str]:
        if not cfg:
            return None
        chunk = cfg.get("chunk_size")
        context = cfg.get("context_sec")
        chunk_str = "full" if chunk is None else str(chunk)
        context_str = "full" if context is None else str(context)
        return f"chunk:{chunk_str}_context:{context_str}"

    chunk_tag = _format_chunk_tag(chunk_config)
    if chunk_tag:
        print(f"Using eval chunk config: {chunk_config}")

    trainLoaders, valLoaders, testLoaders = getDatasetLoaders(
        manifest_paths,
        config["batchSize"], 
        return_transcript=True, 
        shuffle_train=False
    )
    
    if partition == "train":
        dataLoaders = trainLoaders
    elif partition == "val":
        dataLoaders = valLoaders
    elif partition == "test":
        dataLoaders = testLoaders
        
    model.eval()
    with torch.no_grad():
        for dataLoader, participant_id in zip(dataLoaders, participant_ids):
            print(f"Processing participant {participant_id}...")
            
            logits_data = []
            transcriptions = []
            total_edit_distance = 0
            total_seq_length = 0
        
            for batch in tqdm(dataLoader, desc=f"P{participant_id} Logits"):
                
                X, y, X_len, y_len, dayIdxs, transcripts = batch

                X, y, X_len, y_len, dayIdxs = (
                    X.to(device), y.to(device), X_len.to(device), 
                    y_len.to(device), dayIdxs.to(device)
                )
                
                X = gauss_smooth(X, device=device, smooth_kernel_size=config['smooth_kernel_size'], 
                                 smooth_kernel_std=config['gaussianSmoothWidth'])
                
                adjusted_lens = model.compute_length(X_len)
                logits = model.forward(X, X_len, participant_id, dayIdxs)
                
                for i in range(logits.shape[0]):            
                    
                    ali = adjusted_lens[i]
                    yli = y_len[i]
                    
                    logits_data.append(logits[i, :ali].cpu().numpy())
                    transcriptions.append(transcripts[i])
                
                    total_edit_distance, total_seq_length = compute_per(logits[i, :ali].cpu(), 
                    y[i, :yli].cpu(), total_edit_distance, total_seq_length)
            
            if chunk_tag:
                save_path = f"{save_paths[participant_id]}/logits_{partition}_{chunk_tag}.npz"
            else:
                save_path = f"{save_paths[participant_id]}/logits_{partition}.npz"
                
            print(f"Saving logits for participant {participant_id} to {save_path}")
            np.savez_compressed(save_path, *logits_data)
            print(f"Error Rate for participant {participant_id}: {total_edit_distance/total_seq_length}")
                
            
    print("--- Finished: Logit generation complete. ---")

def save_transcripts(manifest_paths, partition, save_paths):
    '''
    Saves the transcripts for each participant in the dataset.

    Args:
        manifest_paths (list): List of manifest.json paths for each participant.
        partition (str): The data partition to process ('train' or 'val').
        save_paths (list): List of paths to save the transcripts.
    '''
    trainLoaders, valLoaders, _ = getDatasetLoaders(
        manifest_paths,
        1, 
        return_transcript=True, 
        shuffle_train=False
    )
    
    dataLoaders = trainLoaders if partition == 'train' else valLoaders
    
    for participant_id, dataLoader in enumerate(dataLoaders):
        
        transcriptions = []
        
        for batch in tqdm(dataLoader, desc=f"P{participant_id} Transcripts"):
            
            X, y, X_len, y_len, dayIdxs, transcripts = batch
            
            #transcript_trial = transcripts[0].replace(".", "").lower()
            
            transcriptions.append(transcripts)
            
        save_path = f"{save_paths[participant_id]}/transcripts_{partition}.pkl"
        with open(save_path, 'wb') as handle:
            pickle.dump(transcriptions, handle)
            
 
