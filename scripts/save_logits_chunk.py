# File: generate_logits.py
# Purpose: Run the model inference and save the resulting logits to a file.
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

from brainaudio.models.transformer_chunking import TransformerModel
from brainaudio.datasets.loading_data import getDatasetLoaders
from brainaudio.training.utils.augmentations import gauss_smooth
from brainaudio.inference.inference_utils import compute_per


# --- Configuration ---
MODEL_NAME = "tm_transformer_b2t25_chunking_test_run"
LOAD_MODEL_FOLDER = f"/data2/brain2text/b2t_25/outputs/{MODEL_NAME}"  
DEVICE = "cuda:0"   
#DATASET_PATHS = ['/data2/brain2text/b2t_25/brain2text25_with_fa', "/data2/brain2text/b2t_24/brain2text24_with_fa"]
#SAVE_PATHS = {0:'/data2/brain2text/b2t_25/logits/', 1:'/data2/brain2text/b2t_24/logits/'}
DATASET_PATHS = ["/data2/brain2text/b2t_25/brain2text25.pkl"]
SAVE_PATHS = {0:'/data2/brain2text/b2t_25/logits/'}
PARTITION = 'val'
PARTICIPANT_IDS = [0]
char_label = False

# For evaluation on different chunk configs for same model
CHUNK_SIZES = [1, 5, 10, 30, 50, None]
CONTEXT_SIZES = [None]
CHUNK_CONFIGS = [{"chunk_size": chk, "context_chunks": ctx} for chk in CHUNK_SIZES for ctx in CONTEXT_SIZES]

def main():
    resolved_save_paths = {
        idx: os.path.join(path, MODEL_NAME)
        for idx, path in SAVE_PATHS.items()
    }
    
    for eval_cfg in CHUNK_CONFIGS:
        tag = _format_eval_tag(eval_cfg)
        print(f"=== Evaluating config: {tag} ===")
        model, args = load_model(LOAD_MODEL_FOLDER, eval_cfg, DEVICE)
        

        generate_and_save_logits(
            model=model,
            config=args,
            partition=PARTITION,
            device=DEVICE,
            dataset_paths=DATASET_PATHS,
            save_paths=resolved_save_paths, 
            participant_ids=PARTICIPANT_IDS, 
            char_label=char_label,
            chunk_config=eval_cfg,
        )

    print("--- Logit generation complete. ---")

def _format_eval_tag(cfg):
    chunk = "full" if cfg["chunk_size"] is None else str(cfg["chunk_size"])
    context = "full" if cfg["context_chunks"] is None else str(cfg["context_chunks"])
    return f"chunk_{chunk} + context_{context}"


def load_model(
    folder: str,
    eval_config: None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Load a pre-trained model from a folder.

    config:
        folder (str): Path to folder containing 'modelWeights'.
        eval_config (dict): For chunked evaluation configuration.
        device (torch.device): Device to map the model onto.

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

    
    if modelType == 'transformer':
        # For chunked evaluation
        if eval_config is not None and model_config.get("chunked_attention", None) is not None:
            model_config["chunked_attention"]["eval"] = eval_config
        
        if 'return_final_layer' not in config:
            config['return_final_layer'] = False
        
        model = TransformerModel(features_list=model_config['features_list'], samples_per_patch=model_config['samples_per_patch'], dim=model_config['d_model'], depth=model_config['depth'], 
                        heads=model_config['n_heads'], mlp_dim_ratio=model_config['mlp_dim_ratio'],  dim_head=model_config['dim_head'], 
                        dropout=config['dropout'], input_dropout=config['input_dropout'], nClasses=config['nClasses'], 
                        max_mask_pct=config['max_mask_pct'], num_masks=config['num_masks'], num_participants=len(model_config['features_list']), return_final_layer=config['return_final_layer'],
                        chunked_attention=model_config.get('chunked_attention', None))


    # Load weights
    ckpt_path = os.path.join(folder, "modelWeights")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    return model, config

def generate_and_save_logits(model, config, partition, device, 
                             dataset_paths, save_paths, participant_ids, char_label, chunk_config=None):
    
    """
    Runs the model forward pass and saves the output logits to a file.

    config:
        model (torch.nn.Module): The neural network model.
        config (dict): Dictionary of arguments.
        partition (str): The data partition to process ('train' or 'val').
        device (torch.device): The device to run the model on.
        dataset_paths (list): List of paths to the dataset files.
        save_paths (list): List of paths to save the output files.
        participant_ids (list): List of integer participant ids.
        char_label (bool): use characters if True, else phonemes.
        chunk_config (dict) Dictionary that records the chunk configuration used in evaluation.
    """
    
    print(f"--- Starting: Generating logits for '{partition}' partition ---")
    chunk_size = None
    context_chunks = None
    if chunk_config is not None:
        chunk_size = chunk_config["chunk_size"]
        context_chunks = chunk_config["context_chunks"]

    trainLoaders, valLoaders, testLoaders, _ = getDatasetLoaders(
        dataset_paths,
        config["batchSize"], 
        return_transcript=True, 
        shuffle_train=False, 
        char_label=char_label
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
            
            if chunk_config is not None:
                save_path = f"{save_paths[participant_id]}/logits_{partition}_{chunk_size}_{context_chunks}.npz"
            else:
                save_path = f"{save_paths[participant_id]}/logits_{partition}.npz"
            print(f"Saving logits for participant {participant_id} to {save_path}")
            np.savez_compressed(save_path, *logits_data)
            print(f"Error Rate for participant {participant_id}: {total_edit_distance/total_seq_length}")
                
            
    print("--- Finished: Logit generation complete. ---")

if __name__ == "__main__":
    main()
