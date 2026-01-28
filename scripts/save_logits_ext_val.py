import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from brainaudio.inference.load_model_generate_logits import load_transformer_model
from save_logits import _format_eval_tag
# Import the MemoryDataset we just created (or paste the class definition here)
from brainaudio.datasets.memory_dataset import MemorySpeechDataset 

# --- Configuration ---
MODEL_NAME = "best_chunked_transformer_combined_seed_0"
local_model_folder = "b2t_combined"
modelWeightsFiles = "modelWeights_PER_25"

LOAD_MODEL_FOLDER = f"/data2/brain2text/{local_model_folder}/outputs/{MODEL_NAME}"  
DEVICE = "cuda:1"   


# Eval Config (Chunking)
EVAL_CONFIGS = [{"chunk_size": None, "context_sec": None}]  

if modelWeightsFiles == "modelWeights_PER_25":
    EXTENDED_VAL_PATH = "/data2/brain2text/b2t_25/val_extended_25.pkl"
    SAVE_PATH = {0:'/data2/brain2text/b2t_25/logits/'}
    PARTICIPANT_IDS = [0]
    
    
if modelWeightsFiles == "modelWeights_PER_24":
    EXTENDED_VAL_PATH = "/data2/brain2text/b2t_24/val_extended_24.pkl"
    SAVE_PATH = {0:'/data2/brain2text/b2t_24/logits/'}
    PARTICIPANT_IDS = [0]
    

# --------------------------

def _padding_collate(batch):
    """Standard collate function for speech data."""
    X, y, X_lens, y_lens, days, transcripts = zip(*batch)
    
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    
    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
        transcripts
    )

def run_inference_on_memory_data(model, args, dataset_dict, dataset_name, save_dir):
    """
    Iterates over 2x, 3x, 4x keys in the loaded pickle and runs inference.
    """
    model.eval()
    
    # Identify keys like 'val_2x_b2t25', 'val_3x_b2t25'
    keys = sorted([k for k in dataset_dict.keys() if 'val_' in k])
    
    for key in keys:
        print(f"--- Processing {key} ---")
        
        # 1. Create Adapter Dataset
        raw_data_list = dataset_dict[key]
        dataset = MemorySpeechDataset(raw_data_list)
        
        loader = DataLoader(
            dataset,
            batch_size=1,  # Inference is usually batch size 1 for accuracy
            shuffle=False,
            num_workers=4,
            collate_fn=_padding_collate
        )
        
        # 2. Setup Saving
        # Structure: /logits_extended/ModelName/b2t25/val_2x/
        subset_save_dir = os.path.join(save_dir, dataset_name, key)
        os.makedirs(subset_save_dir, exist_ok=True)
        
        # 3. Inference Loop
        with torch.no_grad():
            for i, batch in enumerate(loader):
                # Unpack
                X, y, X_len, y_len, day_idx, transcript = batch
                
                # Move to device
                X = X.to(DEVICE)
                X_len = X_len.to(DEVICE)
                day_idx = day_idx.to(DEVICE) # Model needs day index for embedding
                
                # Forward Pass
                # Note: 'participant_id' is needed. 
                # If modelWeights_PER_24 -> trained on participant 0 (b2t24)? 
                # If Combined model -> b2t25 is pid 0, b2t24 is pid 1.
                # You must set this correctly based on which dataset you are processing.
                pid = 0 if "b2t25" in dataset_name else 1 
                # ^ ADJUST THIS LOGIC to match your training PID mapping!
                
                logits = model(X, X_len, participant_idx=pid, day_idx=day_idx)
                
                # Save
                logits_np = logits.cpu().numpy()
                
                # Save as .npz (standard format for decoders)
                save_path = os.path.join(subset_save_dir, f"trial_{i}.npz")
                np.savez(save_path, logits=logits_np, transcript=transcript[0])
                
        print(f"Saved {len(dataset)} trials to {subset_save_dir}")

def main():
    resolved_save_path  = f"{SAVE_PATH}{MODEL_NAME}/extended_val"
    os.makedirs(resolved_save_path, exist_ok=True)

    eval_configs = EVAL_CONFIGS or [None]
    print(f"Loading extended data: {EXTENDED_VAL_PATH[19:]}")
    with open(EXTENDED_VAL_PATH, "rb") as f:
        data = pickle.load(f)
    for eval_cfg in eval_configs:
        tag = _format_eval_tag(eval_cfg)
        print(f"=== Running eval config: {tag} ===")

        # 1. Load Model
        print(f"Loading model from {LOAD_MODEL_FOLDER}...")
        model, args = load_transformer_model(
            LOAD_MODEL_FOLDER,
            DEVICE,
            modelWeightsFile=modelWeightsFiles,
            eval_chunk_config=eval_cfg,
        )
        run_inference_on_memory_data(model, args, data, "b2t25", resolved_save_path)

    del data
if __name__ == "__main__":
    main()