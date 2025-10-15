# File: generate_logits.py
# Purpose: Run the model inference and save the resulting logits to a file.
# Environment: .wfst
from brainaudio.inference.inference_utils import load_model, generate_and_save_logits
import os

# --- Configuration ---
MODEL_NAME = "transformer_short_training_fixed_seed_1"
LOAD_MODEL_FOLDER = f"/data2/brain2text/b2t_24/outputs/updated_transformer/{MODEL_NAME}"  
DEVICE = "cuda:2"   
DATASET_PATHS = ['/data2/brain2text/b2t_24/brain2text24_log.pkl']
SAVE_PATHS = ['/data2/brain2text/b2t_24/logits/']
PARTITION = 'val'

CUSTOM_ARGS_PATH = "../src/brainaudio/training/utils/custom_configs/time_masked_transformer.yaml"

def main():
    
    model, args = load_model(LOAD_MODEL_FOLDER, CUSTOM_ARGS_PATH, DEVICE)
    
    for idx, _ in enumerate(SAVE_PATHS):
        SAVE_PATHS[idx] = f'{SAVE_PATHS[idx]}{MODEL_NAME}'
        os.makedirs(SAVE_PATHS[idx], exist_ok=True)

    generate_and_save_logits(
        model=model,
        args=args,
        partition=PARTITION,
        device=DEVICE,
        dataset_paths=DATASET_PATHS,
        save_paths=SAVE_PATHS
    )
    print("--- Logit generation complete. ---")

if __name__ == "__main__":
    main()