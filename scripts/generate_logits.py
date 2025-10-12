# File: generate_logits.py
# Purpose: Run the model inference and save the resulting logits to a file.
# Environment: .wfst
from brainaudio.inference.inference_utils import load_model, generate_and_save_logits
import os

# --- Configuration ---
LOAD_MODEL_FOLDER = "/data2/brain2text/b2t_combined/outputs_tm_transformer_b2t_24+25_large_wide/"  
DEVICE = "cuda:0"   
DATASET_PATHS = ['/data2/brain2text/b2t_25/brain2text25.pkl', '/data2/brain2text/b2t_24/brain2text24']
SAVE_PATHS = ['/data2/brain2text/b2t_25/', '/data2/brain2text/b2t_24/']
PARTICIPANT_IDS = [0, 1]
PARTITION = 'val'

def main():
    print("--- Step 1: Loading model and generating logits ---")
    model, args = load_model(LOAD_MODEL_FOLDER, DEVICE)
    
    for idx, save_path in enumerate(SAVE_PATHS):
        SAVE_PATHS[idx] = f'{SAVE_PATHS[idx]}{args["modelName"]}'
        os.makedirs(SAVE_PATHS[idx], exist_ok=True)

    generate_and_save_logits(
        model=model,
        args=args,
        partition=PARTITION,
        device=DEVICE,
        dataset_paths=DATASET_PATHS,
        save_paths=SAVE_PATHS,
        participant_ids=PARTICIPANT_IDS
    )
    print("--- Logit generation complete. ---")

if __name__ == "__main__":
    main()