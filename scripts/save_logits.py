# File: generate_logits.py
# Purpose: Run the model inference and save the resulting logits to a file.
# Environment: .wfst
from brainaudio.inference.inference_utils import load_model, generate_and_save_logits
import os

# --- Configuration ---
MODEL_NAME = "tm_transformer_combined_lw_char"
LOAD_MODEL_FOLDER = f"/data2/brain2text/b2t_combined/outputs/{MODEL_NAME}"  
DEVICE = "cuda:2"   
DATASET_PATHS = ['/data2/brain2text/b2t_24/brain2text24_with_fa_char']
SAVE_PATHS = {1:'/data2/brain2text/b2t_24/logits/'}
PARTITION = 'val'
PARTICIPANT_IDS = [1]
char_label = True


def main():
    
    model, args = load_model(LOAD_MODEL_FOLDER, DEVICE)
    
    for idx, value in SAVE_PATHS.items():
        SAVE_PATHS[idx] = f'{SAVE_PATHS[idx]}{MODEL_NAME}'
        os.makedirs(SAVE_PATHS[idx], exist_ok=True)

    generate_and_save_logits(
        model=model,
        config=args,
        partition=PARTITION,
        device=DEVICE,
        dataset_paths=DATASET_PATHS,
        save_paths=SAVE_PATHS, 
        participant_ids=PARTICIPANT_IDS, 
        char_label=char_label
    )
    print("--- Logit generation complete. ---")

if __name__ == "__main__":
    main()