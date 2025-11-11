# File: generate_logits.py
# Purpose: Run the model inference and save the resulting logits to a file.
from brainaudio.inference.inference_utils import load_model, generate_and_save_logits
import os

# --- Configuration ---
MODEL_NAME = "tm_transformer_combined_chunking_seed_0"
LOAD_MODEL_FOLDER = f"/data2/brain2text/b2t_combined/outputs/{MODEL_NAME}"  
DEVICE = "cuda:1"   
DATASET_PATHS = ["/data2/brain2text/b2t_25/brain2text25.pkl"]
SAVE_PATHS = {0:'/data2/brain2text/b2t_25/logits/'}
PARTITION = 'val'
PARTICIPANT_IDS = [0]
char_label = False


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