# File: generate_logits.py
# Purpose: Run the model inference and save the resulting logits to a file.
# Environment: .wfst
from brainaudio.inference.inference_utils import save_transcripts
import os

# --- Configuration ---
DATASET_PATHS = ['/data2/brain2text/b2t_25/brain2text25.pkl', '/data2/brain2text/b2t_24/brain2text24.pkl']
SAVE_PATHS = ['/data2/brain2text/b2t_25/', '/data2/brain2text/b2t_24/']
PARTICIPANT_IDS = [0, 1]
PARTITION = 'val'

def main():
    
    save_transcripts(
        partition=PARTITION,
        dataset_paths=DATASET_PATHS,
        save_paths=SAVE_PATHS
    )

if __name__ == "__main__":
    main()