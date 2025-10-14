# File: generate_logits.py
# Purpose: Run the model inference and save the resulting logits to a file.
# Environment: .wfst
from brainaudio.inference.inference_utils import save_transcripts
import os
from brainaudio.utils.config import B2T_DATASET_PATHS, B2T_SAVE_PATHS

# --- Configuration ---
DATASET_PATHS = B2T_DATASET_PATHS
SAVE_PATHS = B2T_SAVE_PATHS
PARTICIPANT_IDS = [0, 1]
PARTITION = 'val'

def main():
    
    save_transcripts(
        partition=PARTITION,
        dataset_paths=DATASET_PATHS,
        save_paths=SAVE_PATHS,
        participant_ids=PARTICIPANT_IDS
    )

if __name__ == "__main__":
    main()