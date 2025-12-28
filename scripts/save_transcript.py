# File: generate_logits.py
# Purpose: Run the model inference and save the resulting logits to a file.
# Environment: .wfst
from brainaudio.inference.load_model_generate_logits import save_transcripts
import os

# --- Configuration ---
MANIFEST_PATHS = ['/data2/brain2text/b2t_25/trial_level_data/manifest.json']
SAVE_PATHS = ['/data2/brain2text/b2t_25/']
PARTITION = 'train'

def main():
    
    save_transcripts(
        partition=PARTITION,
        manifest_paths=MANIFEST_PATHS,
        save_paths=SAVE_PATHS
    )

if __name__ == "__main__":
    main()