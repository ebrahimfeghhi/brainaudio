# File: compute_alignments.py
# Purpose: Load pre-computed logits and generate forced alignments.
# Environment: brainaudio_env (or any env with the necessary alignment tools)

from brainaudio.inference.inference_utils import compute_forced_alignments

# --- Configuration ---
SAVE_PATHS = ['/data2/brain2text/b2t_25/', '/data2/brain2text/b2t_24/']
PARTICIPANT_IDS = [0, 1]
PARTITION = 'val'

def main():
    print("--- Step 2: Computing forced alignments from saved logits ---")
    compute_forced_alignments(
        partition=PARTITION,
        save_paths=SAVE_PATHS,
        participant_ids=PARTICIPANT_IDS
    )
    print("--- Alignment complete. ---")

if __name__ == "__main__":
    main()