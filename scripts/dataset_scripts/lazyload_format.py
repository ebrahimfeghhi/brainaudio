import pandas as pd
import os
import numpy as np
import pickle
from pathlib import Path

# --- Configuration ---
data_paths = ['/data2/brain2text/b2t_25/brain2text25.pkl', '/data2/brain2text/b2t_24/brain2text24.pkl']
output_dirs = ['/data2/brain2text/b2t_25/trial_level_data/', '/data2/brain2text/b2t_24/trial_level_data/']
# ---------------------

print("Starting preprocessing...")
file_manifest = {'train': [], 'val': [], 'test': []} # To save paths

for p_id, pkl_path in enumerate(data_paths):
    print(f"Processing participant {p_id} from {pkl_path}...")
    
    output_dir = output_dirs[p_id]
    
    with open(pkl_path, "rb") as handle:
        participant_data = pickle.load(handle)

    for split in ['train', 'val', 'test']:
        if split not in participant_data:
            continue
            
        split_data = participant_data[split]
        
        if split == "test":
            is_test = True
        else:
            is_test = False
        
        for day in range(len(split_data)):
            if split_data[day] is None:
                continue
            
            # Use 'sentenceDat' to find the number of trials
            n_trials = len(split_data[day]["sentenceDat"])
            
            for trial in range(n_trials):
                # Define where to save this single trial
                trial_dir = f"{output_dir}/{split}"
                os.makedirs(trial_dir, exist_ok=True)
                trial_path = f"{trial_dir}/day_{day}_trial_{trial}.npz"
                
                # --- Get the data for this trial ---
                sentenceDat = split_data[day]["sentenceDat"][trial]
                                
                transcript = (split_data[day]['transcriptions'][trial] 
                              if not is_test else "FILLER")
                
                text = (split_data[day]["text"][trial] 
                        if not is_test else np.array([0], dtype=np.int32))

                # Save all trial data to a single compressed .npz file
                # We also save 'pid', 'day', and 'trial' as metadata
                np.savez_compressed(
                    trial_path,
                    sentenceDat=sentenceDat,
                    transcription=np.array(transcript, dtype=object), # Save string as 0-D object array
                    text=text,
                    pid=p_id,
                    day=day
                )
                
                # Add the new file path to our manifest
                file_manifest[split].append(str(trial_path))

print("Done preprocessing.")

# Optionally, save the manifest of all file paths for easy loading
with open(output_dir / "manifest.json", "w") as f:
    import json
    json.dump(file_manifest, f)