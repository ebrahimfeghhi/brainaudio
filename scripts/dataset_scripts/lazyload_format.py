import json
import os
import numpy as np
import pickle
from pathlib import Path

# --- Configuration ---
data_paths = ['/data2/brain2text/b2t_25/brain2text25.pkl', '/data2/brain2text/b2t_24/brain2text24.pkl']
output_dirs = ['/data2/brain2text/b2t_25/trial_level_data/', '/data2/brain2text/b2t_24/trial_level_data/']
# ---------------------

print("Starting preprocessing...")


for p_id, pkl_path in enumerate(data_paths):

    file_manifest = {'train': [], 'val': [], 'test': []} # To save paths

    print(f"Processing participant {p_id} from {pkl_path}...")

    output_dir = output_dirs[p_id]

    with open(pkl_path, "rb") as handle:
        participant_data = pickle.load(handle)

    for split in ['train', 'val', 'test']:

        print(f"Processing split: {split}")

        if split not in participant_data:
            continue

        split_data = participant_data[split]

        is_test = split == "test"

        for day in range(len(split_data)):

            if split_data[day] is None:
                continue

            # Use 'sentenceDat' to find the number of trials
            n_trials = len(split_data[day]["sentenceDat"])

            for trial in range(n_trials):
                # Each trial gets its own directory with individual .npy files
                trial_dir = f"{output_dir}/{split}/day_{day}_trial_{trial}"
                os.makedirs(trial_dir, exist_ok=True)

                # --- Get the data for this trial ---
                sentenceDat = split_data[day]["sentenceDat"][trial]

                transcript = (split_data[day]['transcriptions'][trial]
                              if not is_test else "FILLER")

                if not is_test:
                    text_len = split_data[day]["textLens"][trial]
                    padded_text = split_data[day]["text"][trial]
                    trimmed_text = padded_text[:text_len]
                else:
                    trimmed_text = np.array([0], dtype=np.int32)

                # Save arrays as individual .npy files (no zip layer)
                np.save(f"{trial_dir}/sentenceDat.npy", sentenceDat)
                np.save(f"{trial_dir}/text.npy", trimmed_text)

                # Save metadata as plain JSON
                with open(f"{trial_dir}/meta.json", "w") as f:
                    json.dump({"day": int(day), "pid": int(p_id), "transcription": transcript}, f)

                # Manifest now points to trial directories
                file_manifest[split].append(str(trial_dir))

    print("Done preprocessing.")

    with open(f"{output_dir}/manifest.json", "w") as f:
        json.dump(file_manifest, f)