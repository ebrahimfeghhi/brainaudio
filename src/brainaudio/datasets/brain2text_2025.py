"""
Run this file to download data from Dryad and unzip the zip files. Downloaded files end
up in this repostitory's data/ directory.

The data will be formatted into the uniform format (compatible with previous dataset, e.g. F.Willet's b2txt24)
"""

import sys
import os
import urllib.request
import json
import zipfile
import numpy as np
import h5py
import pickle
import shutil
from pathlib import Path

DRYAD_DOI = "10.5061/dryad.dncjsxm85"
# DATA_DIR = "/data3/brain2text/b2t_25/og_data"
DATA_DIR = "/home3/lionehlhu/nejm-brain-to-text/data"
OUT_DIR = "/data3/brain2text/b2t_25"
DRYAD_ROOT = "https://datadryad.org"
SESSIONS = ['t15.2023.08.11', 't15.2023.08.13', 't15.2023.08.18', 't15.2023.08.20', 't15.2023.08.25', 't15.2023.08.27', 
            't15.2023.09.01', 't15.2023.09.03', 't15.2023.09.24', 't15.2023.09.29', 't15.2023.10.01', 't15.2023.10.06',
            't15.2023.10.08', 't15.2023.10.13', 't15.2023.10.15', 't15.2023.10.20', 't15.2023.10.22', 't15.2023.11.03',
            't15.2023.11.04', 't15.2023.11.17', 't15.2023.11.19', 't15.2023.11.26', 't15.2023.12.03', 't15.2023.12.08', 
            't15.2023.12.10', 't15.2023.12.17', 't15.2023.12.29', 't15.2024.02.25', 't15.2024.03.03', 't15.2024.03.08',
            't15.2024.03.15', 't15.2024.03.17', 't15.2024.04.25', 't15.2024.04.28', 't15.2024.05.10', 't15.2024.06.14',
            't15.2024.07.19', 't15.2024.07.21', 't15.2024.07.28', 't15.2025.01.10', 't15.2025.01.12', 't15.2025.03.14',
            't15.2025.03.16', 't15.2025.03.30', 't15.2025.04.13' ]
########################################################################################
#
# Helpers.
#
########################################################################################


def display_progress_bar(block_num, block_size, total_size, message=""):
    """"""
    bytes_downloaded_so_far = block_num * block_size
    MB_downloaded_so_far = bytes_downloaded_so_far / 1e6
    MB_total = total_size / 1e6
    sys.stdout.write(
        f"\r{message}\t\t{MB_downloaded_so_far:.1f} MB / {MB_total:.1f} MB"
    )
    sys.stdout.flush()

def load_h5py_file(file_path):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
        'corpus': [],
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
    return data

def download_dataset():
    data_dirpath = os.path.abspath(DATA_DIR)
    
    # Create the data directory if it doesn't exist
    os.makedirs(data_dirpath, exist_ok=True)
    print(f"Data directory: {data_dirpath}")

    ## Get the list of files from the latest version on Dryad.
    urlified_doi = DRYAD_DOI.replace("/", "%2F")

    versions_url = f"{DRYAD_ROOT}/api/v2/datasets/doi:{urlified_doi}/versions"
    with urllib.request.urlopen(versions_url) as response:
        versions_info = json.loads(response.read().decode())

    files_url_path = versions_info["_embedded"]["stash:versions"][-1]["_links"][
        "stash:files"
    ]["href"]
    files_url = f"{DRYAD_ROOT}{files_url_path}"
    with urllib.request.urlopen(files_url) as response:
        files_info = json.loads(response.read().decode())

    file_infos = files_info["_embedded"]["stash:files"]

    ## Download each file into the data directory (and unzip for certain files).
    for file_info in file_infos:
        filename = file_info["path"]

        if filename == "README.md":
            continue

        download_path = file_info["_links"]["stash:download"]["href"]
        download_url = f"{DRYAD_ROOT}{download_path}"

        download_to_filepath = os.path.join(data_dirpath, filename)

        urllib.request.urlretrieve(
            download_url,
            download_to_filepath,
            reporthook=lambda *args: display_progress_bar(
                *args, message=f"Downloading {filename}"
            ),
        )
        sys.stdout.write("\n")

        # If this file is a zip file, unzip it into the data directory.

        if file_info["mimeType"] == "application/zip":
            print(f"Extracting files from {filename} ...")
            with zipfile.ZipFile(download_to_filepath, "r") as zf:
                zf.extractall(data_dirpath)

    print(f"\nDownload complete. See data files in {data_dirpath}\n")


def cleanup_original_data_dir():
    """
    Remove the entire original data directory after reformatting is complete.
    This is safe since DATA_DIR and OUT_DIR are now separate.
    """
    data_path = Path(DATA_DIR)
    
    if not data_path.exists():
        print(f"Original data directory {DATA_DIR} does not exist, nothing to clean up.")
        return
    
    # Calculate total size before removal
    try:
        total_size = sum(f.stat().st_size for f in data_path.rglob('*') if f.is_file())
        size_gb = total_size / 1e9
        
        print(f"\n=== Cleanup Original Data Directory ===")
        print(f"Directory to remove: {DATA_DIR}")
        print(f"Total size: {size_gb:.2f} GB")
        
        # Ask for confirmation
        response = input(f"\nRemove entire directory {DATA_DIR}? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Cleanup cancelled.")
            return
        
        # Remove the entire directory
        shutil.rmtree(data_path)
        print(f"✓ Successfully removed {DATA_DIR}")
        print(f"✓ Freed {size_gb:.2f} GB of disk space")
        
    except Exception as e:
        print(f"✗ Failed to remove {DATA_DIR}: {e}")


########################################################################################
#
# Main function.
#
########################################################################################

def main():
    # download_dataset()

    # reformat 
    brain2text_2025 = {}
    brain2text_2025['train'] = []
    brain2text_2025['val'] = []
    brain2text_2025['test'] = []

    session_dir = os.path.join(DATA_DIR, 'hdf5_data_final')
    for session in SESSIONS:
        session_path = os.path.join(session_dir, session)
        
        # Check if session directory exists
        if not os.path.exists(session_path):
            print(f"Warning: Session directory not found: {session_path}")
            continue
            
        files = [f for f in os.listdir(session_path) if f.endswith('.hdf5')]
        print(f"Session {session}: Found files {files}")
        
        if f'data_train.hdf5' in files:
            train_file = os.path.join(session_path,f'data_train.hdf5')
            train_data = load_h5py_file(train_file)
            sesh = {}
            sesh['sentenceDat'] = train_data['neural_features']
            sesh['transcriptions'] = train_data['sentence_label']
            sesh['text'] = train_data['seq_class_ids']
            sesh['timeSeriesLen'] = train_data['n_time_steps']
            sesh['textLens'] = train_data['seq_len']
            #sesh['phonePerTime'] = [p.astype(np.float32) / n.astype(np.float32) for (p, n) in zip(sesh['textLen'],sesh['timeSeriesLen'])]
            brain2text_2025['train'].append(sesh)

        if f'data_val.hdf5' in files:
            val_file = os.path.join(session_path, f'data_val.hdf5')
            val_data = load_h5py_file(val_file)
            sesh = {}
            sesh['sentenceDat'] = val_data['neural_features']
            sesh['transcriptions'] = val_data['sentence_label']
            sesh['text'] = val_data['seq_class_ids']
            sesh['timeSeriesLen'] = val_data['n_time_steps']
            sesh['textLens'] = val_data['seq_len']
            #sesh['phonePerTime'] = [p.astype(np.float32) / n.astype(np.float32) for (p, n) in zip(sesh['textLen'],sesh['timeSeriesLen'])]
            brain2text_2025['val'].append(sesh)
        else:
            brain2text_2025['val'].append(None)
    
        if f'data_test.hdf5' in files:
            test_file = os.path.join(session_path, f'data_test.hdf5') 
            test_data = load_h5py_file(test_file)
            sesh = {}
            sesh['sentenceDat'] = test_data['neural_features']
            sesh['transcriptions'] = None
            sesh['text'] = None
            sesh['timeSeriesLen'] = test_data['n_time_steps']
            sesh['textLens'] = None
            #sesh['phonePerTime'] = None
            brain2text_2025['test'].append(sesh)
        else:
            brain2text_2025['test'].append(None)

    # Save the reformatted data to the output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    output_file = os.path.join(OUT_DIR, 'brain2text25_log.pkl')
    
    with open(output_file, 'wb') as handle:
        pickle.dump(brain2text_2025, handle)
    
    print(f"\n✓ Data reformatting complete! Saved to: {output_file}")
    print(f"✓ Reformatted data contains:")
    print(f"  - Train sessions: {len(brain2text_2025['train'])}")
    print(f"  - Validation sessions: {len(brain2text_2025['val'])}")
    print(f"  - Test sessions: {len(brain2text_2025['test'])}")
    
    # Clean up the entire original data directory
    # cleanup_original_data_dir()


if __name__ == "__main__":
    main()
