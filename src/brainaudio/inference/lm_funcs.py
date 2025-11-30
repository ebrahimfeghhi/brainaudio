import re
import numpy as np
import torch
import os
import pickle
from typing import List, Dict, Optional, Tuple
from brainaudio.inference.eval_metrics import _cer_and_wer


# --- The dictionary of informal shorthand words ---
# Does not include "a" or "I" as they are standard English.
SHORTHAND_MAP = {
    'b': 'be',
    'c': 'see',
    'r': 'are',
    'u': 'you',
    'y': 'why'
}

def normalize_shorthand(text: str) -> str:
    """
    Converts informal single-character shorthand (like 'u', 'r', 'c')
    in a string to their full-word equivalents.
    ASSUMES INPUT TEXT IS ALREADY LOWERCASE.

    Args:
        text: The input string (assumed to be lowercase).

    Returns:
        The modified string with shorthand words replaced.
    """
    
    modified_text = text
    
    for shorthand, full_word in SHORTHAND_MAP.items():
        
        # This is the regex pattern to find the whole word.
        # \b = word boundary (matches start/end of a word)
        # re.escape(shorthand) = the letter itself (e.g., 'c')
        # No re.IGNORECASE needed as we assume lowercase input.
        pattern = r'\b' + re.escape(shorthand) + r'\b'

        # Simplified replacement: just use the lowercase full word.
        modified_text = re.sub(
            pattern, 
            full_word, 
            modified_text
        )

    return modified_text

def clean_string(transcript):
    
    transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
    transcript = transcript.replace("--", "").lower()
    
    return transcript


def compute_wer_from_logits(
    logits_paths: List[str],
    dataset_ids: List[int],
    decoder,
    acoustic_scale: float = 0.6,
    verbose: bool = True,
    transcripts_base_dir: str = "/data2/brain2text",
    save_results: bool = True,
    output_filename: str = "wer_results.pkl"
) -> Tuple[Dict[str, float], List[float], Dict[str, List[str]]]:
    """
    Compute WER for multiple models using beam search decoding.
    
    Args:
        logits_paths: List of paths to npz files containing model logits
        dataset_ids: List of dataset identifiers (24 or 25) corresponding to each logits file
        decoder: CTC decoder instance (from torchaudio.models.decoder)
        acoustic_scale: Scaling factor for acoustic scores (default: 0.6)
        verbose: Whether to print progress (default: True)
        transcripts_base_dir: Base directory for transcript files (default: /data2/brain2text)
        save_results: Whether to save results to file (default: True)
        output_filename: Name of output file (default: wer_results.pkl)
    
    Returns:
        Tuple of (wer_dict, wer_list, decoded_sentences) where:
            - wer_dict: Dictionary mapping logits filename (without .npz) to WER value
            - wer_list: List of WER values in same order as logits_paths
            - decoded_sentences: Dict mapping logits filename to list of decoded transcripts
    """
    if len(logits_paths) != len(dataset_ids):
        raise ValueError(f"Number of logits paths ({len(logits_paths)}) must match number of dataset IDs ({len(dataset_ids)})")
    
    wer_results = []
    wer_dict = {}
    decoded_sentences: Dict[str, List[str]] = {}
    
    # Determine output directory from first logits path
    if logits_paths:
        output_dir = os.path.dirname(logits_paths[0])
    else:
        output_dir = "."
    
    for logits_path, dataset_id in zip(logits_paths, dataset_ids):
        if verbose:
            print(f"\nProcessing: {logits_path}")
            print(f"  Dataset: b2t_{dataset_id}")
        
        # Extract filename without extension for dict key
        logits_filename = os.path.basename(logits_path)
        if logits_filename.endswith('.npz'):
            logits_key = logits_filename[:-4]  # Remove .npz
        else:
            logits_key = logits_filename
        
        # Construct transcript path based on dataset ID
        transcripts_path = f"{transcripts_base_dir}/b2t_{dataset_id}/transcripts_val_cleaned.pkl"
        
        # Load data
        model_logits = np.load(logits_path)
        
        with open(transcripts_path, 'rb') as f:
            val_transcripts = pickle.load(f)
        
        # Run beam search on all trials
        ground_truth_arr = []
        pred_arr = []
        
        for idx in range(len(model_logits)):
            if verbose and idx % 100 == 0:
                print(f"  Trial {idx}/{len(model_logits)}")
            
            single_trial_logits = torch.as_tensor(model_logits[f'arr_{idx}']).float().unsqueeze(0)
            beam_search_char = decoder(single_trial_logits * acoustic_scale)
            beam_search_transcript_char = normalize_shorthand(" ".join(beam_search_char[0][0].words).strip())
            ground_truth_sentence = val_transcripts[idx]
            ground_truth_arr.append(ground_truth_sentence)
            pred_arr.append(beam_search_transcript_char)
        
        # Compute WER
        cer, wer, wer_sent = _cer_and_wer(pred_arr, ground_truth_arr)
        wer_results.append(wer)
        wer_dict[logits_key] = wer
        decoded_sentences[logits_key] = pred_arr
        
        if verbose:
            print(f"  WER: {wer:.4f}")
    
    # Save results if requested
    if save_results and wer_dict:
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'wb') as f:
            pickle.dump(wer_dict, f)
        if verbose:
            print(f"\nSaved WER results to: {output_path}")
    
    return wer_dict, wer_results, decoded_sentences