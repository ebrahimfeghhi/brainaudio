
import torch
import pickle
import os
from typing import Dict, Any, List, Tuple
import numpy as np
from edit_distance import SequenceMatcher
import re
from tqdm import tqdm
from itertools import zip_longest
from torchaudio.functional import forced_align
from torch.nn.functional import log_softmax
import torchaudio.functional as F
import pickle
from typing import Optional
import yaml
import re

from brainaudio.models.transformer_chunking_lc_time import TransformerModel
from brainaudio.models.gru_b2t_25 import GRU_25
from brainaudio.models.gru_b2t_24 import GRU_24
from brainaudio.datasets.lazy_data_loading import getDatasetLoaders
from brainaudio.training.utils.augmentations import gauss_smooth


def compute_wer(r, h):
    """
    Calculation of WER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    Parameters
    ----------
    r : list
    h : list
    Returns
    -------
    int
    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def _cer_and_wer(decodedSentences, trueSentences, outputType='speech',
                 returnCI=False):
    allCharErr = []
    allChar = []
    allWordErr = []
    allWord = []
    for x in range(len(decodedSentences)):
        decSent = decodedSentences[x]
        trueSent = trueSentences[x]

        nCharErr = compute_wer([c for c in trueSent], [c for c in decSent])
        if outputType == 'handwriting':
            trueWords = trueSent.replace(">", " > ").split(" ")
            decWords = decSent.replace(">", " > ").split(" ")
        elif outputType == 'speech' or outputType == 'speech_sil':
            trueWords = trueSent.split(" ")
            decWords = decSent.split(" ")
        nWordErr = compute_wer(trueWords, decWords)

        allCharErr.append(nCharErr)
        allWordErr.append(nWordErr)
        allChar.append(len(trueSent))
        allWord.append(len(trueWords))

    cer = np.sum(allCharErr) / np.sum(allChar)
    wer = np.sum(allWordErr) / np.sum(allWord)
    
    per_sentence_wer = np.array(allWordErr) / np.array(allWord)

    if not returnCI:
        return cer, wer, per_sentence_wer
    
    else:
        allChar = np.array(allChar)
        allCharErr = np.array(allCharErr)
        allWord = np.array(allWord)
        allWordErr = np.array(allWordErr)

        nResamples = 10000
        resampledCER = np.zeros([nResamples,])
        resampledWER = np.zeros([nResamples,])
        for n in range(nResamples):
            resampleIdx = np.random.randint(0, allChar.shape[0], [allChar.shape[0]])
            resampledCER[n] = np.sum(allCharErr[resampleIdx]) / np.sum(allChar[resampleIdx])
            resampledWER[n] = np.sum(allWordErr[resampleIdx]) / np.sum(allWord[resampleIdx])
        cerCI = np.percentile(resampledCER, [2.5, 97.5])
        werCI = np.percentile(resampledWER, [2.5, 97.5])

        return (cer, cerCI[0], cerCI[1]), (wer, werCI[0], werCI[1])

def load_model(
    folder: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    modelWeightsFile: Optional[str] = "modelWeights",
    eval_chunk_config: Optional[Dict[str, Optional[int]]] = None,
):
    """
    Load a pre-trained model from a folder.

    config:
        folder (str): Path to folder containing 'modelWeights'.
        device (torch.device): Device to map the model onto.

    Returns:
        torch.nn.Module: The loaded model in eval mode, and the config dictionary.
    """
    
    # --- MODIFIED SECTION ---
    # Determine the path for the config file

    config_path = os.path.join(folder, "args")
    print(f"Loading default pickle config from: {config_path}")
    with open(config_path, "rb") as handle:
        config = pickle.load(handle)
    # ----------------------
        
    # Load config                
    modelType = config['modelType']
    model_config = config['model'][modelType]
    
    if modelType == 'transformer':

        if 'return_final_layer' not in config:
            config['return_final_layer'] = False

        chunked_attention_cfg = model_config.get('chunked_attention')
        if eval_chunk_config is not None and isinstance(chunked_attention_cfg, dict):
            chunked_attention_cfg['eval'] = eval_chunk_config
            print(f"Overriding eval chunk config with: {eval_chunk_config}")

        model = TransformerModel(
            features_list=model_config['features_list'],
            samples_per_patch=model_config['samples_per_patch'],
            dim=model_config['d_model'],
            depth=model_config['depth'],
            heads=model_config['n_heads'],
            mlp_dim_ratio=model_config['mlp_dim_ratio'],
            dim_head=model_config['dim_head'],
            dropout=config['dropout'],
            input_dropout=config['input_dropout'],
            nClasses=config['nClasses'],
            max_mask_pct=config['max_mask_pct'],
            num_masks=config['num_masks'],
            num_participants=len(model_config['features_list']),
            return_final_layer=config['return_final_layer'],
            chunked_attention=chunked_attention_cfg,
        )

    
    elif modelType == 'gru' and model_config['year'] == '2024':
        breakpoint()
        
        model = GRU_24(neural_dim=model_config['nInputFeatures'], n_classes=config['nClasses'], hidden_dim=model_config['nUnits'], 
            layer_dim=model_config['nLayers'], nDays=model_config['nDays'], dropout=config['dropout'], input_dropout=config['input_dropout'],
            strideLen=model_config['strideLen'], kernelLen=model_config['kernelLen'], bidirectional=model_config['bidirectional'], max_mask_pct=config['max_mask_pct'], num_masks=config['num_masks'])

        
    else:
        
        model = GRU_25(neural_dim=model_config['nInputFeatures'], n_classes=config['nClasses'], hidden_dim=model_config['nUnits'], 
            layer_dim=model_config['nLayers'], nDays=model_config['nDays'], dropout=config['dropout'], input_dropout=config['input_dropout'],
            strideLen=model_config['strideLen'], kernelLen=model_config['kernelLen'], bidirectional=model_config['bidirectional'], max_mask_pct=config['max_mask_pct'], num_masks=config['num_masks'])

    # Load weights
    ckpt_path = os.path.join(folder, modelWeightsFile)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    return model, config
            
def compute_per(logits, y, total_edit_distance, total_seq_length):
    
    decodedSeq = torch.argmax(logits, dim=-1)
    decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
    decodedSeq = decodedSeq.numpy()
    decodedSeq = np.array([i for i in decodedSeq if i != 0])
    
    matcher = SequenceMatcher(
                    a=y.tolist(), b=decodedSeq.tolist()
                )  
    
    total_edit_distance += matcher.distance()
    total_seq_length += len(y)
    
    return total_edit_distance, total_seq_length
        

def obtain_word_level_timespans(alignments, scores, ground_truth_sequence, transcript,
                                silence_token_id=40):
    
    """
    Computes word level start and end times.
    
    Parameters
    ----------
    alignments: torch array (T)
        Provides the frame-level alignments using CTC for each output frame, 
        where T is the number of output frames. Each entry is an integer which 
        corresponds to a token of the acoustic model.
        
    scores: torch array (T)
        Provides the log probability for the token in "alignments" at that frame. 
        
    ground_truth_sequence: torch array (N)
        The ground-truth sentence displayed to the participant. 
        Each entry is an integer corresponding to a phoneme or silence token. 
        
    transcript: str
        Ground truth word-level transcript.
        
    silence_token_id: int
        The integer corresponding to the silence token, which denotes word boundaries. 
        
    Returns
    -------
    list
        Entry i in the list contains the start and end times for each token of the ith
        word in the transcript. 
    """
    
    # Compute the length (number of phonemes) in each word
    word_lens = []
    current_word_length = 0
    for tok in ground_truth_sequence:
        if tok == silence_token_id:
            if current_word_length > 0:
                word_lens.append(current_word_length)
            current_word_length = 0 
            word_lens.append(1)
        else:
            current_word_length += 1
        
            
    def unflatten(list_, lengths):
        
        assert len(list_) == sum(lengths)
        i = 0
        ret = []
        # gather all tokens corresponding to a word
        for l in lengths:
            ret.append(list_[i : i + l])
            i += l
        return ret
    
    # apply CTC merging function
    token_spans = F.merge_tokens(alignments, scores)
    word_spans = unflatten(token_spans, word_lens)
    
    word_span_information = []
    
    for word_span, word in zip(word_spans, transcript):
        
        word_span_information.append([word_span[0].start, word_span[-1].end, word])
                
    return word_span_information


def save_transcripts(manifest_paths, partition, save_paths):
    '''
    Saves the transcripts for each participant in the dataset.

    Args:
        manifest_paths (list): List of manifest.json paths for each participant.
        partition (str): The data partition to process ('train' or 'val').
        save_paths (list): List of paths to save the transcripts.
    '''
    trainLoaders, valLoaders, _ = getDatasetLoaders(
        manifest_paths,
        1, 
        return_transcript=True, 
        shuffle_train=False
    )
    
    dataLoaders = trainLoaders if partition == 'train' else valLoaders
    
    for participant_id, dataLoader in enumerate(dataLoaders):
        
        transcriptions = []
        
        for batch in tqdm(dataLoader, desc=f"P{participant_id} Transcripts"):
            
            X, y, X_len, y_len, dayIdxs, transcripts = batch
            
            transcript_trial = transcripts[0].replace(".", "").lower()
            
            transcriptions.append(transcript_trial)
            
        save_path = f"{save_paths[participant_id]}/transcripts_{partition}.pkl"
        with open(save_path, 'wb') as handle:
            pickle.dump(transcriptions, handle)
            
        
def generate_and_save_logits(model, config, partition, device, 
                             manifest_paths, save_paths, participant_ids,
                             chunk_config: Optional[Dict[str, Optional[int]]] = None):
    
    """
    Runs the model forward pass and saves the output logits to a file.

    config:
        model (torch.nn.Module): The neural network model.
        config (dict): Dictionary of arguments.
        partition (str): The data partition to process ('train' or 'val').
        device (torch.device): The device to run the model on.
        manifest_paths (list): List of manifest.json paths per participant.
        save_paths (list): List of paths to save the output files.
        participant_ids (list): List of integer participant ids.
        chunk_config (dict|None): Optional chunking overrides for evaluation.
    """
    
    print(f"--- Starting: Generating logits for '{partition}' partition ---")

    def _format_chunk_tag(cfg: Optional[Dict[str, Optional[int]]]) -> Optional[str]:
        if not cfg:
            return None
        chunk = cfg.get("chunk_size")
        context = cfg.get("context_chunks")
        chunk_str = "full" if chunk is None else str(chunk)
        context_str = "full" if context is None else str(context)
        return f"chunk_{chunk_str}_context_{context_str}"

    chunk_tag = _format_chunk_tag(chunk_config)
    if chunk_tag:
        print(f"Using eval chunk config: {chunk_config}")

    trainLoaders, valLoaders, testLoaders = getDatasetLoaders(
        manifest_paths,
        config["batchSize"], 
        return_transcript=True, 
        shuffle_train=False
    )
    
    if partition == "train":
        dataLoaders = trainLoaders
    elif partition == "val":
        dataLoaders = valLoaders
    elif partition == "test":
        dataLoaders = testLoaders
        
    model.eval()
    with torch.no_grad():
        for dataLoader, participant_id in zip(dataLoaders, participant_ids):
            print(f"Processing participant {participant_id}...")
            
            logits_data = []
            transcriptions = []
            total_edit_distance = 0
            total_seq_length = 0
        
            for batch in tqdm(dataLoader, desc=f"P{participant_id} Logits"):
                
                X, y, X_len, y_len, dayIdxs, transcripts = batch

                X, y, X_len, y_len, dayIdxs = (
                    X.to(device), y.to(device), X_len.to(device), 
                    y_len.to(device), dayIdxs.to(device)
                )
                
                X = gauss_smooth(X, device=device, smooth_kernel_size=config['smooth_kernel_size'], 
                                 smooth_kernel_std=config['gaussianSmoothWidth'])
                
                adjusted_lens = model.compute_length(X_len)
                logits = model.forward(X, X_len, participant_id, dayIdxs)
                
                for i in range(logits.shape[0]):            
                    
                    ali = adjusted_lens[i]
                    yli = y_len[i]
                    
                    logits_data.append(logits[i, :ali].cpu().numpy())
                    transcriptions.append(transcripts[i])
                
                    total_edit_distance, total_seq_length = compute_per(logits[i, :ali].cpu(), 
                    y[i, :yli].cpu(), total_edit_distance, total_seq_length)
            
            if chunk_tag:
                save_path = f"{save_paths[participant_id]}/logits_{partition}_{chunk_tag}.npz"
            else:
                save_path = f"{save_paths[participant_id]}/logits_{partition}.npz"
            print(f"Saving logits for participant {participant_id} to {save_path}")
            np.savez_compressed(save_path, *logits_data)
            print(f"Error Rate for participant {participant_id}: {total_edit_distance/total_seq_length}")
                
            
    print("--- Finished: Logit generation complete. ---")

# ==============================================================================
# FUNCTION 2: Compute Forced Alignments (Self-Contained)
# ==============================================================================
def compute_forced_alignments(partition, save_paths, participant_ids, 
                              silence_token_id=40, blank_id=0):
    """
    Loads pre-computed logits and computes forced alignments.
    All dependencies are passed as arguments.

    config:
        partition (str): The data partition to process.
        save_paths (list): List of paths where logits are and alignments will be saved.
        participant_ids (list): List of participant IDs.
        silence_token_id (int): The ID for the silence token.
        blank_id (int): The ID for the CTC blank token.
    """
    print(f"--- Starting: Computing forced alignments for '{partition}' partition ---")

    for participant_id in participant_ids:
        print(f"Processing participant {participant_id}...")
        
        logits_path = f"{save_paths[participant_id]}logits_{partition}.pkl"
        alignments_save_path = f"{save_paths[participant_id]}word_alignments_{partition}.pkl"
        
        print(f"Loading pre-computed logits from {logits_path}")
        with open(logits_path, 'rb') as handle:
            logits_data_by_day = pickle.load(handle)
            
        word_spans_dict = {}

        for dayIdx, daily_data in tqdm(logits_data_by_day.items(), desc=f"P{participant_id} Alignments"):
            word_spans_dict[dayIdx] = []
            
            for sample_data in daily_data:
                log_probs = sample_data['log_probs'].unsqueeze(0)
                targets = sample_data['targets'].unsqueeze(0)
                transcript = sample_data['transcript']
                
                fa_labels, fa_probs = forced_align(
                    log_probs=log_probs, 
                    targets=targets, 
                    blank=blank_id
                )
                
                words = transcript.split(' ')
                transcript_list = [item for word in words for item in (word, 'SIL')]
                
                word_spans = obtain_word_level_timespans(
                    fa_labels[0], 
                    fa_probs[0], 
                    targets.numpy().squeeze(),
                    transcript=transcript_list, 
                    silence_token_id=silence_token_id
                )
                
                word_spans_dict[dayIdx].append(word_spans)

        print(f"Saving word alignments for participant {participant_id} to {alignments_save_path}")
        with open(alignments_save_path, 'wb') as handle:
            pickle.dump(word_spans_dict, handle)
            
    print("--- Finished: Forced alignment complete. ---")
    

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