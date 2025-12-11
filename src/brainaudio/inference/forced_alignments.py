
import torchaudio.functional as F
import pickle
from torchaudio.functional import forced_align
from tqdm import tqdm
import re


def clean_string(transcript):
    
    transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
    transcript = transcript.replace("--", "").lower()
    
    return transcript



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