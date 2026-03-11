import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path



parser = argparse.ArgumentParser("Reformat data with forced alignment")

parser.add_argument("--b2t24-data-path", type=Path, default=None,
                        help="Path to b2t24 source data")
parser.add_argument("--b2t25-data-path", type=Path, default=None,
                        help="Path to b2t25 source data")
parser.add_argument("--alignments-24-train", type=Path, default=None,
                        help="Path to presaved alignments of b2t24 train set")
parser.add_argument("--alignments-25-train", type=Path, default=None,
                        help="Path to presaved alignments of b2t25 train set")
parser.add_argument("--alignments-24-val", type=Path, default=None,
                        help="Path to presaved alignments of b2t24 validation set")
parser.add_argument("--alignments-25-train", type=Path, default=None,
                        help="Path to presaved alignments of b2t25 validation set")
parser.add_argument("")



def convert_alignments_to_dict(alignments):
    
    """Converts alignments into a dict with keys as word end frames and values as words.
    
    Parameters
    ----------
    alignments_list: list
        A list containing the forced alignments for all trials from a given day.
        Each element in the list contains information for a given trial, also organized 
        as a list. Each trial list is formatted so that each element is a list containing
        three elements: [word_start_frame, word_end_frame, word_text], where frame denotes 
        the index of the model output. 
        
    Returns 
    -------
    list
        Returns a list of dicts. Each dict is formatted so that the word_end_frames are keys
        and the values are the corresponding word. 
    """
        
    alignments_as_dict = []
    
    for trial_level_alignment in alignments:
        
        trial_level_dict = {}
        
        for word_level_alignment in trial_level_alignment:
            
            word_end = word_level_alignment[1]
            word = word_level_alignment[2]
            
            if word == 'SIL':
                continue
            
            trial_level_dict[word_end] = word
            
            
        alignments_as_dict.append(trial_level_dict)
        
    return alignments_as_dict



def save_with_alignments(brain2text_data, alignments):
    
    """
    Reformat and save alignments into the origial pkl dataset.
    """
    
    b2t_with_fa = []
    
    for dayIdx, day in enumerate(brain2text_data):
        
        if dayIdx in alignments.keys():
            day_specific_forced_alignments = alignments[dayIdx]
            
            # make sure that the number of trials is the same
            assert len(day_specific_forced_alignments) == len(day['sentenceDat'])
            
            day["forced_alignments"] = convert_alignments_to_dict(day_specific_forced_alignments)
            
        b2t_with_fa.append(day)
        
    return b2t_with_fa



def main():
    args = parser.parse_args()
    brain2text_24_data = pd.read_pickle(args.b2t24_data_path)
    brain2text_25_data = pd.read_pickle(args.b2t25_data_path)
    alignments_24_train = pd.read_pickle(args.alignments_24_train)
    alignments_24_val = pd.read_pickle(args.alignments_24_val)
    alignments_25_train = pd.read_pickle(args.alignments_25_train)
    alignments_25_val = pd.read_pickle(args.alignments_25_val)

    b2t_24_with_fa_train = save_with_alignments(brain2text_24_data["train"], alignments_24_train)
    b2t_24_with_fa_val = save_with_alignments(brain2text_24_data["val"], alignments_24_val)

    b2t_25_with_fa_train = save_with_alignments(brain2text_25_data["train"], alignments_25_train)
    b2t_25_with_fa_val = save_with_alignments(brain2text_25_data["val"], alignments_25_val)

    brain2text_24_data["train"] = b2t_24_with_fa_train
    brain2text_24_data["val"] = b2t_24_with_fa_val

    with open("/data2/brain2text/b2t_24/brain2text24_with_fa", "wb") as handle:
        pickle.dump(brain2text_24_data, handle)
        
        
    brain2text_25_data["train"] = b2t_25_with_fa_train
    brain2text_25_data["val"] = b2t_25_with_fa_val

    with open("/data2/brain2text/b2t_25/brain2text25_with_fa", "wb") as handle:
        pickle.dump(brain2text_25_data, handle)

    print("----Forced Alignment added to data set----")
        

if __name__ == "__main__":
    main()
