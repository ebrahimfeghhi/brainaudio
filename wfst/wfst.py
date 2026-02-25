# do a hacky import since brainaudio requires python 3.12 and the .wfst env requires an older
# python version 
import importlib                                                                                                                                                                                                    
import wfst_utils
importlib.reload(wfst_utils)
from wfst_utils import build_lm_decoder, lm_decode, arrange_logits, _cer_and_wer, augment_nbest
import sys
import numpy as np
import re
import pickle
import pandas as pd
import sys                                      


dataset = 'b2t_24'
load_lm = True
seeds_list = [0,1,2,3,4,5,6,7,8,9]


if dataset == "b2t_25":
    blank_penalty = np.log(90)
    acoustic_scale = 0.325
    beam = 17
    rescore = True
    base_path = "/home/ebrahim/data2/brain2text/b2t_25/logits/"
    save_path = "/home/ebrahim/data2/brain2text/b2t_25/wfst_outputs/"
    folder_name = [f"neurips_b2t_25_causal_transformer_v4_prob_1_seed_{i}" for i in seeds_list]
    logits_file_name = ["logits_train_1:full_context:20.npz"] * len(seeds_list)
    nbest_save_path_arr = [f"time_masked_transformer_25/seed_{i}_unidir_train" for i in seeds_list]
    ms_per_output = 80

if dataset == "b2t_24":

    blank_penalty = np.log(7)
    acoustic_scale = 0.5
    beam = 18
    rescore = True
    base_path = "/home/ebrahim/data2/brain2text/b2t_24/logits/"
    save_path = "/home/ebrahim/data2/brain2text/b2t_24/wfst_outputs/"
    folder_name = [f"neurips_b2t_24_bidir_transformer_seed_{i}" for i in seeds_list]
    logits_file_name = ["logits_train_chunk:full_context:full.npz"] * len(seeds_list)
    nbest_save_path_arr = [f"time_masked_transformer_24/seed_{i}_bidir_train" for i in seeds_list]
    ms_per_output = 100 # 100 for transformer, 80 for gru
    
lmDir = "/home/ebrahim/data2/brain2text/lm/speech_5gram/lang_test"
nbest_value = 100
return_n_best = True

print(f"MS PER OUTPUT: {ms_per_output}")
breakpoint()

import resource

rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

if load_lm and 'ngramDecoder' not in globals(): 
    
    ngramDecoder = build_lm_decoder(
        lmDir,
        acoustic_scale=acoustic_scale, #1.2
        nbest=nbest_value,
        beam=beam,
    )
    print("loaded LM")

rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"Peak RSS after LM load: {rss_after / (1024**2):.2f} GB (delta: {(rss_after - rss_before) / (1024**2):.2f} GB)")

# folder name
seeds_list = [0,1,2,3,4,5,6,7,8,9]
folder_name = [f"neurips_b2t_25_bidirectional_transformer_v4_seed_{i}" for i in seeds_list]
logits_file_name = ["logits_train_chunk:full_context:full.npz"] * len(seeds_list)
nbest_save_path_arr = [f"time_masked_transformer_25/seed_{i}_bidir_train" for i in seeds_list]

import tracemalloc
import resource
import time

tracemalloc.start()
rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

rescore_bool = True

for fn, lm, nb in zip(folder_name, logits_file_name, nbest_save_path_arr):

    logits_all = np.load(f"{base_path}{fn}/{lm}")

    nbest_save_path = f"{save_path}{nb}.pkl"
    timing_save_path = f"{save_path}{nb}_timing.pkl"

    if dataset == "b2t_25":
        
        nbest_augmented_save_path = f"{save_path}{nb}_augmented"

    nbest_outputs = []
    nbest_augmented_outputs = []
    trial_lengths_ms = []
    wfst_decode_times_ms = []

    for trial_idx in range(len(logits_all.keys())):
        
        if trial_idx % 100 == 0:
            
            print(f"Decoding trial {trial_idx}...")
        
        logits = logits_all[f'arr_{trial_idx}']

        trial_length_ms = logits.shape[0] * ms_per_output
        trial_lengths_ms.append(trial_length_ms)
            
        rearranged_logits = arrange_logits(logits)
        
        t_start = time.perf_counter()
        nbest = lm_decode(
                        ngramDecoder,
                        rearranged_logits[0],
                        blankPenalty=blank_penalty,
                        returnNBest=return_n_best,
                        rescore=rescore_bool,
                    )
        t_end = time.perf_counter()
        decode_time_ms = (t_end - t_start) * 1000
        wfst_decode_times_ms.append(decode_time_ms)

        rtf = decode_time_ms / trial_length_ms
        if trial_idx % 100 == 0:
            print(f"  Trial {trial_idx}: RTF = {rtf:.4f} (decode={decode_time_ms:.1f}ms, trial_len={trial_length_ms:.1f}ms)")

        if dataset == 'b2t_25':
            
            nbest_augmented = augment_nbest(nbest, acoustic_scale=acoustic_scale, top_candidates_to_augment=20, score_penalty_percent=0.01)
            nbest_augmented_outputs.append(nbest_augmented)
            
            
        nbest_outputs.append(nbest)
        
        with open(nbest_save_path, 'wb') as f:
            pickle.dump(nbest_outputs, f)
            
        if dataset == 'b2t_25':
            with open(nbest_augmented_save_path, 'wb') as f:
                pickle.dump(nbest_augmented_outputs, f)

    # Save timing data for OPT rescoring RTF computation
    with open(timing_save_path, 'wb') as f:
        pickle.dump({'trial_lengths_ms': trial_lengths_ms, 'wfst_decode_times_ms': wfst_decode_times_ms}, f)

    mean_rtf = np.mean([dt / tl for dt, tl in zip(wfst_decode_times_ms, trial_lengths_ms)])
    print(f"\nWFST Mean RTF: {mean_rtf:.4f}")

current_ram, peak_ram = tracemalloc.get_traced_memory()
tracemalloc.stop()

rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

print(f"\n--- Decoding Memory Usage ---")
print(f"Peak Python RAM (tracemalloc): {peak_ram / (1024**3):.2f} GB")
print(f"Peak process RSS:              {rss_after / (1024**2):.2f} GB (delta: {(rss_after - rss_before) / (1024**2):.2f} GB)")