from torchaudio.models.decoder import ctc_decoder, cuda_ctc_decoder
import numpy as np 
import torch
from brainaudio.inference.eval_metrics import _cer_and_wer
from brainaudio.inference.load_model_generate_logits import normalize_shorthand
import pandas as pd
from argparse import ArgumentParser

# Directories
language_model_path = "/data2/brain2text/lm/"
lexicon_phonemes_file = f"{language_model_path}lexicon_phonemes.txt"
units_txt_file_pytorch = f"{language_model_path}units_pytorch.txt"

units_txt_file_pytorch_char = f"{language_model_path}units_pytorch_character.txt"
lexicon_char_file= f"{language_model_path}lexicon_char.txt"
imagineville_vocab_phoneme = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt"

model_logits_path = "tm_transformer_combined_reduced_reg_seed_0/logits_val_None_None.npz"


def main(args):
    
    year = args.year
    # Loading datasets, models, and LMs
    if year == "25":
        val_transcripts = pd.read_pickle("/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl")
        full_model_logits_path = "/data2/brain2text/b2t_25/logits/" + model_logits_path
    elif year == "24":
        val_transcripts = pd.read_pickle("/data2/brain2text/b2t_24/transcripts_val_cleaned.pkl") #! to be resolved
        full_model_logits_path ="/data2/brain2text/b2t_24/logits/" + model_logits_path
    else:
        raise ValueError("Specified year does not have existing validation transcripts")

    model_logits = np.load(full_model_logits_path)

    # Initialize ctc decoder (parameters already optimized)
    decoder = ctc_decoder(tokens=units_txt_file_pytorch, lexicon=imagineville_vocab_phoneme, 
                        beam_size=300, nbest=20, lm="/data2/brain2text/lm/lm_dec19_huge_4gram.kenlm", 
                        lm_weight=2.0, word_score=0.1, log_add=True, sil_score=0, beam_threshold=1e3)

    acoustic_scale = 0.6
    ground_truth_arr = []
    pred_arr = []
    for idx in range(len(model_logits)):
        if idx % 100 == 0:
            print(idx)
        single_trial_logits = torch.as_tensor(model_logits[f'arr_{idx}']).float().unsqueeze(0)
        beam_search_char = decoder(single_trial_logits * acoustic_scale)
        beam_search_transcript_char = normalize_shorthand(" ".join(beam_search_char[0][0].words).strip())
        ground_truth_sentence = val_transcripts[idx]
        ground_truth_arr.append(ground_truth_sentence)
        pred_arr.append(beam_search_transcript_char)
        
    _, wer, _ = _cer_and_wer(pred_arr, ground_truth_arr)
    print(f"The Word Error Rate of model on b2t_{year}'s val dataset is {wer:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument('-y', '--year', type=str, default="25", help="Determines which year's logits")
    # parser.add_argument('-p', '--path', type=str, default=None, help="Determines the path to the saved logits of the model to be evaluated")
    
    args = parser.parse_args()
    main(args)