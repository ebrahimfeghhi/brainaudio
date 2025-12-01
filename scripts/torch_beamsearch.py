from torchaudio.models.decoder import ctc_decoder
import numpy as np 
import torch
from torch.nn.functional import softmax, log_softmax
from brainaudio.inference.eval_metrics import _cer_and_wer
from brainaudio.inference.lm_funcs import normalize_shorthand, compute_wer_from_logits
import pandas as pd

language_model_path = "/data2/brain2text/lm/"
units_txt_file_pytorch = f"{language_model_path}units_pytorch.txt"
imagineville_vocab_phoneme = f"{language_model_path}vocab_lower_100k_pytorch_phoneme.txt"

decoder = ctc_decoder(tokens=units_txt_file_pytorch, lexicon=imagineville_vocab_phoneme, 
                       beam_size=300, nbest=20, lm="/data2/brain2text/lm/lm_dec19_huge_4gram.kenlm", 
                       lm_weight=2.0, word_score=0.1, log_add=True, sil_score=0, beam_threshold=1e3)


decoder_no_lm = ctc_decoder(tokens=units_txt_file_pytorch, lexicon=imagineville_vocab_phoneme, 
                       beam_size=500, nbest=500, lm=None, 
                       lm_weight=0.0, word_score=0.0, log_add=True)


logits_paths = [
    "/data2/brain2text/b2t_25/logits/tm_transformer_combined_reduced_reg_seed_0/logits_val_None_None.npz"]

dataset_ids = [25]  # Specify 24 or 25 for each logits file


wer_dict, wer_list, decoded_sentences = compute_wer_from_logits(
    logits_paths=logits_paths,
    dataset_ids=dataset_ids,
    decoder=decoder_no_lm,
    acoustic_scale=1.0,
    verbose=True,
    save_results=True,
    output_filename="scratch.pkl"
)

breakpoint()

print("\nWER Results:")
for key, wer in wer_dict.items():
    print(f"  {key}: {wer:.4f}")