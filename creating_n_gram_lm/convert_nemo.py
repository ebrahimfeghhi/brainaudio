from brainaudio.inference.decoder.ngram_lm import NGramGPULanguageModel
import os

# --- 1. Define Only the Tokens in the ARPA ---
core_phonemes = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]
# SIL is also in the ARPA
arpa_tokens = core_phonemes + ["SIL"]

# --- 2. Build the Map ---
# LM indices 0-39 for phonemes (no blank in LM)
# The CTC decoder handles the mapping: CTC index i -> LM index i-1
# 'AA' -> 0, 'AE' -> 1, ..., 'SIL' -> 39
vocab_map = {token: i for i, token in enumerate(arpa_tokens)}

print(f"Vocab Map Size: {len(vocab_map)}")
print(f"Check: 'AA' maps to index {vocab_map['AA']} (Should be 0)")
print(f"Check: 'SIL' maps to index {vocab_map['SIL']} (Should be 39)")

# --- 3. Convert ---
arpa_path = "/home/ebrahim/brainaudio/creating_n_gram_lm/phoneme_10gram.arpa"
nemo_output_path = "/home/ebrahim/brainaudio/creating_n_gram_lm/phoneme_10gram.nemo"

# vocab_size=40 for the 40 phonemes (no blank in LM)
lm = NGramGPULanguageModel.from_arpa(
    lm_path=arpa_path,
    vocab_size=40,
    token_offset=None,
    vocab_map=vocab_map
)

print(f"Saving to {nemo_output_path}...")
lm.save_to(nemo_output_path)
print("Done.")