from brainaudio.inference.decoder.ngram_lm.ngram_lm_batched import NGramGPULanguageModel
from brainaudio.inference.decoder import BatchedBeamCTCComputer
from brainaudio.inference.decoder.beam_helpers import load_log_probs
import torch

CHAR_VOCAB = [
    "<sp>",          # space token
    "!", ",", ".", "?", "'",   # punctuation (incl. apostrophe)
] + [chr(i) for i in range(ord('a'), ord('z') + 1)]  # 'a'..'z'

# Build mappings
_CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
_ID_TO_CHAR = {i: c for c, i in _CHAR_TO_ID.items()}

# Convenience indices
SPACE_ID = _CHAR_TO_ID["<sp>"]

def charToId(c: str) -> int:
    """Map raw input char to ID, normalizing space and lowercase."""
    if c == " ":
        c = "<sp>"
    c = c.lower()
    return _CHAR_TO_ID[c]

def idToChar(i: int) -> str:
    return _ID_TO_CHAR[i]



language_model_path = "/data2/brain2text/lm/char_lm/lm_dec19_char_huge_12gram.nemo"
vocab_size = 32
lm = NGramGPULanguageModel.from_nemo(
    lm_path=language_model_path,
    vocab_size=vocab_size
)

log_probs, log_probs_length = load_log_probs("/data2/brain2text/b2t_24/logits/tm_transformer_combined_lw_char/logits_val.npz", [0,1,2,3,4,5,6,7,8,9,10], device="cuda", blank_last_index=True)

alpha = 1.5
decoder = BatchedBeamCTCComputer(blank_index=32, beam_size=200, 
                                 return_best_hypothesis=True, fusion_models=[lm], 
                                fusion_models_alpha=[alpha], beam_threshold=20)

idx = 0

log_probs_one_sample = torch.unsqueeze(log_probs[idx], dim=0)
log_probs_length_one_sample = torch.unsqueeze(log_probs_length[idx], dim=0)
transcripts = decoder.batched_beam_search_torch(log_probs_one_sample, log_probs_length_one_sample)
best_hyp_lm = transcripts.to_nbest_hyps_list()[0]
for i in range(len(best_hyp_lm.n_best_hypotheses)):
    decoded_lm = "".join(idToChar(idx) for idx in best_hyp_lm.n_best_hypotheses[i].y_sequence).replace('<sp>', ' ')
    print(f'{i}: {decoded_lm}')