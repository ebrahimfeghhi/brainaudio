from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer,
    LexiconConstraint,
    VectorizedLexiconConstraint,
    apply_ctc_rules,
    load_token_to_phoneme_mapping,
    load_phoneme_to_word_mapping,
    compute_wer,
)

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser()
parser.add_argument("start", type=int, nargs="?", default=0, help="Start Index (default: 0)")
parser.add_argument("end", type=int, nargs="?", default=1400, help="End Index (default: 1400)")
parser.add_argument("--no-lexicon", action="store_true", help="Disable lexicon constraint")
parser.add_argument("--beam-size", type=int, default=1, help="Beam size for CTC decoding (default: 1)")
parser.add_argument(
    "--no-vectorized-lexicon",
    dest="use_vectorized_lexicon",
    action="store_false",
    help="Disable the vectorized lexicon and use the CPU implementation",
)
parser.set_defaults(use_vectorized_lexicon=True)

args = parser.parse_args()
start_idx = args.start
end_idx = args.end
beam_size = max(1, args.beam_size)
use_lexicon = not args.no_lexicon
use_vectorized = args.use_vectorized_lexicon
# Configuration
LANGUAGE_MODEL_PATH = "/data2/brain2text/lm/"
TOKENS_TXT = f"{LANGUAGE_MODEL_PATH}units_pytorch.txt"
WORDS_TXT = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt"
LOGITS_PATH = "/data2/brain2text/b2t_25/logits/tm_transformer_combined_reduced_reg_seed_0/logits_val_None_None.npz"

PHONE_DEF = [
    'CTC BLANK', 'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH', 'SIL'
]

def load_and_stack_logits(path, start, end, device='cuda:0'):
    # Load all arrays into a list of Tensors
    data = np.load(path)
    tensors = [torch.from_numpy(data[f"arr_{i}"]) for i in range(start, end)]
    
    # Create the lengths tensor
    lengths = torch.tensor([t.size(0) for t in tensors], device=device)
    
    # Pad and stack automatically (batch_first=True puts batch in dim 0)
    # padding_value=0 is the default, but explicit here for clarity
    batched = pad_sequence(tensors, batch_first=True, padding_value=0).to(device)
    
    return batched, lengths

# load val transcripts
ground_truth_transcript = pd.read_pickle('/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl')

logits_batched, logits_lengths = load_and_stack_logits(LOGITS_PATH, start_idx, end_idx, device='cuda:0')

print(f"\nAfter padding:")
print(f"Logits shape: {logits_batched.shape}")
print(f"Logits lengths: {logits_lengths}")
print(f"Device: {logits_batched.device}\n")

# Load lexicon and mappings
lexicon_cls = VectorizedLexiconConstraint if use_vectorized else LexiconConstraint

if use_lexicon:
    lexicon = lexicon_cls.from_file_paths(
        tokens_file=TOKENS_TXT,
        lexicon_file=WORDS_TXT,
        device=torch.device('cuda:0'),
    )
    phoneme_to_word = load_phoneme_to_word_mapping(WORDS_TXT)
    lexicon_variant = "Vectorized" if use_vectorized else "Standard"
    print(f"{lexicon_variant} lexicon loaded: {len(phoneme_to_word)} words")
else:
    lexicon = None
    phoneme_to_word = None
    print("Lexicon constraint disabled")

token_to_symbol = load_token_to_phoneme_mapping(TOKENS_TXT)

# Create decoder (disable CUDA graphs to allow breakpoints)
blank_index = 0  # CTC blank is always token 0
decoder = BatchedBeamCTCComputer(
    blank_index=blank_index,
    beam_size=beam_size,
    lexicon=lexicon,
    allow_cuda_graphs=False,  # Disable CUDA graphs to allow Python breakpoints
)
print(f"Decoder created (beam size: {decoder.beam_size}, CUDA graphs: {decoder.cuda_graphs_mode})")

# Run beam search
print(f"Running beam search...")
decode_start = time.perf_counter()
result = decoder(logits_batched, logits_lengths)
decode_time = time.perf_counter() - decode_start
print(f"Decoding complete in {decode_time:.2f} seconds!\n")

wer_values = []

# Display results
for b in range(logits_batched.shape[0]):
    
    print(f"=== Utterance {b} ===")
    
    # Show ground truth
    if b < len(ground_truth_transcript):
        gt = ground_truth_transcript[b+start_idx]
        print(f"Ground truth: {gt}")
    
    seq = result.transcript_wb[b, 0]
    seq_filtered = seq[seq >= 0]
    score = result.scores[b, 0].item()

    print(f"Score: {score}")
    print(f"Sequence length: {len(seq_filtered)}")
    
    if score > float('-inf'):
        # Apply CTC rules and decode
        ids_no_blanks = apply_ctc_rules(seq_filtered)
        phoneme_sequence = ' '.join([token_to_symbol.get(t, f'?{t}') for t in ids_no_blanks])
        print(f"Phonemes:     {phoneme_sequence}")
        
        if use_lexicon:
            word_alts = lexicon.decode_sequence_to_words(ids_no_blanks, token_to_symbol, phoneme_to_word, return_alternatives=True)
            # Get primary words for comparison
            decoded_text = ' '.join([alt[0] if alt else word for word, alt in word_alts])
        else:
            # Without lexicon, just join phonemes
            decoded_text = ' '.join([token_to_symbol.get(t, f'?{t}') for t in ids_no_blanks])
        
        print(f"Decoded:      {decoded_text}")
        
        # Compute WER
        if b < len(ground_truth_transcript):
            wer = compute_wer(decoded_text, gt)
            print(f"WER:          {wer:.2%}")
            wer_values.append(wer)
        
        # Display words with homophones (only if using lexicon)
        if use_lexicon:
            print("\nDecoded words with alternatives:")
            for i, (primary_word, alternatives) in enumerate(word_alts, 1):
                if alternatives:
                    print(f"  {i}. {alternatives}")
                else:
                    print(f"  {i}. {primary_word}")
    else:
        print("No valid decode (score is -inf)")
    print()

if wer_values:
    mean_wer = sum(wer_values) / len(wer_values)
    print(f"Mean WER over {len(wer_values)} trials: {mean_wer:.2%}")
else:
    print("No ground-truth transcripts found within the requested range; mean WER unavailable.")
    
print(f"Total decoding wall time: {decode_time:.2f} seconds")
