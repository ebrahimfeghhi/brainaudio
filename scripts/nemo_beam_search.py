from brainaudio.inference.decoder import BatchedBeamCTCComputer, LexiconConstraint
import numpy as np
import torch
import pandas as pd

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

def apply_ctc_rules(ids):
    """Apply CTC rules: remove blanks (0) and merge consecutive repeats."""
    if hasattr(ids, 'cpu'):
        ids = ids.cpu().numpy()
    
    clean_ids = []
    prev_id = None
    
    for id_val in ids:
        if id_val == 0:  # Skip blank
            prev_id = None
            continue
        if id_val == prev_id:  # Skip repeats
            continue
        clean_ids.append(int(id_val))
        prev_id = id_val
    
    return clean_ids

def load_phoneme_to_word_mapping(lexicon_file):
    """Build phoneme sequence -> word mapping from lexicon."""
    phoneme_to_word = {}
    with open(lexicon_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            phonemes = tuple(p for p in parts[1:] if p != '|')
            phoneme_to_word[phonemes] = word
    return phoneme_to_word

def load_token_to_phoneme_mapping(tokens_file):
    """Load token ID -> phoneme symbol mapping."""
    token_to_symbol = {}
    with open(tokens_file, 'r') as f:
        for idx, line in enumerate(f):
            token_to_symbol[idx] = line.strip()
    return token_to_symbol


def compute_wer(hypothesis, reference):
    """Simple WER computation using Levenshtein distance."""
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()
    
    # Levenshtein distance on words
    d = [[0] * (len(ref_words) + 1) for _ in range(len(hyp_words) + 1)]
    
    for i in range(len(hyp_words) + 1):
        d[i][0] = i
    for j in range(len(ref_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(hyp_words) + 1):
        for j in range(1, len(ref_words) + 1):
            if hyp_words[i-1] == ref_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return d[len(hyp_words)][len(ref_words)] / max(len(ref_words), 1)


# Load and prepare logits
model_logits = np.load(LOGITS_PATH)
logits_0 = model_logits["arr_0"]
logits_1 = model_logits["arr_1"]

# load val transcripts
ground_truth_transcript = pd.read_pickle('/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl')

print(f"Original logits shapes:")
print(f"  Utterance 0: {logits_0.shape}, min={logits_0.min():.2f}, max={logits_0.max():.2f}")
print(f"  Utterance 1: {logits_1.shape}, min={logits_1.min():.2f}, max={logits_1.max():.2f}")
print(f"  Utterance 1 all -inf? {np.all(np.isinf(logits_1))}")

# Pad to same time dimension
max_time = max(logits_0.shape[0], logits_1.shape[0])

# Store original shapes
orig_shape_0 = logits_0.shape[0]
orig_shape_1 = logits_1.shape[0]

if logits_0.shape[0] < max_time:
    # Pad with zeros (neutral probabilities), will be masked by logits_lengths
    pad_width = ((0, max_time - logits_0.shape[0]), (0, 0))
    logits_0 = np.pad(logits_0, pad_width, mode='constant', constant_values=0)
if logits_1.shape[0] < max_time:
    # Pad with zeros (neutral probabilities), will be masked by logits_lengths
    pad_width = ((0, max_time - logits_1.shape[0]), (0, 0))
    logits_1 = np.pad(logits_1, pad_width, mode='constant', constant_values=0)

# Batch and convert to torch
logits_batched = torch.from_numpy(np.stack([logits_0, logits_1], axis=0)).to('cuda:0')
logits_lengths = torch.from_numpy(np.array([
    model_logits["arr_0"].shape[0],
    model_logits["arr_1"].shape[0]
])).to('cuda:0')

print(f"\nAfter padding:")
print(f"Logits shape: {logits_batched.shape}")
print(f"Logits lengths: {logits_lengths}")
print(f"Device: {logits_batched.device}\n")

# Load lexicon and mappings
lexicon = LexiconConstraint.from_file_paths(
    tokens_file=TOKENS_TXT,
    lexicon_file=WORDS_TXT,
    device=torch.device('cuda:0'),
)

phoneme_to_word = load_phoneme_to_word_mapping(WORDS_TXT)
token_to_symbol = load_token_to_phoneme_mapping(TOKENS_TXT)

print(f"Lexicon loaded: {len(phoneme_to_word)} words")

# Create decoder (disable CUDA graphs to allow breakpoints)
decoder = BatchedBeamCTCComputer(
    blank_index=lexicon.blank_index,
    beam_size=10,
    lexicon=lexicon,
    allow_cuda_graphs=True,  # Disable CUDA graphs to allow Python breakpoints
)
print(f"Decoder created (beam size: {decoder.beam_size}, CUDA graphs: {decoder.cuda_graphs_mode})")

# Run beam search
print(f"Running beam search...")
result = decoder(logits_batched, logits_lengths)
print(f"Decoding complete!\n")

# Display results
for b in range(logits_batched.shape[0]):
    print(f"=== Utterance {b} ===")
    
    # Show ground truth
    if b < len(ground_truth_transcript):
        gt = ground_truth_transcript[b]
        print(f"Ground truth: {gt}")
    
    seq = result.transcript_wb[b, 0]
    seq_filtered = seq[seq >= 0]
    score = result.scores[b, 0].item()
    
    print(f"Score: {score}")
    print(f"Sequence length: {len(seq_filtered)}")
    
    if score > float('-inf'):
        # Apply CTC rules and decode
        ids_no_blanks = apply_ctc_rules(seq_filtered)
        word_alts = lexicon.decode_sequence_to_words(ids_no_blanks, token_to_symbol, phoneme_to_word, return_alternatives=True)
        
        # Get primary words for comparison
        decoded_text = ' '.join([alt[0] if alt else word for word, alt in word_alts])
        print(f"Decoded:      {decoded_text}")
        
        # Compute WER
        if b < len(ground_truth_transcript):
            wer = compute_wer(decoded_text, gt)
            print(f"WER:          {wer:.2%}")
        
        # Display words with homophones
        print("\nDecoded words with alternatives:")
        for i, (primary_word, alternatives) in enumerate(word_alts, 1):
            if alternatives:
                print(f"  {i}. {alternatives}")
            else:
                print(f"  {i}. {primary_word}")
    else:
        print("No valid decode (score is -inf)")
    print()