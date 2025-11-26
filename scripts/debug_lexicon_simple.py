"""Very simple test: manually check what the lexicon allows at each step for utterance 2"""

from brainaudio.inference.decoder import (
    LexiconConstraint, 
    load_token_to_phoneme_mapping,
)

LANGUAGE_MODEL_PATH = "/data2/brain2text/lm/"
TOKENS_TXT = f"{LANGUAGE_MODEL_PATH}units_pytorch.txt"
WORDS_TXT = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt"

# Load
lexicon = LexiconConstraint.from_file_paths(
    tokens_file=TOKENS_TXT,
    lexicon_file=WORDS_TXT,
    device='cuda:0',
)
token_to_symbol = load_token_to_phoneme_mapping(TOKENS_TXT)
symbol_to_token = {v: k for k, v in token_to_symbol.items()}

# The problematic sequence from utterance 2
# "not to" is fine, then we get "K AE K P ..."
problem_seq = "N AA T | T UW | K AE K".split()
problem_ids = [symbol_to_token[s] for s in problem_seq]

print(f"Testing: {' '.join(problem_seq)}")
print(f"IDs: {problem_ids}\n")

# Check what's valid after "K AE K"
valid_next = lexicon.get_valid_next_tokens(problem_ids)
valid_symbols = sorted([token_to_symbol[t] for t in valid_next if t in token_to_symbol])

print(f"Valid tokens after 'K AE K': {len(valid_next)}")
print(f"Valid symbols: {valid_symbols}")

# Check if P is in there
p_token = symbol_to_token['P']
print(f"\nIs 'P' (token {p_token}) valid? {p_token in valid_next}")

# Now check what happens if we add P anyway
problem_seq_with_p = problem_seq + ['P']
problem_ids_with_p = problem_ids + [p_token]
valid_after_p = lexicon.get_valid_next_tokens(problem_ids_with_p)

print(f"\nValid tokens after 'K AE K P': {len(valid_after_p)}")
print(f"Valid symbols: {sorted([token_to_symbol[t] for t in valid_after_p if t in token_to_symbol])}")
