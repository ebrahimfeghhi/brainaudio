#!/usr/bin/env python3
"""
Train a simple bigram phoneme LM from training transcripts.

This creates an ARPA file compatible with NGramGPULanguageModel.from_arpa().

The key insight: NGramGPULanguageModel expects single-character tokens where
each phoneme index i maps to chr(i + token_offset), with token_offset=100.

Required special tokens in ARPA (handled specially by the parser):
- <s>   : Begin of sentence (maps to BOS_ID = -1)
- </s>  : End of sentence (maps to EOS_ID = -2)
- <unk> : Unknown token (maps to UNK_ID = -3)

Usage:
    # Train on real transcripts
    python scripts/train_simple_phoneme_bigram_lm.py --output test_bigram.arpa

    # Use custom corpus file
    python scripts/train_simple_phoneme_bigram_lm.py --corpus path/to/phonemes.txt --output bigram.arpa
"""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from g2p_en import G2p

# Phoneme vocabulary (matching units_pytorch.txt, excluding blank at index 0)
# Index 0 in the LM corresponds to index 1 in units_pytorch.txt (AA)
# Index 39 in the LM corresponds to index 40 in units_pytorch.txt (| = SIL)
PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
    'SIL'  # word boundary (| in units_pytorch.txt)
]

VOCAB_SIZE = len(PHONEMES)  # 40

# Mapping from phoneme name to index (0-39)
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEMES)}
IDX_TO_PHONEME = {i: p for i, p in enumerate(PHONEMES)}

# Token offset for ARPA encoding
TOKEN_OFFSET = 100

# Default paths
DEFAULT_TRANSCRIPTS = Path("/data2/brain2text/b2t_25/transcripts_train.pkl")


def phoneme_to_char(phoneme: str) -> str:
    """Convert phoneme name to single character for ARPA."""
    idx = PHONEME_TO_IDX[phoneme]
    return chr(idx + TOKEN_OFFSET)


def char_to_phoneme(char: str) -> str:
    """Convert ARPA character back to phoneme name."""
    idx = ord(char) - TOKEN_OFFSET
    return IDX_TO_PHONEME[idx]


def text_to_phonemes(text: str, g2p: G2p, add_inter_word_sil: bool = True) -> list[str]:
    """
    Convert text to phoneme sequence using the same logic as the CTC model.

    This matches the preprocessing in the training code:
        for p in g2p(thisTranscription):
            if addInterWordSymbol and p == ' ':
                phonemes.append('SIL')
            p = re.sub(r'[0-9]', '', p)           # remove stress
            if re.match(r'[A-Z]+', p):            # keep only phoneme labels
                phonemes.append(p)
    """
    phonemes = []

    for p in g2p(text):
        # Space -> SIL (word boundary)
        if add_inter_word_sil and p == ' ':
            phonemes.append('SIL')
            continue

        # Remove stress markers (0, 1, 2)
        p_clean = re.sub(r'[0-9]', '', p)

        # Keep only valid phoneme labels (uppercase letters)
        if re.match(r'^[A-Z]+$', p_clean):
            if p_clean in PHONEME_TO_IDX:
                phonemes.append(p_clean)
            else:
                print(f"Warning: Unknown phoneme '{p_clean}' from g2p output")

    return phonemes


def load_transcripts(transcripts_path: Path) -> list[str]:
    """Load transcripts from pickle file."""
    data = pd.read_pickle(transcripts_path)

    texts = []
    for item in data:
        if isinstance(item, list):
            # Format: [[text], [text], ...]
            text = item[0] if item else ""
        else:
            text = str(item)

        # Clean text: lowercase, remove punctuation except apostrophes
        text = text.lower()
        # Keep letters, spaces, and apostrophes
        text = re.sub(r"[^a-z\s']", '', text)
        # Normalize whitespace
        text = ' '.join(text.split())

        if text:
            texts.append(text)

    return texts


def texts_to_phoneme_sequences(texts: list[str], g2p: G2p) -> list[list[str]]:
    """Convert list of texts to list of phoneme sequences."""
    sequences = []
    for text in texts:
        phonemes = text_to_phonemes(text, g2p, add_inter_word_sil=True)
        if phonemes:
            sequences.append(phonemes)
    return sequences


def compute_probabilities(sequences: list[list[str]], smoothing: float = 0.01):
    """
    Compute unigram and bigram probabilities with add-alpha smoothing.

    Returns:
        unigram_probs: dict mapping phoneme -> probability
        bigram_probs: dict mapping (context, phoneme) -> probability
    """
    # Count unigrams and bigrams
    unigram_counts = Counter()
    bigram_counts = Counter()
    total_tokens = 0

    for seq in sequences:
        # Add <s> at start and </s> at end
        seq_with_markers = ['<s>'] + seq + ['</s>']

        for i, token in enumerate(seq_with_markers):
            # Count unigrams (but not <s>)
            if token != '<s>':
                unigram_counts[token] += 1
                total_tokens += 1

            # Count bigrams
            if i > 0:
                bigram = (seq_with_markers[i-1], token)
                bigram_counts[bigram] += 1

    # Compute unigram probabilities with add-alpha smoothing
    vocab = set(PHONEMES) | {'</s>'}
    alpha = smoothing

    unigram_probs = {}
    for token in vocab:
        count = unigram_counts.get(token, 0)
        prob = (count + alpha) / (total_tokens + alpha * len(vocab))
        unigram_probs[token] = prob

    # Count contexts for bigram MLE
    context_counts = Counter()
    for (ctx, _), count in bigram_counts.items():
        context_counts[ctx] += count

    # Bigram probabilities (MLE with small floor)
    bigram_probs = {}
    for (ctx, tok), count in bigram_counts.items():
        prob = count / context_counts[ctx]
        bigram_probs[(ctx, tok)] = prob

    return unigram_probs, bigram_probs, unigram_counts, bigram_counts


def write_arpa(output_path: Path, unigram_probs: dict, bigram_probs: dict):
    """
    Write ARPA format file with single-character encoding.

    ARPA format requires these special tokens:
    - <s>   : Begin of sentence (with backoff weight)
    - </s>  : End of sentence
    - <unk> : Unknown token
    """
    # Count n-grams for header
    num_unigrams = 1 + 1 + 1 + len(PHONEMES)  # <s> + </s> + <unk> + phonemes
    num_bigrams = len(bigram_probs)

    with open(output_path, 'w') as f:
        # Header
        f.write("\\data\\\n")
        f.write(f"ngram 1={num_unigrams}\n")
        f.write(f"ngram 2={num_bigrams}\n")
        f.write("\n")

        # Unigrams section
        f.write("\\1-grams:\n")

        # <s> - BOS: very low probability (it's only used as context), with backoff
        bos_backoff = -0.5  # log10 backoff weight
        f.write(f"-99.0\t<s>\t{bos_backoff:.6f}\n")

        # </s> - EOS
        eos_prob = unigram_probs.get('</s>', 0.001)
        f.write(f"{math.log10(max(eos_prob, 1e-10)):.6f}\t</s>\n")

        # <unk> - very low probability
        f.write(f"-10.0\t<unk>\n")

        # Regular phonemes (as single chars)
        for phoneme in PHONEMES:
            prob = unigram_probs.get(phoneme, 0.001)
            char = phoneme_to_char(phoneme)
            backoff = -0.4  # log10 backoff weight
            f.write(f"{math.log10(max(prob, 1e-10)):.6f}\t{char}\t{backoff:.6f}\n")

        f.write("\n")

        # Bigrams section
        f.write("\\2-grams:\n")

        for (context, token), prob in sorted(bigram_probs.items()):
            # Convert to ARPA format
            if context == '<s>':
                ctx_str = '<s>'
            else:
                ctx_str = phoneme_to_char(context)

            if token == '</s>':
                tok_str = '</s>'
            else:
                tok_str = phoneme_to_char(token)

            f.write(f"{math.log10(max(prob, 1e-10)):.6f}\t{ctx_str} {tok_str}\n")

        f.write("\n\\end\\\n")


def print_statistics(unigram_probs, bigram_probs, unigram_counts, bigram_counts):
    """Print statistics about the trained LM."""
    print("\n" + "="*70)
    print("TRAINING STATISTICS")
    print("="*70)

    total_unigrams = sum(unigram_counts.values())
    total_bigrams = sum(bigram_counts.values())
    print(f"  Total unigram tokens: {total_unigrams:,}")
    print(f"  Total bigram tokens: {total_bigrams:,}")
    print(f"  Unique unigrams: {len(unigram_counts)}")
    print(f"  Unique bigrams: {len(bigram_counts)}")

    # Check coverage
    covered_phonemes = set(unigram_counts.keys()) & set(PHONEMES)
    missing = set(PHONEMES) - covered_phonemes
    print(f"\n  Phoneme coverage: {len(covered_phonemes)}/{len(PHONEMES)}")
    if missing:
        print(f"  Missing phonemes: {missing}")

    print("\n" + "="*70)
    print("TOP 15 UNIGRAM PROBABILITIES")
    print("="*70)

    sorted_unigrams = sorted(unigram_probs.items(), key=lambda x: -x[1])
    for phoneme, prob in sorted_unigrams[:15]:
        count = unigram_counts.get(phoneme, 0)
        log_prob = math.log10(prob) if prob > 0 else float('-inf')
        if phoneme in PHONEME_TO_IDX:
            char = phoneme_to_char(phoneme)
            idx = PHONEME_TO_IDX[phoneme]
            print(f"  {phoneme:5s} idx={idx:2d} ('{char}'): count={count:6d}, p={prob:.4f}, log10(p)={log_prob:.2f}")
        else:
            print(f"  {phoneme:5s}: count={count:6d}, p={prob:.4f}, log10(p)={log_prob:.2f}")

    print("\n" + "="*70)
    print("TOP 20 BIGRAM PROBABILITIES P(next | context)")
    print("="*70)

    sorted_bigrams = sorted(bigram_probs.items(), key=lambda x: -x[1])
    for (ctx, tok), prob in sorted_bigrams[:20]:
        count = bigram_counts.get((ctx, tok), 0)
        log_prob = math.log10(prob) if prob > 0 else float('-inf')

        if ctx == '<s>':
            ctx_str = '<s>  '
        else:
            ctx_str = f"{ctx:5s}"

        if tok == '</s>':
            tok_str = '</s> '
        else:
            tok_str = f"{tok:5s}"

        print(f"  P({tok_str} | {ctx_str}) = {prob:.4f}  count={count:5d}  log10={log_prob:.2f}")

    print("\n" + "="*70)
    print("PHONEME TO CHAR MAPPING (first 10)")
    print("="*70)
    print("  idx  phoneme  char  ord")
    print("  ---  -------  ----  ---")
    for i in range(min(10, len(PHONEMES))):
        phoneme = PHONEMES[i]
        char = chr(i + TOKEN_OFFSET)
        print(f"  {i:3d}  {phoneme:7s}  '{char}'   {i + TOKEN_OFFSET}")
    print(f"  ...")
    print(f"  {39:3d}  {'SIL':7s}  '{chr(39 + TOKEN_OFFSET)}'   {39 + TOKEN_OFFSET}")


def test_load_arpa(arpa_path: Path):
    """Test loading the ARPA file with NGramGPULanguageModel."""
    try:
        from brainaudio.inference.decoder.ngram_lm import NGramGPULanguageModel
        import torch
    except ImportError as e:
        print(f"\nCannot test loading: {e}")
        return

    print("\n" + "="*70)
    print("TESTING LOAD WITH NGramGPULanguageModel")
    print("="*70)

    try:
        lm = NGramGPULanguageModel.from_arpa(
            str(arpa_path),
            vocab_size=VOCAB_SIZE,  # 40
            token_offset=TOKEN_OFFSET,  # 100
        )
        print(f"  Successfully loaded ARPA file!")
        print(f"  num_states: {lm.num_states}")
        print(f"  num_arcs: {lm.num_arcs}")
        print(f"  max_order: {lm.max_order}")
        print(f"  vocab_size: {lm.vocab_size}")

        # Test advance from BOS state
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lm = lm.to(device)

        states = lm.get_init_states(batch_size=1, bos=True)
        scores, next_states = lm.advance(states)

        print(f"\n  From BOS state, top 10 most likely next phonemes:")
        top10 = scores[0].topk(10)
        for score, idx in zip(top10.values, top10.indices):
            idx = idx.item()
            phoneme = PHONEMES[idx] if idx < len(PHONEMES) else f"?{idx}"
            print(f"    {idx:2d}: {phoneme:5s} score={score.item():.4f}")

        # Test a sequence: "DH AH" (the)
        print(f"\n  Testing sequence 'DH AH' (like 'the'):")
        states = lm.get_init_states(batch_size=1, bos=True)

        for phoneme in ['DH', 'AH']:
            scores, states = lm.advance(states)
            idx = PHONEME_TO_IDX[phoneme]
            score = scores[0, idx].item()
            print(f"    P({phoneme} | context) = {math.exp(score):.4f} (log={score:.4f})")
            # Update state
            states = states[:, idx:idx+1].squeeze(-1)

    except Exception as e:
        print(f"  Error loading: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Train simple bigram phoneme LM")
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS,
                        help="Path to transcripts pickle file")
    parser.add_argument("--corpus", type=Path, default=None,
                        help="Path to pre-phonemized corpus (one sequence per line, space-separated)")
    parser.add_argument("--output", type=Path, default=Path("test_phoneme_bigram.arpa"),
                        help="Output ARPA file path")
    parser.add_argument("--max-sentences", type=int, default=None,
                        help="Limit number of sentences for quick testing")
    parser.add_argument("--no-test-load", action="store_true",
                        help="Skip testing the ARPA with NGramGPULanguageModel")
    args = parser.parse_args()

    # Initialize g2p
    print("Initializing G2P...")
    g2p = G2p()

    # Load data
    if args.corpus and args.corpus.exists():
        print(f"Loading pre-phonemized corpus from {args.corpus}")
        with open(args.corpus) as f:
            lines = f.readlines()
        sequences = []
        for line in lines:
            phonemes = line.strip().split()
            valid = [p for p in phonemes if p in PHONEME_TO_IDX]
            if valid:
                sequences.append(valid)
    else:
        print(f"Loading transcripts from {args.transcripts}")
        texts = load_transcripts(args.transcripts)

        if args.max_sentences:
            texts = texts[:args.max_sentences]
            print(f"  Limited to {len(texts)} sentences")

        print(f"Loaded {len(texts)} sentences")
        print("Sample sentences:")
        for i, t in enumerate(texts[:5]):
            print(f"  {i}: {t}")

        print("\nConverting to phonemes...")
        sequences = texts_to_phoneme_sequences(texts, g2p)

    print(f"\nProcessed {len(sequences)} sequences")
    print(f"Total phoneme tokens: {sum(len(s) for s in sequences):,}")

    # Show sample phoneme sequences
    print("\nSample phoneme sequences:")
    for i, seq in enumerate(sequences[:3]):
        print(f"  {i}: {' '.join(seq[:20])}{'...' if len(seq) > 20 else ''}")

    # Compute probabilities
    print("\nComputing n-gram probabilities...")
    unigram_probs, bigram_probs, unigram_counts, bigram_counts = compute_probabilities(sequences)

    # Print statistics
    print_statistics(unigram_probs, bigram_probs, unigram_counts, bigram_counts)

    # Write ARPA file
    write_arpa(args.output, unigram_probs, bigram_probs)
    print(f"\n{'='*70}")
    print(f"Wrote ARPA file to: {args.output}")
    print(f"{'='*70}")

    # Test loading
    if not args.no_test_load:
        test_load_arpa(args.output)


if __name__ == "__main__":
    main()
