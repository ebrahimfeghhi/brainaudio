#!/usr/bin/env python3
"""
Train a phoneme-level n-gram LM on LibriSpeech LM corpus using KenLM.

This creates an ARPA file compatible with NGramGPULanguageModel.from_arpa().

The corpus is phonemized using g2p_en with single-character encoding where
each phoneme index i maps to chr(i + 100).

Usage:
    python scripts/train_librispeech_phoneme_lm.py

Output files saved to /data2/brain2text/lm/phoneme_lm/:
    - librispeech_phonemes.txt  (phonemized corpus, single-char encoding)
    - phoneme_10gram.arpa       (ARPA format LM)
"""

import os
import sys
import gzip
import shutil
import urllib.request
import subprocess
import re
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

# Try to import g2p
try:
    from g2p_en import G2p
except ImportError:
    print("ERROR: g2p_en not installed. Run: pip install g2p_en")
    sys.exit(1)

# Configuration
CORPUS_URL = "http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
OUTPUT_DIR = Path("/data2/brain2text/lm/phoneme_lm")
RAW_TEXT_FILE = OUTPUT_DIR / "librispeech_raw.txt"
PHONEME_FILE = OUTPUT_DIR / "librispeech_phonemes.txt"
ARPA_FILE = OUTPUT_DIR / "phoneme_10gram.arpa"

# N-gram order
NGRAM_ORDER = 10

# Token offset for ARPA encoding (must match NGramGPULanguageModel)
TOKEN_OFFSET = 100

# Phoneme vocabulary (matching units_pytorch.txt, excluding blank)
# Index 0 = AA, Index 39 = SIL (word boundary)
PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
    'SIL'  # word boundary
]
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEMES)}
VALID_PHONEMES = set(PHONEMES)


def phoneme_to_char(phoneme: str) -> str:
    """Convert phoneme name to single character for ARPA."""
    idx = PHONEME_TO_IDX[phoneme]
    return chr(idx + TOKEN_OFFSET)


def clean_phoneme(p: str) -> str | None:
    """Clean phoneme: remove stress markers, validate."""
    p_clean = re.sub(r'[0-9]', '', p)
    if re.match(r'^[A-Z]+$', p_clean) and p_clean in VALID_PHONEMES:
        return p_clean
    return None


# Global g2p instance for multiprocessing
_g2p = None

def init_worker():
    """Initialize g2p in worker process."""
    global _g2p
    # Suppress g2p stdout
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        _g2p = G2p()


def process_lines(lines: list[str]) -> list[str]:
    """Convert text lines to phoneme sequences (single-char encoded).

    Matches the phonemization in brain2text_2024.py:
        for p in g2p(thisTranscription):
            if addInterWordSymbol and p == ' ':
                phonemes.append('SIL')
            p = re.sub(r'[0-9]', '', p)           # remove stress
            if re.match(r'[A-Z]+', p):            # keep only phoneme labels
                phonemes.append(p)
        if addInterWordSymbol:
            phonemes.append('SIL')  # SIL at end of sentence
    """
    global _g2p
    results = []

    for line in lines:
        text = line.strip().lower()
        if not text:
            continue

        # Get phonemes from g2p - matching brain2text_2024.py exactly
        phoneme_chars = []
        for p in _g2p(text):
            # Add SIL for word boundaries (spaces)
            if p == ' ':
                phoneme_chars.append(phoneme_to_char('SIL'))
                continue
            # Remove stress markers and keep only valid phonemes
            p_clean = re.sub(r'[0-9]', '', p)
            if re.match(r'^[A-Z]+$', p_clean) and p_clean in VALID_PHONEMES:
                phoneme_chars.append(phoneme_to_char(p_clean))

        # Add SIL at end of sentence (matching brain2text_2024.py)
        if phoneme_chars:
            phoneme_chars.append(phoneme_to_char('SIL'))
            # Join with spaces (KenLM expects space-separated tokens)
            results.append(' '.join(phoneme_chars))

    return results


def download_corpus():
    """Download LibriSpeech LM corpus if not present."""
    if RAW_TEXT_FILE.exists():
        print(f"Corpus already exists: {RAW_TEXT_FILE}")
        return

    print(f"Downloading LibriSpeech LM corpus from {CORPUS_URL}...")
    zip_path = RAW_TEXT_FILE.with_suffix('.txt.gz')

    # Download with progress
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent}%)", end='', flush=True)

    urllib.request.urlretrieve(CORPUS_URL, zip_path, show_progress)
    print()

    print("Decompressing...")
    with gzip.open(zip_path, 'rb') as f_in:
        with open(RAW_TEXT_FILE, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    zip_path.unlink()
    print(f"Saved to {RAW_TEXT_FILE}")


def phonemize_corpus():
    """Phonemize the corpus using multiprocessing."""
    if PHONEME_FILE.exists():
        print(f"Phonemized corpus already exists: {PHONEME_FILE}")
        return

    print(f"Phonemizing corpus...")
    print(f"  Input: {RAW_TEXT_FILE}")
    print(f"  Output: {PHONEME_FILE}")

    # Count total lines
    print("  Counting lines...")
    with open(RAW_TEXT_FILE, 'r') as f:
        total_lines = sum(1 for _ in f)
    print(f"  Total lines: {total_lines:,}")

    # Process in parallel
    num_workers = max(1, mp.cpu_count() - 2)
    chunk_size = 5000
    print(f"  Using {num_workers} workers, chunk size {chunk_size}")

    processed_lines = 0

    with open(RAW_TEXT_FILE, 'r') as f_in, open(PHONEME_FILE, 'w') as f_out:
        pool = mp.Pool(num_workers, initializer=init_worker)
        pbar = tqdm(total=total_lines, desc="  Phonemizing", unit=" lines")

        chunk = []
        pending_results = []

        for line in f_in:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                # Submit chunk for processing
                result = pool.apply_async(process_lines, (chunk,))
                pending_results.append(result)
                chunk = []

                # Write completed results
                while pending_results and pending_results[0].ready():
                    result = pending_results.pop(0)
                    phoneme_lines = result.get()
                    for pl in phoneme_lines:
                        f_out.write(pl + '\n')
                    pbar.update(len(phoneme_lines))
                    processed_lines += len(phoneme_lines)

        # Submit final chunk
        if chunk:
            result = pool.apply_async(process_lines, (chunk,))
            pending_results.append(result)

        # Wait for all remaining results
        for result in pending_results:
            phoneme_lines = result.get()
            for pl in phoneme_lines:
                f_out.write(pl + '\n')
            pbar.update(len(phoneme_lines))
            processed_lines += len(phoneme_lines)

        pool.close()
        pool.join()
        pbar.close()

    print(f"  Wrote {processed_lines:,} phonemized lines to {PHONEME_FILE}")


def train_kenlm():
    """Train n-gram LM with KenLM. (Disabled - requires KenLM installation)"""
    print("SKIPPING KenLM training (lmplz not installed)")
    print("To train later, install KenLM and run:")
    print(f"  lmplz -o {NGRAM_ORDER} -S 80% --text {PHONEME_FILE} --arpa {ARPA_FILE} --discount_fallback")
    return


def print_sample():
    """Print sample of phonemized corpus for verification."""
    print("\n" + "="*60)
    print("SAMPLE PHONEMIZED LINES (for verification)")
    print("="*60)

    with open(PHONEME_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            tokens = line.strip().split()
            # Convert back to phoneme names for readability
            phoneme_names = []
            for t in tokens[:20]:  # First 20 phonemes
                idx = ord(t) - TOKEN_OFFSET
                if 0 <= idx < len(PHONEMES):
                    phoneme_names.append(PHONEMES[idx])
                else:
                    phoneme_names.append(f'?{idx}')
            print(f"  {i+1}: {' '.join(phoneme_names)}{'...' if len(tokens) > 20 else ''}")

    print("\n" + "="*60)
    print("PHONEME TO CHAR MAPPING (for reference)")
    print("="*60)
    print("  idx  phoneme  char")
    for i in [0, 1, 2, 38, 39]:
        print(f"  {i:3d}  {PHONEMES[i]:7s}  '{chr(i + TOKEN_OFFSET)}'")


def test_load():
    """Test loading the ARPA file with NGramGPULanguageModel."""
    if not ARPA_FILE.exists():
        print("\n[ARPA file not created - skipping load test]")
        return

    print("\n" + "="*60)
    print("TESTING LOAD WITH NGramGPULanguageModel")
    print("="*60)

    try:
        from brainaudio.inference.decoder.ngram_lm import NGramGPULanguageModel
        import torch
    except ImportError as e:
        print(f"  Cannot test: {e}")
        return

    try:
        lm = NGramGPULanguageModel.from_arpa(
            str(ARPA_FILE),
            vocab_size=40,
            token_offset=TOKEN_OFFSET,
        )
        print(f"  Successfully loaded!")
        print(f"  num_states: {lm.num_states}")
        print(f"  num_arcs: {lm.num_arcs}")
        print(f"  max_order: {lm.max_order}")
        print(f"  vocab_size: {lm.vocab_size}")

        # Quick test
        lm = lm.to('cuda' if torch.cuda.is_available() else 'cpu')
        states = lm.get_init_states(batch_size=1, bos=True)
        scores, _ = lm.advance(states)
        print(f"\n  Score stats from BOS:")
        print(f"    min={scores.min().item():.3f}, max={scores.max().item():.3f}, mean={scores.mean().item():.3f}")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("="*60)
    print("LIBRISPEECH PHONEME LM TRAINING")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"N-gram order: {NGRAM_ORDER}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download corpus
    download_corpus()

    # Step 2: Phonemize
    phonemize_corpus()

    # Step 3: Train LM
    train_kenlm()

    # Step 4: Show sample
    print_sample()

    # Step 5: Test load
    test_load()

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Phonemized corpus: {PHONEME_FILE}")
    print(f"ARPA LM: {ARPA_FILE}")
    print()
    print("To use in your decoder:")
    print(f"  --phoneme-lm-path {ARPA_FILE}")


if __name__ == "__main__":
    main()
