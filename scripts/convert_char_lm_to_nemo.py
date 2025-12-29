#!/usr/bin/env python3
"""
Convert character-level ARPA LM to NeMo format.

This script:
1. Reads the gzipped character ARPA file (with literal symbols like 'a', 'b', '<sp>')
2. Converts to the chr(token_id + 100) encoding expected by NGramGPULanguageModel
3. Saves as .nemo format for fast loading

Usage:
    python scripts/convert_char_lm_to_nemo.py
"""

import gzip
import re
import sys
from pathlib import Path

# Paths
ARPA_GZ_PATH = Path("/data2/brain2text/lm/char_lm/lm_dec19_char_huge_12gram.arpa.gz")
UNITS_PATH = Path("/data2/brain2text/lm/char_lm/units_pytorch_character.txt")
OUTPUT_ARPA = Path("/data2/brain2text/lm/char_lm/lm_dec19_char_huge_12gram_encoded.arpa")
OUTPUT_NEMO = Path("/data2/brain2text/lm/char_lm/lm_dec19_char_huge_12gram.nemo")

# Token offset (must match NGramGPULanguageModel)
TOKEN_OFFSET = 100


def load_vocabulary(units_path: Path) -> dict[str, int]:
    """Load vocabulary mapping from units file."""
    vocab = {}
    with open(units_path) as f:
        for idx, line in enumerate(f):
            symbol = line.strip()
            if symbol:
                vocab[symbol] = idx
    return vocab


def create_arpa_symbol_map(vocab: dict[str, int]) -> dict[str, str]:
    """Create mapping from ARPA symbols to encoded tokens.

    ARPA file uses:
        - <s>, </s>: BOS/EOS (keep as-is, handled specially)
        - <unk>: unknown (keep as-is)
        - <sp>: space (maps to '|' in vocab)
        - Single chars: a-z, punctuation

    Vocab (units_pytorch_character.txt):
        - Index 0: '-' (blank for CTC)
        - Index 1: '|' (space/word boundary)
        - Index 2-6: '!', ',', '.', '?', "'"
        - Index 7-32: a-z
    """
    symbol_map = {
        "<s>": "<s>",      # Keep special symbols
        "</s>": "</s>",
        "<unk>": "<unk>",
    }

    # Map <sp> to | (word boundary)
    if "|" in vocab:
        symbol_map["<sp>"] = chr(vocab["|"] + TOKEN_OFFSET)

    # Map all other symbols directly
    for symbol, idx in vocab.items():
        if symbol == "-":
            # Skip blank token - shouldn't appear in LM
            continue
        if symbol == "|":
            # Already handled as <sp>
            continue
        # Map symbol to encoded token
        symbol_map[symbol] = chr(idx + TOKEN_OFFSET)

    return symbol_map


def convert_ngram_line(line: str, symbol_map: dict[str, str]) -> str:
    """Convert a single n-gram line to encoded format."""
    parts = line.strip().split("\t")
    if len(parts) < 2:
        return line  # Not an n-gram line

    # Parse: weight \t symbol1 symbol2 ... \t [backoff]
    weight = parts[0]
    symbols = parts[1].split()
    backoff = parts[2] if len(parts) > 2 else None

    # Convert symbols
    converted = []
    for sym in symbols:
        if sym in symbol_map:
            converted.append(symbol_map[sym])
        else:
            print(f"Warning: Unknown symbol '{sym}', keeping as-is", file=sys.stderr)
            converted.append(sym)

    # Rebuild line
    result = f"{weight}\t{' '.join(converted)}"
    if backoff is not None:
        result += f"\t{backoff}"

    return result


def convert_arpa_file(input_path: Path, output_path: Path, symbol_map: dict[str, str]):
    """Convert entire ARPA file to encoded format."""
    print(f"Converting {input_path} -> {output_path}")
    print(f"Symbol map sample: {dict(list(symbol_map.items())[:10])}")

    # Open input (gzipped or plain)
    if input_path.suffix == ".gz":
        f_in = gzip.open(input_path, "rt", encoding="utf-8")
    else:
        f_in = open(input_path, "r", encoding="utf-8")

    in_ngrams = False
    ngram_pattern = re.compile(r"^\\(\d+)-grams:$")
    line_count = 0

    with f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line_count += 1
            if line_count % 10_000_000 == 0:
                print(f"  Processed {line_count:,} lines...")

            stripped = line.strip()

            # Check for section markers
            if stripped.startswith("\\"):
                if ngram_pattern.match(stripped):
                    in_ngrams = True
                elif stripped in ("\\data\\", "\\end\\"):
                    in_ngrams = False
                f_out.write(line)
                continue

            # Convert n-gram lines
            if in_ngrams and stripped and not stripped.startswith("ngram"):
                converted = convert_ngram_line(stripped, symbol_map)
                f_out.write(converted + "\n")
            else:
                f_out.write(line)

    print(f"  Total lines processed: {line_count:,}")
    print(f"  Output saved to: {output_path}")


def load_and_save_nemo(arpa_path: Path, nemo_path: Path, vocab_size: int):
    """Load ARPA with NGramGPULanguageModel and save as .nemo."""
    print(f"\nLoading encoded ARPA into NGramGPULanguageModel...")
    print(f"  ARPA: {arpa_path}")
    print(f"  Vocab size: {vocab_size}")

    from brainaudio.inference.decoder.ngram_lm import NGramGPULanguageModel

    model = NGramGPULanguageModel.from_arpa(
        lm_path=str(arpa_path),
        vocab_size=vocab_size,
        token_offset=TOKEN_OFFSET,
        normalize_unk=True,
    )

    print(f"  Loaded successfully!")
    print(f"    num_states: {model.num_states:,}")
    print(f"    num_arcs: {model.num_arcs:,}")
    print(f"    max_order: {model.max_order}")

    print(f"\nSaving to .nemo format: {nemo_path}")
    model.save_to(str(nemo_path))
    print(f"  Saved successfully! Size: {nemo_path.stat().st_size / 1e9:.2f} GB")


def main():
    print("=" * 60)
    print("CHARACTER LM CONVERSION TO NEMO FORMAT")
    print("=" * 60)

    # Step 1: Load vocabulary
    print("\n[1/4] Loading vocabulary...")
    vocab = load_vocabulary(UNITS_PATH)
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Tokens: {list(vocab.keys())}")

    # Step 2: Create symbol mapping
    print("\n[2/4] Creating symbol mapping...")
    symbol_map = create_arpa_symbol_map(vocab)
    print(f"  Mapping examples:")
    for sym, encoded in list(symbol_map.items())[:8]:
        if encoded in ("<s>", "</s>", "<unk>"):
            print(f"    '{sym}' -> '{encoded}' (special)")
        else:
            idx = ord(encoded) - TOKEN_OFFSET
            print(f"    '{sym}' -> '{encoded}' (idx={idx})")

    # Step 3: Convert ARPA file
    print("\n[3/4] Converting ARPA file (this may take a while for large files)...")
    convert_arpa_file(ARPA_GZ_PATH, OUTPUT_ARPA, symbol_map)

    # Step 4: Load and save as .nemo
    print("\n[4/4] Converting to .nemo format...")
    load_and_save_nemo(OUTPUT_ARPA, OUTPUT_NEMO, vocab_size=len(vocab))

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Encoded ARPA: {OUTPUT_ARPA}")
    print(f"NeMo model: {OUTPUT_NEMO}")
    print("\nTo use in your decoder:")
    print(f"  --char-lm-path {OUTPUT_NEMO}")


if __name__ == "__main__":
    main()
