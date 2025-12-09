#!/usr/bin/env python3
"""Print per-beam phoneme decodes for a single validation trial."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer,
    LexiconConstraint,
    VectorizedLexiconConstraint,
    apply_ctc_rules,
    materialize_beam_transcript,
    load_token_to_phoneme_mapping,
    load_phoneme_to_word_mapping,
)

LANGUAGE_MODEL_PATH = Path("/data2/brain2text/lm/")
TOKENS_TXT = LANGUAGE_MODEL_PATH / "units_pytorch.txt"
WORDS_TXT = LANGUAGE_MODEL_PATH / "vocab_lower_100k_pytorch_phoneme.txt"
LOGITS_PATH = Path("/data2/brain2text/b2t_25/logits/tm_transformer_combined_reduced_reg_seed_0/logits_val_None_None.npz")
TRANSCRIPTS_PKL = Path("/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect all beams for a single trial")
    parser.add_argument("trial", type=int, default=1, help="Validation trial index to decode (0-based)")
    parser.add_argument("--beam-size", type=int, default=25, help="Beam size to use")
    parser.add_argument(
        "--max-beams", type=int, default=None, help="Limit how many beams to print (default: all)"
    )
    parser.add_argument("--device", default="cuda:0", help="Device for logits and decoding")
    parser.add_argument("--no-lexicon", action="store_true", help="Disable lexicon constraint")
    parser.add_argument(
        "--vectorized-lexicon",
        dest="use_vectorized_lexicon",
        action="store_true",
        help="Use vectorized lexicon (default)",
    )
    parser.add_argument(
        "--no-vectorized-lexicon",
        dest="use_vectorized_lexicon",
        action="store_false",
        help="Use CPU lexicon implementation",
    )
    parser.set_defaults(use_vectorized_lexicon=True)
    return parser.parse_args()


def load_single_trial(path: Path, trial_idx: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(path)
    key = f"arr_{trial_idx}"
    if key not in data:
        raise KeyError(f"{key} not found in {path}")
    logits = torch.from_numpy(data[key]).unsqueeze(0).to(device)
    lengths = torch.tensor([logits.shape[1]], device=device)
    return logits, lengths


def load_word_to_phonemes(lexicon_file: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    with open(lexicon_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            phonemes = [p for p in parts[1:] if p]
            if word not in mapping:
                mapping[word] = phonemes
    return mapping


def sentence_to_phoneme_string(sentence: str, word_map: dict[str, list[str]]) -> str:
    if not sentence:
        return ""
    tokens: list[str] = []
    for raw_word in sentence.split():
        normalized = re.sub(r"[^a-z']", "", raw_word.lower())
        if not normalized:
            continue
        phonemes = word_map.get(normalized)
        if phonemes:
            tokens.extend(phonemes)
        else:
            tokens.append(f"<UNK:{raw_word}>")
    return " ".join(tokens)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    logits_batched, logits_lengths = load_single_trial(LOGITS_PATH, args.trial, device=device)

    lexicon = None
    phoneme_to_word = None
    token_to_symbol = load_token_to_phoneme_mapping(TOKENS_TXT)
    word_to_phonemes = load_word_to_phonemes(WORDS_TXT)

    if not args.no_lexicon:
        lexicon_cls = VectorizedLexiconConstraint if args.use_vectorized_lexicon else LexiconConstraint
        lexicon = lexicon_cls.from_file_paths(TOKENS_TXT, WORDS_TXT, device=device)
        phoneme_to_word = load_phoneme_to_word_mapping(WORDS_TXT)
        print(f"Lexicon enabled ({'vectorized' if args.use_vectorized_lexicon else 'standard'})")
    else:
        print("Lexicon disabled\n")

    decoder = BatchedBeamCTCComputer(
        blank_index=0,
        beam_size=max(1, args.beam_size),
        lexicon=lexicon,
        allow_cuda_graphs=False,
    )

    ground_truth_sentence = None
    ground_truth_phonemes = ""
    if TRANSCRIPTS_PKL.exists():
        transcripts = pd.read_pickle(TRANSCRIPTS_PKL)
        if 0 <= args.trial < len(transcripts):
            ground_truth_sentence = transcripts[args.trial]
            ground_truth_phonemes = sentence_to_phoneme_string(ground_truth_sentence, word_to_phonemes)

    print(f"Decoding trial {args.trial} with beam size {decoder.beam_size}...")
    result = decoder(logits_batched, logits_lengths)

    if ground_truth_sentence is not None:
        print("\nGround truth text:     " + ground_truth_sentence)
        print("Ground truth phonemes: " + (ground_truth_phonemes or "<unavailable>"))
    else:
        print("\nGround truth text unavailable")

    if lexicon is None or phoneme_to_word is None:
        print("\nLexicon not available, cannot print words/homophones.")
        return

    beams_to_show = args.max_beams or decoder.beam_size
    beams_to_show = min(beams_to_show, decoder.beam_size)

    scores = result.scores[0]
    for beam_idx in range(beams_to_show):
        seq_filtered = materialize_beam_transcript(result, 0, beam_idx).cpu()
        score = scores[beam_idx].item()
        tokens = apply_ctc_rules(seq_filtered)

        print(f"\nBeam {beam_idx:02d} | score={score:.3f}")
        if seq_filtered.numel() == 0 or score == float("-inf"):
            print("  <empty beam>")
            continue

        word_alts = lexicon.decode_sequence_to_words(
            token_ids=tokens,
            token_to_symbol=token_to_symbol,
            lexicon_word_map=phoneme_to_word,
            return_alternatives=True,
        )
        words_only = " ".join(primary for primary, _ in word_alts) or "<none>"
        print(f"  Words: {words_only}")
        for pos, (primary, alts) in enumerate(word_alts, 1):
            if alts:
                pretty = ", ".join(alts)
                print(f"    {pos:02d}. {primary} -> [{pretty}]")
            else:
                print(f"    {pos:02d}. {primary}")


if __name__ == "__main__":
    main()
