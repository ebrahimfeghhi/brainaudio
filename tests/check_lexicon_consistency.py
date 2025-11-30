"""Compare lexicon-constrained vs unconstrained decoding outputs.

For every trial where the unconstrained decoder emits phonemes that fully map
onto words in the lexicon, this script checks that enabling the lexicon does
not change the decoded sentence (and therefore keeps the WER identical).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer,
    LexiconConstraint,
    apply_ctc_rules,
    compute_wer,
    load_phoneme_to_word_mapping,
    load_token_to_phoneme_mapping,
)

LANGUAGE_MODEL_PATH = "/data2/brain2text/lm/"
TOKENS_TXT = f"{LANGUAGE_MODEL_PATH}units_pytorch.txt"
WORDS_TXT = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt"
LOGITS_PATH = (
    "/data2/brain2text/b2t_25/logits/"
    "tm_transformer_combined_reduced_reg_seed_0/logits_val_None_None.npz"
)
TRANSCRIPT_PICKLE = "/data2/brain2text/b2t_25/transcripts_val_cleaned.pkl"
DEVICE = torch.device("cuda:0")


@dataclass
class DecodeOutput:
    text: str
    tokens: List[int]
    phonemes: List[str]
    phonemes_pre_ctc: List[str]
    valid_words: bool

def _load_logits(path: str, start: int, end: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.load(path)
    tensors = [torch.from_numpy(data[f"arr_{i}"]) for i in range(start, end)]
    lengths = torch.tensor([t.size(0) for t in tensors], device=device)
    batched = pad_sequence(tensors, batch_first=True, padding_value=0).to(device)
    return batched, lengths


def _decode(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    beam_size: int,
    runtime_lexicon: LexiconConstraint | None,
    mapping_lexicon: LexiconConstraint,
) -> Tuple[DecodeOutput, ...]:
    
    
    decoder = BatchedBeamCTCComputer(
        blank_index=0,
        beam_size=beam_size,
        lexicon=runtime_lexicon,
        allow_cuda_graphs=False,
    )
    
    result = decoder(logits, lengths)
    token_to_symbol = load_token_to_phoneme_mapping(TOKENS_TXT)
    phoneme_to_word = load_phoneme_to_word_mapping(WORDS_TXT)

    outputs: List[DecodeOutput] = []
    for batch_idx in range(logits.size(0)):
        seq = result.transcript_wb[batch_idx, 0]
        seq_filtered = seq[seq >= 0]
        phonemes_pre = [token_to_symbol.get(int(t), f"?{int(t)}") for t in seq_filtered]
        ids_no_blanks = apply_ctc_rules(seq_filtered)
        phoneme_seq = [token_to_symbol.get(int(t), f"?{int(t)}") for t in ids_no_blanks]

        decoded_words: List[str]
        valid_words = True
        if runtime_lexicon is not None:
            word_alts = runtime_lexicon.decode_sequence_to_words(
                token_ids=ids_no_blanks,
                token_to_symbol=token_to_symbol,
                lexicon_word_map=phoneme_to_word,
                return_alternatives=True,
            )
            decoded_words = [alts[0] if alts else word for word, alts in word_alts]
            valid_words = all(not word.startswith("<") for word in decoded_words)
        else:
            word_alts = mapping_lexicon.decode_sequence_to_words(
                token_ids=ids_no_blanks,
                token_to_symbol=token_to_symbol,
                lexicon_word_map=phoneme_to_word,
                return_alternatives=True,
            )
            decoded_words = [alts[0] if alts else word for word, alts in word_alts]
            valid_words = all(not word.startswith("<") for word in decoded_words)

        text = " ".join(decoded_words)
        token_list = ids_no_blanks if isinstance(ids_no_blanks, list) else ids_no_blanks.tolist()
        outputs.append(
            DecodeOutput(
                text=text,
                tokens=token_list,
                phonemes=phoneme_seq,
                phonemes_pre_ctc=phonemes_pre,
                valid_words=valid_words,
            )
        )
    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check lexicon consistency")
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("--beam-size", type=int, default=1)
    args = parser.parse_args()

    logits, lengths = _load_logits(LOGITS_PATH, args.start, args.end, DEVICE)
    refs = pd.read_pickle(TRANSCRIPT_PICKLE)
    if hasattr(refs, "iloc"):
        refs = refs.iloc[args.start : args.end].tolist()
    else:
        refs = list(refs)[args.start : args.end]

    lexicon = LexiconConstraint.from_file_paths(TOKENS_TXT, WORDS_TXT, device=DEVICE)

    outputs_no_lex = _decode(logits, lengths, args.beam_size, None, lexicon)
    outputs_with_lex = _decode(logits, lengths, args.beam_size, lexicon, lexicon)

    mismatches = []
    valid_trials = 0
    for idx, (ref, no_lex, with_lex) in enumerate(zip(refs, outputs_no_lex, outputs_with_lex)):
        if not no_lex.valid_words:
            continue
        valid_trials += 1
        wer_no = compute_wer(no_lex.text, ref)
        wer_yes = compute_wer(with_lex.text, ref)
        if wer_no != wer_yes:
            mismatches.append(
                (
                    idx + args.start,
                    wer_no,
                    wer_yes,
                    no_lex.text,
                    with_lex.text,
                    no_lex.phonemes,
                    with_lex.phonemes,
                    no_lex.phonemes_pre_ctc,
                    with_lex.phonemes_pre_ctc,
                )
            )

    print(f"Checked {valid_trials} valid trials between [{args.start}, {args.end}).")
    if mismatches:
        print(f"Found {len(mismatches)} mismatches:")
        for row in mismatches:
            (
                idx,
                wer_no,
                wer_yes,
                text_no,
                text_yes,
                phones_no,
                phones_yes,
                phones_pre_no,
                phones_pre_yes,
            ) = row
            print("- Trial", idx)
            print(f"  WER no-lex: {wer_no:.2%}")
            print(f"  WER lex:    {wer_yes:.2%}")
            print(f"  no-lex text: {text_no}")
            print(f"  lex text:    {text_yes}")
            print(f"  no-lex phonemes: {' '.join(phones_no)}")
            print(f"  lex phonemes:    {' '.join(phones_yes)}")
            print(f"  no-lex raw phonemes: {' '.join(phones_pre_no)}")
            print(f"  ye-lex raw phonemes: {' '.join(phones_pre_yes)}")
    else:
        print("No WER differences detected on valid trials.")


if __name__ == "__main__":
    main()
