"""Tiny end-to-end CTC beam-search check with neural LM fusion.

By default the script decodes validation trials 0 and 1 in a single
batch, applies the vectorized lexicon constraint, and fuses a
HuggingFace causal LM at word boundaries. Keeping the surface area small
still exercises the real decoder path (including batched contexts) so it
is easy to reason about `HuggingFaceLMFusion` changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer,
    VectorizedLexiconConstraint,
    HuggingFaceLMFusion,
)

DEFAULT_LOGITS = "/data2/brain2text/b2t_25/logits/tm_transformer_combined_reduced_reg_seed_0/logits_val_None_None.npz"
DEFAULT_TOKENS = "/data2/brain2text/lm/units_pytorch.txt"
DEFAULT_LEXICON = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CTC beam search + HF LM fusion smoke test")
    parser.add_argument(
        "--trials",
        type=int,
        nargs="+",
        default=[1],
        help="Validation trial indices to decode together (default: 0 1)",
    )
    parser.add_argument("--beam-size", type=int, default=50, help="CTC beam size (default: 3)")
    parser.add_argument(
        "--top-beams",
        type=int,
        default=10,
        help="Number of beams to print with scores (default: 10)",
    )
    parser.add_argument("--model", default="google/gemma-3-270m", help="HuggingFace causal LM checkpoint")
    parser.add_argument("--hf-token", default=None, help="Optional HF token for gated models")
    parser.add_argument("--lm-weight", type=float, default=1, help="Fusion weight passed to HuggingFaceLMFusion")
    parser.add_argument("--max-context-length", type=int, default=128, help="Token budget (including BOS)")
    parser.add_argument("--device", default=None, help="Torch device for CTC + LM (default: cuda if available)")
    parser.add_argument("--logits", type=Path, default=Path(DEFAULT_LOGITS), help="NPZ file containing validation logits")
    parser.add_argument("--tokens", type=Path, default=Path(DEFAULT_TOKENS), help="units_pytorch.txt file")
    parser.add_argument("--lexicon", type=Path, default=Path(DEFAULT_LEXICON), help="lexicon file")
    return parser.parse_args()


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_logits(npz_path: Path, trial_indices: Sequence[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(npz_path)
    arrays = []
    lengths = []
    for idx in trial_indices:
        key = f"arr_{idx}"
        if key not in data:
            raise KeyError(f"{npz_path} missing {key}")
        arr = data[key]
        arrays.append(arr)
        lengths.append(arr.shape[0])

    max_time = max(lengths)
    padded = []
    for arr in arrays:
        if arr.shape[0] < max_time:
            pad = ((0, max_time - arr.shape[0]), (0, 0))
            arr = np.pad(arr, pad, mode="constant", constant_values=0.0)
        padded.append(arr)

    logits = torch.from_numpy(np.stack(padded, axis=0)).to(device)
    lengths_tensor = torch.tensor(lengths, device=device)
    return logits, lengths_tensor


def load_token_table(tokens_file: Path) -> Dict[int, str]:
    table: Dict[int, str] = {}
    with tokens_file.open() as fh:
        for idx, line in enumerate(fh):
            table[idx] = line.strip()
    return table

def load_phoneme_to_word(lexicon_file: Path) -> Dict[tuple[str, ...], str]:
    mapping: Dict[tuple[str, ...], str] = {}
    with lexicon_file.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[tuple(p for p in parts[1:] if p != "|")] = parts[0]
    return mapping


def strip_ctc(ids: Sequence[int], blank: int = 0) -> List[int]:
    clean: List[int] = []
    prev = None
    for idx in ids:
        if idx == blank:
            prev = None
            continue
        if idx == prev:
            continue
        clean.append(int(idx))
        prev = idx
    return clean


def decode_best_beams(
    transcripts: torch.Tensor,
    token_table: Dict[int, str],
    lexicon: VectorizedLexiconConstraint,
    phoneme_to_word: Dict[tuple[str, ...], str],
) -> List[str]:
    top_texts = decode_beam_texts(
        transcripts=transcripts,
        token_table=token_table,
        lexicon=lexicon,
        phoneme_to_word=phoneme_to_word,
        top_k=1,
    )
    return [texts[0] if texts else "<EMPTY>" for texts in top_texts]


def decode_beam_texts(
    transcripts: torch.Tensor,
    token_table: Dict[int, str],
    lexicon: VectorizedLexiconConstraint,
    phoneme_to_word: Dict[tuple[str, ...], str],
    top_k: int,
) -> List[List[str]]:
    """Decode up to top_k beams per batch element into word strings."""

    decoded: List[List[str]] = []
    beam_limit = min(top_k, transcripts.shape[1]) if top_k > 0 else 0
    for batch_idx in range(transcripts.shape[0]):
        beam_texts: List[str] = []
        for beam_idx in range(beam_limit):
            seq = transcripts[batch_idx, beam_idx]
            valid = seq[seq >= 0].tolist()
            tokens = strip_ctc(valid, blank=lexicon.blank_index)
            if not tokens:
                beam_texts.append("<EMPTY>")
                continue
            word_alts = lexicon.decode_sequence_to_words(
                token_ids=tokens,
                token_to_symbol=token_table,
                lexicon_word_map=phoneme_to_word,
                return_alternatives=True,
            )
            words = [alts[0] if alts else word for word, alts in word_alts]
            beam_texts.append(" ".join(words))
        decoded.append(beam_texts)
    return decoded


def main():
    args = parse_args()
    device = pick_device(args.device)
    logits, lengths = load_logits(args.logits, args.trials, device)

    lexicon = VectorizedLexiconConstraint.from_file_paths(
        tokens_file=str(args.tokens),
        lexicon_file=str(args.lexicon),
        device=device,
    )
    token_table = load_token_table(args.tokens)
    phoneme_to_word = load_phoneme_to_word(args.lexicon)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        token=args.hf_token,
    ).to(device)

    lm_fusion = HuggingFaceLMFusion(
        model=model,
        tokenizer=tokenizer,
        weight=args.lm_weight,
        homophone_aggregation="max",
        device=device,
        max_context_length=args.max_context_length,
    )

    decoder = BatchedBeamCTCComputer(
        blank_index=lexicon.blank_index,
        beam_size=args.beam_size,
        lexicon=lexicon,
        lm_fusion=lm_fusion,
        allow_cuda_graphs=False,
    )

    result = decoder(logits, lengths)
    top_k = max(0, min(args.top_beams, result.transcript_wb.shape[1]))
    decoded_beams = decode_beam_texts(
        transcripts=result.transcript_wb,
        token_table=token_table,
        lexicon=lexicon,
        phoneme_to_word=phoneme_to_word,
        top_k=top_k,
    )
    decoded_texts = [texts[0] if texts else "<EMPTY>" for texts in decoded_beams]

    print("\n=== Neural LM Fusion Decode ===")
    for batch_idx, (trial_idx, text) in enumerate(zip(args.trials, decoded_texts)):
        best_score = result.scores[batch_idx, 0].item()
        print(f"Trial {trial_idx:3d} | Beam size {args.beam_size} | Score {best_score:.4f}")
        print(f"   Best: {text}")

        if top_k == 0:
            continue

        print("   Top beams:")
        for beam_rank, beam_text in enumerate(decoded_beams[batch_idx]):
            beam_score = result.scores[batch_idx, beam_rank].item()
            print(f"     #{beam_rank:02d} | log {beam_score:.4f} | {beam_text}")


if __name__ == "__main__":
    main()
