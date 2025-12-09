"""Shared helper functions for inspecting beam hypotheses during decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch

from .batched_beam_decoding_utils import BatchedBeamHyps, NON_EXISTENT_LABEL_VALUE

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from .lexicon_constraint import VectorizedLexiconConstraint


def materialize_beam_transcript(
    batched_hyps: BatchedBeamHyps, batch_idx: int, beam_idx: int
) -> torch.Tensor:
    """Rebuild the emitted token path for a beam by following parent pointers."""

    seq_len = int(batched_hyps.current_lengths_wb[batch_idx, beam_idx].item())
    if seq_len <= 0:
        return batched_hyps.transcript_wb.new_empty((0,), dtype=torch.long)

    tokens: list[int] = []
    ptr_beam = beam_idx

    for idx in range(seq_len - 1, -1, -1):
        token = int(batched_hyps.transcript_wb[batch_idx, ptr_beam, idx].item())
        if token == NON_EXISTENT_LABEL_VALUE:
            break
        tokens.append(token)

        parent_ptr = int(batched_hyps.transcript_wb_prev_ptr[batch_idx, ptr_beam, idx].item())
        if parent_ptr < 0:
            break
        ptr_beam = parent_ptr

    tokens.reverse()
    return torch.tensor(tokens, device=batched_hyps.transcript_wb.device, dtype=torch.long)


def format_beam_phonemes(
    batched_hyps: BatchedBeamHyps,
    batch_idx: int,
    beam_idx: int,
    token_to_symbol: dict[int, str] | None = None,
) -> str:
    """Return a human-readable phoneme string for the requested beam."""

    seq_tensor = materialize_beam_transcript(batched_hyps, batch_idx, beam_idx)
    if seq_tensor.numel() == 0:
        return "<EMPTY>"

    if token_to_symbol is None:
        return " ".join(str(int(t)) for t in seq_tensor)

    return " ".join(token_to_symbol.get(int(t), str(int(t))) for t in seq_tensor)


def strip_ctc(ids: Sequence[int], blank: int = 0) -> List[int]:
    """Collapse a raw token sequence according to standard CTC rules."""

    return collapse_ctc_sequence(ids, blank)


def decode_beam_texts(
    beam_hyps: BatchedBeamHyps,
    token_table: Dict[int, str],
    lexicon: "VectorizedLexiconConstraint",
    phoneme_to_word: Dict[tuple[str, ...], str] | None,
    top_k: int,
) -> List[List[str]]:
    """Decode up to `top_k` beams per batch element into word strings."""

    decoded: List[List[str]] = []
    beam_limit = min(top_k, beam_hyps.transcript_wb.shape[1]) if top_k > 0 else 0
    for batch_idx in range(beam_hyps.transcript_wb.shape[0]):
        beam_texts: List[str] = []
        for beam_idx in range(beam_limit):
            seq_tensor = materialize_beam_transcript(beam_hyps, batch_idx, beam_idx)
            tokens = strip_ctc(seq_tensor.tolist(), blank=lexicon.blank_index)
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


def decode_best_beams(
    beam_hyps: BatchedBeamHyps,
    token_table: Dict[int, str],
    lexicon: "VectorizedLexiconConstraint",
    phoneme_to_word: Dict[tuple[str, ...], str] | None,
) -> List[str]:
    """Convenience wrapper that returns just the best decoded beam per sample."""

    top_texts = decode_beam_texts(
        beam_hyps=beam_hyps,
        token_table=token_table,
        lexicon=lexicon,
        phoneme_to_word=phoneme_to_word,
        top_k=1,
    )
    return [texts[0] if texts else "<EMPTY>" for texts in top_texts]


def collapse_ctc_sequence(sequence: Sequence[int] | torch.Tensor, blank_index: int) -> List[int]:
    """Remove blanks and repeated tokens from a raw hypothesis."""

    if hasattr(sequence, "cpu"):
        sequence = sequence.cpu().tolist()

    collapsed: List[int] = []
    prev_token: int | None = None
    for token in sequence:
        if token == blank_index:
            prev_token = None
            continue
        if token == prev_token:
            continue
        collapsed.append(int(token))
        prev_token = int(token)
    return collapsed


def decode_sequence_to_text(
    token_sequence: Sequence[int],
    lexicon: "VectorizedLexiconConstraint",
    token_to_symbol: Dict[int, str] | None = None,
    exclude_last_word: bool = False,
) -> str:
    """Convert a collapsed token sequence into a plain word context for the LM."""

    if not token_sequence or token_to_symbol is None:
        return ""

    if not hasattr(lexicon, "decode_sequence_to_words"):
        return ""

    words_with_alts = lexicon.decode_sequence_to_words(
        token_ids=token_sequence,
        token_to_symbol=token_to_symbol,
        lexicon_word_map=None,
        return_alternatives=True,
    )

    if not words_with_alts:
        return ""

    if exclude_last_word:
        words_with_alts = words_with_alts[:-1]

    def _is_real_word(word: str) -> bool:
        return word and not word.startswith("<UNK") and not word.startswith("<PARTIAL")

    context_words = [word for word, _ in words_with_alts if _is_real_word(word)]
    return " ".join(context_words).strip()


def pick_device(requested: str | None) -> torch.device:
    """Return requested torch device or best available default."""

    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_log_probs(
    npz_path: Path,
    trial_indices: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load logits from NPZ, pad to the longest trial, and return log-probs + lengths."""

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
    log_probs = torch.log_softmax(logits, dim=-1)
    lengths_tensor = torch.tensor(lengths, device=device)
    return log_probs, lengths_tensor


def apply_ctc_rules(ids, blank: int = 0) -> List[int]:
    """Remove blanks and merge repeats (wrapper for `collapse_ctc_sequence`)."""

    return collapse_ctc_sequence(ids, blank)


def load_token_to_phoneme_mapping(tokens_file: Path) -> Dict[int, str]:
    """Load token ID -> phoneme symbol mapping."""

    token_to_symbol = {}
    with open(tokens_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            token_to_symbol[idx] = line.strip()
    return token_to_symbol


def load_phoneme_to_word_mapping(lexicon_file: Path) -> Dict[Tuple[str, ...], str]:
    """Build phoneme sequence -> word mapping from lexicon file."""

    phoneme_to_word: Dict[Tuple[str, ...], str] = {}
    with open(lexicon_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            phonemes = tuple(p for p in parts[1:] if p != "|")
            phoneme_to_word[phonemes] = word
    return phoneme_to_word


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute Word Error Rate (WER) using Levenshtein distance."""

    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()

    d = [[0] * (len(ref_words) + 1) for _ in range(len(hyp_words) + 1)]

    for i in range(len(hyp_words) + 1):
        d[i][0] = i
    for j in range(len(ref_words) + 1):
        d[0][j] = j

    for i in range(1, len(hyp_words) + 1):
        for j in range(1, len(ref_words) + 1):
            if hyp_words[i - 1] == ref_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1

    return d[len(hyp_words)][len(ref_words)] / max(len(ref_words), 1)
