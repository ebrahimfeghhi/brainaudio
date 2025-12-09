import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer,
    LexiconConstraint,
    apply_ctc_rules,
    load_token_to_phoneme_mapping,
)


def load_logits(npz_path: Path, trial_index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load logits for a single trial and return (logits, lengths)."""
    data = np.load(npz_path)
    key = f"arr_{trial_index}"
    if key not in data:
        raise KeyError(f"Trial {trial_index} not found in {npz_path}")
    tensor = torch.from_numpy(data[key])
    # Shape [T, V]; add batch dim and move to device
    batched = pad_sequence([tensor], batch_first=True).to(device)
    lengths = torch.tensor([tensor.size(0)], device=device)
    return batched, lengths


def decode_sequence(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    token_to_symbol: dict[int, str],
    lexicon: LexiconConstraint | None,
    allow_cuda_graphs: bool,
) -> tuple[list[str], list[str]]:
    """Run CTC beam search and return raw + collapsed phoneme strings."""
    blank_index = lexicon.blank_index if lexicon is not None else 0
    decoder = BatchedBeamCTCComputer(
        blank_index=blank_index,
        beam_size=1,
        lexicon=lexicon,
        allow_cuda_graphs=allow_cuda_graphs,
    )
    result = decoder(logits, lengths)
    seq = result.transcript_wb[0, 0]
    seq = seq[seq >= 0]
    raw = [token_to_symbol[int(i)] for i in seq]
    collapsed = [token_to_symbol[int(i)] for i in apply_ctc_rules(seq)]
    return raw, collapsed


def find_divergence(seq_a: list[str], seq_b: list[str]) -> int:
    """Return index of first mismatch between two sequences (len overlap)."""
    for idx, (a, b) in enumerate(zip(seq_a, seq_b)):
        if a != b:
            return idx
    return min(len(seq_a), len(seq_b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare phoneme sequences with/without lexicon constraints")
    parser.add_argument("trial", type=int, help="Trial index inside the logits NPZ file (arr_<trial>)")
    parser.add_argument(
        "--logits-path",
        default="/data2/brain2text/b2t_25/logits/tm_transformer_combined_reduced_reg_seed_0/logits_val_None_None.npz",
        help="Path to NPZ file containing logits",
    )
    parser.add_argument(
        "--tokens-path",
        default="/data2/brain2text/lm/units_pytorch.txt",
        help="Path to tokens.txt (phoneme inventory)",
    )
    parser.add_argument(
        "--lexicon-path",
        default="/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt",
        help="Path to lexicon file",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for decoding",
    )
    parser.add_argument(
        "--no-cuda-graphs",
        action="store_true",
        help="Set to disable CUDA graphs inside the decoder",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    token_to_symbol = load_token_to_phoneme_mapping(Path(args.tokens_path))

    logits, lengths = load_logits(Path(args.logits_path), args.trial, device)

    lexicon = LexiconConstraint.from_file_paths(
        tokens_file=args.tokens_path,
        lexicon_file=args.lexicon_path,
        device=device,
    )

    raw_with, collapsed_with = decode_sequence(
        logits,
        lengths,
        token_to_symbol,
        lexicon,
        allow_cuda_graphs=not args.no_cuda_graphs,
    )
    raw_without, collapsed_without = decode_sequence(
        logits,
        lengths,
        token_to_symbol,
        lexicon=None,
        allow_cuda_graphs=not args.no_cuda_graphs,
    )

    print("=== Collapsed sequences ===")
    print("LEX:", " ".join(collapsed_with))
    print("NLE:", " ".join(collapsed_without))

    div_collapsed = find_divergence(collapsed_with, collapsed_without)
    if div_collapsed < min(len(collapsed_with), len(collapsed_without)):
        print(f"Collapsed divergence at index {div_collapsed}: {collapsed_with[div_collapsed]!r} vs {collapsed_without[div_collapsed]!r}")
    else:
        print("Collapsed sequences match within shared length")

    print("\n=== Raw (non-collapsed) sequences ===")
    print("With lexicon   :", " ".join(raw_with))
    print("Without lexicon:", " ".join(raw_without))

    div_raw = find_divergence(raw_with, raw_without)
    if div_raw < min(len(raw_with), len(raw_without)):
        print(f"Raw divergence at index {div_raw}: {raw_with[div_raw]!r} vs {raw_without[div_raw]!r}")
    else:
        print("Raw sequences match within shared length")


if __name__ == "__main__":
    main()
