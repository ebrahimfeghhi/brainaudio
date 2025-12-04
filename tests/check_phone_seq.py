from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

class PhonemeTrie:
    def __init__(self) -> None:
        self.root: Dict[str, Dict] = {}

    def add(self, phonemes: List[str]) -> None:
        """Insert a phoneme sequence into the trie."""
        node = self.root
        for token in phonemes:
            node = node.setdefault(token, {})
        node.setdefault("__end__", True)

    def is_valid_prefix(self, phonemes: List[str]) -> bool:
        """Return True if the sequence is a valid prefix (not necessarily a full word)."""
        node = self.root
        for token in phonemes:
            if token not in node:
                return False
            node = node[token]
        return True

DEFAULT_LEXICON = Path("/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt")


def iter_lexicon_sequences(path: Path) -> Iterable[List[str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            yield parts[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check if phoneme prefixes exist in the full lexicon trie.")
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=DEFAULT_LEXICON,
        help="Path to lexicon file (word followed by phonemes per line).",
    )
    parser.add_argument(
        "--query",
        action="append",
        nargs="+",
        help="Phoneme prefix to test (provide multiple --query blocks to check several prefixes).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.lexicon.exists():
        raise FileNotFoundError(f"Lexicon file not found: {args.lexicon}")

    trie = PhonemeTrie()
    for seq in iter_lexicon_sequences(args.lexicon):
        trie.add(seq)

    queries = args.query if args.query is not None else [["HH"], ["AA"], ["IH"], ["G"], ["NG"]]

    for q in queries:
        print(f"{q} => {trie.is_valid_prefix(q)}")