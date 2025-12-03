from typing import Dict, List

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

if __name__ == "__main__":
    # Example lexicon
    lexicon = [
        ["HH", "W", "AH", "L", "OW", "|"],
        ["HH", "W", "IY", "|"],
        ["B", "AE", "T", "|"],
    ]

    trie = PhonemeTrie()
    for seq in lexicon:
        trie.add(seq)

    queries = [
        ["HH"],         # True – valid prefix
        ["HH", "W"],   # True – still on a path
        ["HH", "W", "N"],   # False – diverges
    ]

    for q in queries:
        print(q, "=>", trie.is_valid_prefix(q))