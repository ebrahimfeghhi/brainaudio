"""Unit tests for the baseline (non-vectorized) LexiconConstraint."""

from pathlib import Path
import sys

import torch
import pytest

# Ensure the source tree is importable when tests run via `pytest tests`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from brainaudio.inference.decoder.lexicon_constraint import LexiconConstraint


@pytest.fixture()
def simple_lexicon() -> LexiconConstraint:
    """Create a tiny lexicon with two multi-phoneme words and one single phoneme word."""

    # Each lexicon entry already contains the trailing word-boundary token ("|").
    lexicon_sequences = [
        [1, 2, 5],  # word "AB"
        [1, 3, 5],  # word "AC"
        [4, 5],     # word "D"
    ]
    return LexiconConstraint(
        lexicon=lexicon_sequences,
        blank_index=0,
        device=torch.device("cpu"),
        word_list=["ab", "ac", "d"],
        word_boundary_token=5,
    )


def test_root_valid_tokens(simple_lexicon: LexiconConstraint) -> None:
    """At the start of decoding we should be able to emit the first phoneme of any word."""

    valid = simple_lexicon.get_valid_next_tokens([])
    assert valid == {1, 4, 5}, "Root should allow any word start plus optional boundary/silence token"


def test_partial_word_valid_tokens(simple_lexicon: LexiconConstraint) -> None:
    """After emitting the first phoneme, both continuations remain available."""

    valid = simple_lexicon.get_valid_next_tokens([1])
    assert valid == {2, 3}, "Prefix '1' can only continue to '2' or '3'"


def test_word_boundary_resets_to_root(simple_lexicon: LexiconConstraint) -> None:
    """Emitting the boundary token should return the trie cursor to the root."""

    valid = simple_lexicon.get_valid_next_tokens([1, 2, 5])
    assert valid == {1, 4, 5}, "After a completed word we should be back at the root choices"


def test_invalid_prefix_returns_empty(simple_lexicon: LexiconConstraint) -> None:
    """Unknown prefixes should have no valid continuations."""

    valid = simple_lexicon.get_valid_next_tokens([9])
    assert valid == set(), "Non-existent tokens must yield an empty continuation set"


def test_constraint_mask_matches_expected(simple_lexicon: LexiconConstraint) -> None:
    """get_constraint_mask should agree with get_valid_next_tokens for simple beams."""

    # Build a fake hypothesis buffer representing two beams.
    sequences = torch.full((1, 2, 4), fill_value=-1, dtype=torch.long)
    sequences[0, 0, :3] = torch.tensor([1, 2, 5])  # Completed word, back at root
    sequences[0, 1, 0] = 4  # In the middle of the word "D"

    last_labels = torch.tensor([[5, 4]])
    mask = simple_lexicon.get_constraint_mask(sequences=sequences, last_labels=last_labels, vocab_size=6)

    # Beam 0 is back at the root, so it should allow {1, 4, 5} plus blank.
    expected_tokens_beam0 = {0, 1, 4, 5}
    actual_beam0 = set(torch.nonzero(mask[0, 0], as_tuple=False).flatten().tolist())
    assert actual_beam0 == expected_tokens_beam0

    # Beam 1 is inside the word "D", so only the boundary token (5) and blank should be allowed.
    expected_tokens_beam1 = {0, 5}
    actual_beam1 = set(torch.nonzero(mask[0, 1], as_tuple=False).flatten().tolist())
    assert actual_beam1 == expected_tokens_beam1
