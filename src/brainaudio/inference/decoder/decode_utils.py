# Copyright (c) 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for decoding CTC outputs and working with lexicons."""

from pathlib import Path
from typing import Dict, Tuple


def apply_ctc_rules(ids):
    """
    Apply CTC rules: remove blanks (0) and merge consecutive repeats.
    
    Args:
        ids: Sequence of token IDs (tensor or list)
        
    Returns:
        List of token IDs with blanks removed and consecutive repeats merged
    """
    if hasattr(ids, 'cpu'):
        ids = ids.cpu().numpy()
    
    clean_ids = []
    prev_id = None
    
    for id_val in ids:
        if id_val == 0:  # Skip blank
            prev_id = None
            continue
        if id_val == prev_id:  # Skip repeats
            continue
        clean_ids.append(int(id_val))
        prev_id = id_val
    
    return clean_ids


def load_token_to_phoneme_mapping(tokens_file: Path) -> Dict[int, str]:
    """
    Load token ID -> phoneme symbol mapping.
    
    Args:
        tokens_file: Path to tokens.txt file (one token per line)
        
    Returns:
        Dictionary mapping token ID (line index) to phoneme symbol
    """
    token_to_symbol = {}
    with open(tokens_file, 'r') as f:
        for idx, line in enumerate(f):
            token_to_symbol[idx] = line.strip()
    return token_to_symbol


def load_phoneme_to_word_mapping(lexicon_file: Path) -> Dict[Tuple[str, ...], str]:
    """
    Build phoneme sequence -> word mapping from lexicon file.
    
    Args:
        lexicon_file: Path to lexicon.txt file (format: "word phoneme1 phoneme2 | ...")
        
    Returns:
        Dictionary mapping phoneme tuple to word string
    """
    phoneme_to_word = {}
    with open(lexicon_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            phonemes = tuple(p for p in parts[1:] if p != '|')
            phoneme_to_word[phonemes] = word
    return phoneme_to_word


def compute_wer(hypothesis: str, reference: str) -> float:
    """
    Compute Word Error Rate (WER) using Levenshtein distance.
    
    Args:
        hypothesis: Decoded text string
        reference: Ground truth text string
        
    Returns:
        WER as a float (0.0 = perfect match, 1.0 = no words match)
        
    Example:
        >>> compute_wer("hello world", "hello there")
        0.5  # 1 substitution out of 2 words
    """
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()
    
    # Levenshtein distance on words
    d = [[0] * (len(ref_words) + 1) for _ in range(len(hyp_words) + 1)]
    
    for i in range(len(hyp_words) + 1):
        d[i][0] = i
    for j in range(len(ref_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(hyp_words) + 1):
        for j in range(1, len(ref_words) + 1):
            if hyp_words[i-1] == ref_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return d[len(hyp_words)][len(ref_words)] / max(len(ref_words), 1)
