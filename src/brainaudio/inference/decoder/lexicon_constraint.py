# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Tuple

import torch


class LexiconConstraint:
    """
    Lexicon constraint for CTC beam search decoding.
    Constrains the beam search to only produce sequences that are valid according to a provided lexicon.
    
    This implementation uses a prefix tree (trie) structure to efficiently determine which tokens
    can legally follow a given sequence prefix. Optimized for speed with caching and batch operations.
    
    For LM rescoring: The trie allows multiple words to share the same phoneme sequence (homophones).
    During beam search, all valid phoneme sequences are allowed. After decoding, use
    `get_word_alternatives()` to get all possible word interpretations for LM rescoring.
    """

    def __init__(
        self,
        lexicon: List[List[int]],
        blank_index: int,
        device: torch.device = None,
        word_list: List[str] = None,
    ):
        """
        Initialize the lexicon constraint.
        
        Note: Prefer using the factory methods `from_files()` or `from_file_paths()` 
        to load from tokens.txt and lexicon.txt files.
        
        Args:
            lexicon: List of valid sequences, where each sequence is a list of token IDs.
                     Example: [[1, 2, 3], [1, 2, 4], [5, 6]] for words "cat", "car", "dog"
            blank_index: The index of the blank token in the vocabulary.
            device: The device on which to store constraint data.
            word_list: Optional list of words corresponding to lexicon entries (for homophone tracking).
        """
        self.blank_index = blank_index
        self.device = device
        self.word_list = word_list
        
        # Build the prefix tree with word tracking
        self.trie = self._build_trie(lexicon)
        
        # Extract all valid tokens that appear anywhere in the lexicon
        self.all_valid_tokens = self._get_all_valid_tokens()
        
        # Cache for fast lookups
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    @classmethod
    def from_file_paths(
        cls,
        tokens_file: Union[str, Path],
        lexicon_file: Union[str, Path],
        device: torch.device = None,
    ) -> 'LexiconConstraint':
        """
        Create a LexiconConstraint from tokens.txt and lexicon.txt file paths.
        
        Args:
            tokens_file: Path to tokens.txt file. Each line contains a token symbol.
                        Line index corresponds to token ID.
                        Example:
                            <blank>
                            a
                            b
                            c
                            .
                            .
                            .
                            SIL
            lexicon_file: Path to lexicon.txt file. Each line is: word followed by space-separated tokens and SIL (|) token. 
                         Example:
                            hello h e l l o |
                            world w o r l d |
            device: The device on which to store constraint data.
            
        Returns:
            LexiconConstraint instance with homophone tracking enabled.
        """
        tokens_file = Path(tokens_file)
        lexicon_file = Path(lexicon_file)
        
        # Load tokens
        token_to_id, blank_index = cls._load_tokens_file(tokens_file)
        
        # Load lexicon with word tracking for homophones
        lexicon_sequences, word_list = cls._load_lexicon_file_with_words(lexicon_file, token_to_id)
        
        return cls(
            lexicon=lexicon_sequences,
            blank_index=blank_index,
            device=device,
            word_list=word_list,
        )
    
    @staticmethod
    def _load_tokens_file(tokens_file: Path) -> tuple[Dict[str, int], int]:
        """
        Load tokens.txt file.
        
        Args:
            tokens_file: Path to tokens.txt
            
        Returns:
            Tuple of (token_to_id dict, blank_index)
            token_to_id maps token string (key) to its ID (value, line index).
            blank_index is an int corresponding to the blank token index.
        """
        token_to_id = {}
        
        with open(tokens_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                if not token:  # Skip empty lines
                    continue
                    
                token_to_id[token] = idx
        
        # Always assume blank token is at index 0
        blank_index = 0
        
        return token_to_id, blank_index
    
    @staticmethod
    def _load_lexicon_file_with_words(lexicon_file: Path, token_to_id: Dict[str, int]) -> tuple[List[List[int]], List[str]]:
        """
        Load lexicon.txt file with word tracking for homophone support.
        
        Args:
            lexicon_file: Path to lexicon.txt
            token_to_id: Mapping from token string to token ID
            
        Returns:
            Tuple of (lexicon_sequences, word_list) where indices correspond
            lexicon_sequences is a list containing token Id sequences for each word.
            word_list is a list of words corresponding to each lexicon entry in lexicon_sequences.
        """
        lexicon_sequences = []
        word_list = []
        
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                word = parts[0]
                token_strings = parts[1:]
                
                if len(token_strings) == 0:
                    continue
                
                # Convert token strings to IDs
                try:
                    token_ids = [token_to_id[token] for token in token_strings]
                    lexicon_sequences.append(token_ids)
                    word_list.append(word)
                except KeyError:
                    continue
        
        return lexicon_sequences, word_list
        
    def _build_trie(self, lexicon: List[List[int]]) -> Dict:
        """
        Build a prefix tree (trie) from the lexicon with word index tracking.
        
        Structure: {token_id: {next_token_id: {...}, ...}, '__end__': [word_idx1, word_idx2, ...]}
        The '__end__' key maps to list of word indices (for homophones).
        """
        trie = {}
        
        for word_idx, sequence in enumerate(lexicon):
            # node points to the same memory location as trie at start of each word
            node = trie
            for token in sequence:
                # if token is not in trie, start a new branch
                if token not in node:
                    node[token] = {}
                # otherwise enter existing branch
                node = node[token]
            # Track which word(s) end here (supports homophones)
            if '__end__' not in node:
                node['__end__'] = []
            # add the word index to the list of words ending here
            if self.word_list:
                node['__end__'].append(word_idx)
            else:
                node['__end__'] = True
                
        return trie
    
    def _get_all_valid_tokens(self) -> Set[int]:
        """Extract all token IDs that appear anywhere in the lexicon."""
        valid_tokens = set()
        
        def traverse(node):
            for key in node:
                if key != '__end__':
                    valid_tokens.add(key)
                    traverse(node[key])
        
        traverse(self.trie)
        return valid_tokens
    
    def get_valid_next_tokens(self, sequence: List[int]) -> Set[int]:
        """
        Get the set of valid tokens that can follow the given sequence (with caching).
        
        The lexicon trie contains individual words. When processing multi-word sequences,
        we need to track where we are in the CURRENT word being decoded. After completing
        a word (reaching '__end__'), we restart from the trie root for the next word.
        
        Args:
            sequence: The current full sequence (list of token IDs), potentially containing multiple words.
            
        Returns:
            Set of valid next token IDs that can extend this sequence.
        """
        valid_tokens, _, _ = self.get_valid_next_tokens_with_word_info(sequence)
        return valid_tokens
    
    def get_valid_next_tokens_with_word_info(self, sequence: List[int]) -> Tuple[Set[int], bool, List[int]]:
        """
        Get valid next tokens AND information about word boundaries for LM fusion.
        
        Args:
            sequence: Current token sequence
            
        Returns:
            Tuple of:
                - Set of valid next token IDs
                - Bool indicating if current position is at a word boundary (just completed a word)
                - List of word indices that just completed (for homophones)
        """
        # Check cache first to see if the next valid tokens for this sequence have been computed
        cache_key = tuple(sequence)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        # Track current position and word completion
        node = self.trie
        word_indices = []
        at_word_boundary = False
        
        for token in sequence:
            if token in node:
                node = node[token]
                # Check if we completed a word
                if '__end__' in node:
                    at_word_boundary = True
                    if isinstance(node['__end__'], list):
                        word_indices = node['__end__'].copy()
                    node = self.trie  # Reset for next word
            else:
                result = (set(), False, [])
                self._cache[cache_key] = result
                return result
        
        """
        If a word was completed, node is back at root and valid tokens are all tokens starting new words.
        If still within a word, valid tokens are those continuing the current branch.
        """
        valid_tokens = set()
        for key in node:
            if key != '__end__':
                valid_tokens.add(key)
        
        result = (valid_tokens, at_word_boundary, word_indices)
        self._cache[cache_key] = result
        return result
    
    def get_word_alternatives(self, phoneme_sequence: List[int], token_to_symbol: Dict[int, str] = None) -> List[str]:
        
        """
        Get all possible words that match a phoneme sequence (for LM rescoring of homophones).
        
        Args:
            phoneme_sequence: Sequence of token IDs representing phonemes
            token_to_symbol: Optional mapping for debugging
            
        Returns:
            List of word strings that match this phoneme sequence.
            Returns empty list if sequence is invalid or no words match.
        """
        if not self.word_list:
            return []
        
        # Navigate to the end of this sequence
        node = self.trie
        for token in phoneme_sequence:
            if token in node:
                node = node[token]
            else:
                return []
        
        # Check if this is a valid word ending
        if '__end__' in node and isinstance(node['__end__'], list):
            # Return all words that end here (homophones)
            return [self.word_list[idx] for idx in node['__end__']]
        
        return []
    
    def decode_sequence_to_words(self, token_ids: List[int], token_to_symbol: Dict[int, str], lexicon_word_map: Dict[tuple, str] = None, return_alternatives: bool = False) -> Union[str, List[tuple[str, List[str]]]]:
        """
        Decode a sequence of token IDs to words using the lexicon.
        
        This method splits the sequence by word boundaries and looks up each word segment.
        Supports returning all alternative words (homophones) for LM rescoring.
        
        Args:
            token_ids: List of token IDs (after CTC rules applied - blanks removed, repeats merged)
            token_to_symbol: Mapping from token ID to symbol string
            lexicon_word_map: Optional mapping from (phoneme_tuple) to word string (for single best word).
                             The tuple should NOT include the word boundary token (e.g., "|")
            return_alternatives: If True, returns list of (chosen_word, alternative_words_list) tuples
                                for LM rescoring. If False, returns space-separated string.
            
        Returns:
            If return_alternatives=False: Space-separated string of words
            If return_alternatives=True: List of tuples (word, alternatives) where alternatives
                                        includes all homophones for that position.
        """
        words = []
        word_alternatives = []
        current_word_tokens = []
        
        node = self.trie
        for token_id in token_ids:
            if token_id in node:
                current_word_tokens.append(token_id)
                node = node[token_id]
                
                # Check if word complete
                if '__end__' in node:
                    # Get all alternative words for this phoneme sequence (homophones)
                    alternatives = self.get_word_alternatives(current_word_tokens, token_to_symbol)
                    
                    # Convert token IDs to symbols for fallback
                    word_symbols = [token_to_symbol.get(t, f'?{t}') for t in current_word_tokens]
                    word_symbols_for_lookup = [s for s in word_symbols if s != '|']
                    word_phonemes_tuple = tuple(word_symbols_for_lookup)
                    
                    # Choose primary word
                    if alternatives:
                        # Use first alternative as primary (or from lexicon_word_map if provided)
                        if lexicon_word_map and word_phonemes_tuple in lexicon_word_map:
                            primary_word = lexicon_word_map[word_phonemes_tuple]
                        else:
                            primary_word = alternatives[0]
                    else:
                        primary_word = ' '.join(word_symbols)
                        alternatives = []
                    
                    words.append(primary_word)
                    word_alternatives.append((primary_word, alternatives if len(alternatives) > 1 else []))
                    
                    current_word_tokens = []
                    node = self.trie
            else:
                # Invalid token - add as unknown
                if current_word_tokens:
                    word_phonemes = tuple(token_to_symbol.get(t, f'?{t}') for t in current_word_tokens)
                    unk_word = f'<UNK:{" ".join(word_phonemes)}>'
                    words.append(unk_word)
                    word_alternatives.append((unk_word, []))
                    current_word_tokens = []
                unk_word = f'<UNK:{token_to_symbol.get(token_id, token_id)}>'
                words.append(unk_word)
                word_alternatives.append((unk_word, []))
                node = self.trie
        
        # Handle remaining tokens
        if current_word_tokens:
            word_phonemes = tuple(token_to_symbol.get(t, f'?{t}') for t in current_word_tokens)
            partial_word = f'<PARTIAL:{" ".join(word_phonemes)}>'
            words.append(partial_word)
            word_alternatives.append((partial_word, []))
        
        if return_alternatives:
            return word_alternatives
        return ' '.join(words)
    
    def get_constraint_mask(
        self,
        sequences: torch.Tensor,
        last_labels: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """
        Generate a boolean mask indicating which tokens are valid for each beam (optimized).
        
        This is the main function called during beam search to apply lexicon constraints.
        Uses vectorization and caching for speed.
        
        Args:
            sequences: [B, beam_size, seq_len] - current decoded sequences
                       (may contain NON_EXISTENT_LABEL_VALUE=-1 for padding)
            last_labels: [B, beam_size] - last emitted non-blank token for each beam
            vocab_size: Total vocabulary size (including blank)
            
        Returns:
            mask: [B, beam_size, vocab_size] - True for valid tokens, False otherwise
                  Blank is always True (blanks don't advance the sequence)
        """
        B, beam_size, seq_len = sequences.shape
        
        # Pre-allocate mask on GPU
        mask = torch.zeros(B, beam_size, vocab_size, dtype=torch.bool, device=self.device)
        
        # Blank token is always allowed (vectorized)
        mask[:, :, self.blank_index] = True
        
        # Flatten for vectorized processing
        sequences_flat = sequences.view(B * beam_size, seq_len)
        
        # Process all beams (still some CPU overhead, but batched)
        for idx in range(B * beam_size):
            seq = sequences_flat[idx]
            seq_filtered = seq[seq >= 0].tolist()
            
            # Query trie with caching
            valid_tokens = self.get_valid_next_tokens(seq_filtered)
            
            if len(valid_tokens) == 0 and len(seq_filtered) == 0:
                valid_tokens = self.all_valid_tokens
            
            # Compute batch/beam indices
            b = idx // beam_size
            k = idx % beam_size
            
            # Vectorized mask update
            if valid_tokens:
                valid_tokens_list = [t for t in valid_tokens if t < vocab_size]
                if valid_tokens_list:
                    mask[b, k, valid_tokens_list] = True
        
        return mask
    
    def clear_cache(self):
        """Clear the constraint cache and print statistics."""
        total = self._cache_hits + self._cache_misses
        if total > 0:
            hit_rate = 100.0 * self._cache_hits / total
            print(f"Lexicon cache: {len(self._cache)} entries, {hit_rate:.1f}% hit rate ({self._cache_hits}/{total})")
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

