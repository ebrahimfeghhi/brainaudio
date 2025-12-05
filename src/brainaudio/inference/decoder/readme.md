# lexicon_constraint.py

## Overview
This module implements the `LexiconConstraint` and the `VectorizedLexiconConstraint` class, which enforces linguistic constraints during the decoding of neural signals. 


## LexiconConstraint 
It utilizes a Trie-based structure to map phoneme sequences to words, handling valid path navigation, word boundary detection, and Connectionist Temporal Classification (CTC) rules.

## Data Structure: The Trie
The core of this class is built within `_build_trie`.

### `_build_trie`
Constructs the navigation structure as a nested dictionary.
* **Structure:** Top-level keys are phoneme IDs that can start words.
* **Traversal:** Each value is a nested dictionary representing possible phoneme continuations.
* **Word Boundaries:**
    * All words must pass through the **silence token (index 40)** to reach a valid endpoint.
    * Valid endpoints are marked by an `"end"` key.
* **Homophones:** The value of the `"end"` key is a **list of word indices**. This list handles homophones, where a single phoneme sequence maps to multiple distinct words.

---

## Core Methods

### `_get_valid_tokens`
Evaluates a specific token sequence to determine valid next steps and word completions.

* **Inputs:** A sequence of tokens.
* **Returns:**
    1.  `valid_tokens`: A set of valid next token IDs.
    2.  `is_word_boundary`: Boolean indicating if the current position completes a word.
    3.  `word_indices`: A list of word indices completed (if at a boundary).
* **Logic:**
    1.  Loops through the token sequence.
    2.  **Invalid Path:** If a token is not in the current node, the path is invalid.
    3.  **Word Boundary:** If the token is the silence token (boundary), checks for the `"end"` key.
        * If found: Retrieves word indices and resets the node to the **root**.
    4.  **Return Values:** Returns only the *final* word if multiple exist in the sequence. If the final word is incomplete, `is_word_boundary` is `False`.
    5.  **CTC Handling:** If at a word boundary, the boundary token is explicitly allowed in the `valid_tokens` set to account for CTC collapse rules.

### `_get_constraint_mask`
Generates a boolean mask for batch processing, identifying valid tokens for the next step in beam search.

* **Inputs:**
    * `sequences`: Shape `(B * beam_size, seq_len)` (unraveled).
    * `last_labels`: The last emitted non-blank token for each beam.
    * `vocab_size`: Total size of the vocabulary.
* **Logic:**
    1.  Loops through each trial in the batch.
    2.  Applies **CTC merging rules** to the sequence.
    3.  Calls `_get_valid_tokens` on the merged sequence.
    4.  **Mask Update:** Sets valid tokens to `True`.
    5.  **Repeat Token:** Explicitly sets the `last_token` entry to `True` to allow consecutive repeats (standard CTC behavior).

---

## Decoding & Analysis Tools

### `_get_word_alternatives`
* **Input:** A sequence of token IDs.
* **Behavior:** Steps through the trie following the sequence.
* **Return:** If an `"end"` key is hit, returns the list of word strings (homophones). Returns an empty list if the sequence implies no valid words.

### `_decode_sequence_to_words`
Decodes raw token IDs into a human-readable string, with optional homophone reporting.

* **Inputs:** Token IDs (post-CTC), token-to-symbol mapping, phoneme-to-word mapping, and a boolean for returning homophones.
* **Logic:**
    1.  Traverses the Trie.
    2.  **Success:** If `"end"` is found, calls `_get_word_alternatives`. The first entry is used for the decoded text; remaining entries are listed as homophones.
    3.  **Unknown:** If the token ID is not in the node (invalid path), appends `<UNK>`.
    4.  **Incomplete:** If the path is valid but no word boundary is reached yet, appends `<PARTIAL>`.


## VectorizedLexiconConstraint 
Provides a GPU friendly version of `LexiconConstraint`. 

## Data Structure: Transition Matrix 
The core of this class is built within `_build_dense_transition_table`.

### `_build_dense_transition_table`
Converts the trie into a transition table through the following sequence of steps.
    1. Store nodes in BFS order, and assign each node id to a unique integer id in BFS order. 
    2. Construct the transition table. The entry at row i, column j in the table tells us which node we should move to if we are at node i and observe token j. Each node represents a prefix in the lexicon trie. 
    
---
## Core Methods

### get_constraint_mask_with_state
Generates a boolean mask for batch processing, identifying valid tokens for the next step in beam search.
* **Inputs:**
    * `state`: Shape `(B, beam_size)`, each state represents a prefix in trie.
    * `last_labels`: The last emitted non-blank token for each beam.
    * `vocab_size`: Total size of the vocabulary.
* **Logic:**
    1. Uses the state vector to retrieve transitions from the transition table.
    2. Mask is set to True for all valid transitions, and False for invalid transitions.
    3. Additionally set mask to true so that last emitted token can be repeated.


### update_state

* **Inputs:**
    * `parent_state`: Shape `(B, beam_size)`, previous state values.
    * `emitted_labels`: Shape `(B, beam_size)`, current token for each beam. 
    * `prev_last_labels`:  Shape `(B, beam_size)`, previous token for each beam.
* **Logic:**
    1. Computes advance_mask, which is true only if emitted_labels is not a blank token or repeat.
    2. All invalid emitted_labels or emitted_labels which return an invalid state are directed to the sink state.
    

