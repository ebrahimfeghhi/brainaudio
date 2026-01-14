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

"""Simplified LLM rescoring for CTC beam search decoding."""

from typing import List, TYPE_CHECKING
import torch
import torch.nn.functional as F
import truecase

if TYPE_CHECKING:
    from .batched_beam_decoding_utils import BatchedBeamHyps
    from .neural_lm_fusion import HuggingFaceLMFusion
    from .word_ngram_lm_optimized_v2 import WordHistory
    
truecase_enabled = True


@torch.no_grad()
def score_texts_batch(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
) -> List[float]:
    """
    Compute log probability of each text in a single batched forward pass.

    Args:
        model: HuggingFace causal LM model
        tokenizer: Corresponding tokenizer
        texts: List of text strings to score
        device: Device for inference

    Returns:
        List of log probabilities (one per text)
    """
    if not texts:
        return []

    # Tokenize all texts with padding
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    ).to(device)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask)

    # Shift logits and labels for next-token prediction
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs for actual tokens
    gathered_probs = torch.gather(
        log_probs,
        dim=2,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask out padding positions (shifted attention mask)
    shift_mask = attention_mask[..., 1:].contiguous()
    gathered_probs = gathered_probs * shift_mask

    # Sum log probs per sequence
    scores = gathered_probs.sum(dim=-1).tolist()

    return scores


def apply_llm_rescoring_full(
    lm_fusion: "HuggingFaceLMFusion | None",
    word_history: "WordHistory",
    beam_hyps: "BatchedBeamHyps",
) -> None:
    """
    Rescore all beams by computing LLM log prob of full text.

    Replaces context_texts score with: alpha * llm_score + beta * num_words

    Args:
        lm_fusion: The HuggingFaceLMFusion module (or None to skip)
        word_history: WordHistory object to reconstruct text from history_id
        beam_hyps: The beam hypotheses to update in-place
    """
    if lm_fusion is None:
        return

    batch_size, beam_size = beam_hyps.scores.shape

    # Collect all texts to score and save old best scores
    texts_to_score: List[str] = []
    mapping: List[tuple] = []  # (batch_idx, beam_idx, tuple_idx, num_words)
    old_best_scores: dict = {}  # (b, k) -> old best score before update

    for b in range(batch_size):
        for k in range(beam_size):
            # Skip invalid beams
            if beam_hyps.scores[b, k] == float('-inf'):
                continue

            context_tuples = beam_hyps.context_texts[b][k]

            # Save old best score BEFORE any updates
            if context_tuples and (b, k) not in old_best_scores:
                old_best_scores[(b, k)] = context_tuples[0][0]

            for tuple_idx, tup in enumerate(context_tuples):
                # Handle both tuple formats:
                # - Initial/legacy: (score, text) - 2 elements
                # - After N-gram LM: (score, lm_state_id, history_id) - 3 elements
                if len(tup) == 2:
                    # Legacy format - skip (no history_id available)
                    continue

                history_id = tup[2]
                if history_id < 0:
                    # Empty/root node
                    continue

                text = word_history.get_text(history_id)

                if not text:
                    continue
                
                if truecase_enabled:
                    # Apply truecasing
                    text = truecase.get_true_case(text)
                else:
                    # Capitalize first word for proper LLM scoring
                    text = text[0].upper() + text[1:] if text else text

                num_words = len(text.split())
                texts_to_score.append(text)
                mapping.append((b, k, tuple_idx, num_words))

    if not texts_to_score:
        return

    # Batch score all texts in one LLM call
    lm_fusion.llm_call_count += 1
    raw_scores = score_texts_batch(
        model=lm_fusion.model,
        tokenizer=lm_fusion.tokenizer,
        texts=texts_to_score,
        device=lm_fusion.device,
    )

    # Get alpha (weight) and beta (word_insertion_bonus) from lm_fusion
    alpha = lm_fusion.weight
    beta = lm_fusion.word_insertion_bonus

    # Track which beams were updated for re-sorting
    updated_beams = set()

    # Apply scores back to context_texts
    for idx, (b, k, tuple_idx, num_words) in enumerate(mapping):
        llm_score = raw_scores[idx]
        new_score = alpha * llm_score + beta * num_words

        # Replace score in tuple (keep lm_state_id and history_id)
        old_tuple = beam_hyps.context_texts[b][k][tuple_idx]
        beam_hyps.context_texts[b][k][tuple_idx] = (new_score, old_tuple[1], old_tuple[2])
        updated_beams.add((b, k))

    # Re-sort tuples and update beam scores
    for b, k in updated_beams:
        context_tuples = beam_hyps.context_texts[b][k]

        # Sort by score descending
        context_tuples.sort(key=lambda x: x[0], reverse=True)

        # Update beam score: subtract old best, add new best
        old_best_score = old_best_scores[(b, k)]
        new_best_score = context_tuples[0][0]
        beam_hyps.scores[b, k] += (new_best_score - old_best_score)


def apply_llm_eos_scoring(
    lm_fusion: "HuggingFaceLMFusion | None",
    word_history: "WordHistory",
    beam_hyps: "BatchedBeamHyps",
) -> None:
    """
    Add end-of-sentence scoring by appending punctuation.

    For each beam, scores text + ".", "?", "!" and picks the best.
    Registers the punctuation in word_history so final output includes it.

    Args:
        lm_fusion: The HuggingFaceLMFusion module (or None to skip)
        word_history: WordHistory object for text reconstruction and registration
        beam_hyps: The beam hypotheses to update in-place
    """
    if lm_fusion is None:
        return

    batch_size, beam_size = beam_hyps.scores.shape
    eos_candidates = [".", "?", "!"]

    # Collect beams that need EOS scoring
    texts_to_score: List[str] = []
    mapping: List[tuple] = []  # (b, k, tuple_idx, base_text, history_id)

    for b in range(batch_size):
        for k in range(beam_size):
            if beam_hyps.scores[b, k] == float('-inf'):
                continue

            context_tuples = beam_hyps.context_texts[b][k]
            for tuple_idx, tup in enumerate(context_tuples):
                # Handle 3-tuple format only
                if len(tup) != 3:
                    continue

                history_id = tup[2]
                if history_id < 0:
                    continue

                text = word_history.get_text(history_id)
                if not text:
                    continue

                # Skip if already ends with punctuation
                if text.rstrip() and text.rstrip()[-1] in '.?!':
                    continue

                # Capitalize for LLM scoring
                text_cap = text[0].upper() + text[1:] if text else text

                # Score text + each punctuation candidate
                for punct in eos_candidates:
                    texts_to_score.append(text_cap + punct)
                    mapping.append((b, k, tuple_idx, text, history_id, punct))

    if not texts_to_score:
        return

    # Batch score all texts
    lm_fusion.llm_call_count += 1
    raw_scores = score_texts_batch(
        model=lm_fusion.model,
        tokenizer=lm_fusion.tokenizer,
        texts=texts_to_score,
        device=lm_fusion.device,
    )

    alpha = lm_fusion.weight
    beta = lm_fusion.word_insertion_bonus

    # Group scores by (b, k, tuple_idx) and find best punctuation
    scores_by_tuple: dict = {}  # (b, k, tuple_idx) -> [(punct, score), ...]
    for idx, (b, k, tuple_idx, text, history_id, punct) in enumerate(mapping):
        key = (b, k, tuple_idx)
        if key not in scores_by_tuple:
            scores_by_tuple[key] = {'history_id': history_id, 'options': []}

        # Score for full text with punct (no word insertion bonus for punct)
        num_words = len(text.split())
        full_score = alpha * raw_scores[idx] + beta * num_words
        scores_by_tuple[key]['options'].append((punct, full_score))

    # Save old best scores and update tuples
    old_best_scores: dict = {}
    updated_beams = set()

    for (b, k, tuple_idx), data in scores_by_tuple.items():
        # Find best punctuation
        best_punct, best_score = max(data['options'], key=lambda x: x[1])
        history_id = data['history_id']

        # Register punctuation in word_history
        new_history_id = word_history.add(history_id, best_punct)

        # Save old best score if not already saved
        if (b, k) not in old_best_scores:
            old_best_scores[(b, k)] = beam_hyps.context_texts[b][k][0][0]

        # Update tuple with new score and history_id
        old_tuple = beam_hyps.context_texts[b][k][tuple_idx]
        beam_hyps.context_texts[b][k][tuple_idx] = (best_score, old_tuple[1], new_history_id)
        updated_beams.add((b, k))

    # Re-sort and update beam scores
    for b, k in updated_beams:
        context_tuples = beam_hyps.context_texts[b][k]
        context_tuples.sort(key=lambda x: x[0], reverse=True)

        old_best = old_best_scores[(b, k)]
        new_best = context_tuples[0][0]
        beam_hyps.scores[b, k] += (new_best - old_best)
