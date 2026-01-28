# Copyright (c) 2025
# Licensed under the Apache License, Version 2.0

"""
Optimized LLM rescoring with Fused Cross Entropy and Deduplication.
"""

from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, TYPE_CHECKING
import re

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .batched_beam_decoding_utils import BatchedBeamHyps
    from .word_ngram_lm_optimized import WordHistory
    from .vectorized_lexicon_constraint import VectorizedLexiconConstraint


def caps_text(text):
    if not text:
        return text

    # 1. Capitalize the first letter of the sentence (Your original logic)
    text = text[0].upper() + text[1:]

    # 2. Capitalize "i" and its contractions
    # The \b ensures we only match whole words (so "idiom" isn't affected)
    # We look for: i, i'm, i'll, i'd, and i've
    pattern = r"\b(i|i'm|i'll|i'd|i've)\b"
    
    # re.sub finds the pattern and capitalizes the match
    text = re.sub(pattern, lambda match: match.group(0).capitalize(), text)

    return text

# Global pre-compiled regex (Optimization #2)
I_PATTERN = re.compile(r"\b(i|i'm|i'll|i'd|i've)\b")

def caps_text(text):
    if not text:
        return text
    # Capitalize first char
    text = text[0].upper() + text[1:]
    # Use pre-compiled regex
    return I_PATTERN.sub(lambda match: match.group(0).capitalize(), text)


@dataclass
class LLMRescorer:
    """Lightweight container for LLM rescoring configuration."""
    model: Any
    tokenizer: Any
    device: torch.device
    llm_weight: float = 1.0
    ngram_weight: float = 0.0
    scoring_chunk_size: int = 256
    llm_call_count: int = field(default=0, repr=False)


@torch.no_grad()
def score_texts_batch(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    chunk_size: int = 256,
) -> List[float]:
    """
    Compute log probability using Fused Cross Entropy (Memory Optimized).

    Instead of calculating the full (B, T, V) log_softmax tensor,
    we use F.cross_entropy which runs a fused kernel and only outputs (B, T).
    This reduces VRAM usage by ~100x for large vocabularies.
    """
    if not texts:
        return []

    model.eval()

    # Sort by length to minimize padding waste (speeds up batching)
    sorted_indices = sorted(range(len(texts)), key=lambda k: len(texts[k]))
    sorted_texts = [texts[i] for i in sorted_indices]

    all_scores_sorted = []

    for chunk_start in range(0, len(sorted_texts), chunk_size):
        chunk_texts = sorted_texts[chunk_start : chunk_start + chunk_size]

        inputs = tokenizer(
            chunk_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True, # Adds BOS
        ).to(device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # --- OPTIMIZATION START ---
        # Fused calculation: calculates softmax + nll_loss in one go without
        # materializing the huge (Batch, Seq, Vocab) probability tensor.

        # Flatten to (N, C) for cross_entropy API
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        # reduction='none' gives loss per token
        token_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction='none',
            ignore_index=tokenizer.pad_token_id
        )

        # Reshape back to (Batch, Seq)
        token_losses = token_losses.view(shift_labels.shape)

        # Convert Loss (positive) to LogProb (negative)
        # We explicitly mask again to be safe, though ignore_index handles values
        shift_mask = attention_mask[..., 1:].contiguous()
        seq_log_probs = -(token_losses * shift_mask).sum(dim=1)
        # --- OPTIMIZATION END ---

        all_scores_sorted.extend(seq_log_probs.tolist())

        del outputs, logits, shift_logits, shift_labels, token_losses
        del inputs, input_ids, attention_mask
        torch.cuda.empty_cache()

    # Restore original order
    final_scores = [0.0] * len(texts)
    for original_idx, score in zip(sorted_indices, all_scores_sorted):
        final_scores[original_idx] = score

    return final_scores


def apply_llm_rescoring_full(
    lm_fusion: "LLMRescorer | None",
    word_history: "WordHistory",
    beam_hyps: "BatchedBeamHyps",
) -> None:
    """
    Rescore beams with DEDUPLICATION.

    1. Collects all unique text strings across all beams.
    2. Scores each unique string once.
    3. Broadcasts the score back to all requesting beams.
    """
    if lm_fusion is None:
        return

    batch_size, beam_size = beam_hyps.scores.shape

    # --- PHASE 1: Deduplication ---
    # Map: unique_text -> List of (batch_idx, beam_idx, tuple_idx)
    unique_requests: Dict[str, List[Tuple[int, int, int]]] = {}

    # Store old best scores to calculate the final beam update delta
    old_best_scores: Dict[Tuple[int, int], float] = {}

    for b in range(batch_size):
        for k in range(beam_size):
            if beam_hyps.scores[b, k] == float('-inf'):
                continue

            context_tuples = beam_hyps.context_texts[b][k]

            # Save the current best score (usually index 0) for the delta update later
            if context_tuples:
                 old_best_scores[(b, k)] = context_tuples[0][0]

            for tuple_idx, tup in enumerate(context_tuples):
                # Format: (score, lm_state_id, history_id)
                if len(tup) != 3: continue

                history_id = tup[2]
                if history_id < 0: continue

                text = word_history.get_text(history_id)
                if not text: continue
     
                # Capitalize first letter
                text = caps_text(text)
                
                # Register request
                if text not in unique_requests:
                    unique_requests[text] = []

                unique_requests[text].append((b, k, tuple_idx))

    if not unique_requests:
        return

    # --- PHASE 2: Batched Scoring ---
    unique_texts = list(unique_requests.keys())

    lm_fusion.llm_call_count += 1

    unique_scores = score_texts_batch(
        model=lm_fusion.model,
        tokenizer=lm_fusion.tokenizer,
        texts=unique_texts,
        device=lm_fusion.device,
        chunk_size=lm_fusion.scoring_chunk_size,
    )

    llm_weight = lm_fusion.llm_weight
    ngram_weight = lm_fusion.ngram_weight

    # --- PHASE 3: Broadcast Updates ---
    updated_beams = set()

    for text, llm_score in zip(unique_texts, unique_scores):

        # Retrieve everyone who asked for this text
        requests = unique_requests[text]

        for (b, k, tuple_idx) in requests:
            # Update tuple in place
            old_tuple = beam_hyps.context_texts[b][k][tuple_idx]
            old_score = old_tuple[0]
            new_score = ngram_weight * old_score + llm_weight * llm_score
            beam_hyps.context_texts[b][k][tuple_idx] = (new_score, old_tuple[1], old_tuple[2])

            updated_beams.add((b, k))

    # --- PHASE 4: Sort & Finalize ---
    for b, k in updated_beams:
        context_tuples = beam_hyps.context_texts[b][k]

        # Sort by new scores
        context_tuples.sort(key=lambda x: x[0], reverse=True)

        # Update beam score based on the change in the BEST path
        old_best = old_best_scores[(b, k)]
        new_best = context_tuples[0][0]

    
        beam_hyps.scores[b, k] += (new_best - old_best)


def apply_llm_eos_scoring(
    lm_fusion: "LLMRescorer | None",
    word_history: "WordHistory",
    beam_hyps: "BatchedBeamHyps",
) -> None:
    """
    Scores EOS candidates with deduplication.
    Prevents scoring "The cat." 50 times if 50 beams have it.
    """
    if lm_fusion is None:
        return

    batch_size, beam_size = beam_hyps.scores.shape
    eos_candidates = [".", "?", "!"]

    # Map: unique_punct_text -> List of (b, k, tuple_idx, history_id, punct_char)
    unique_requests: Dict[str, List[Tuple[int, int, int, int, str]]] = {}

    for b in range(batch_size):
        for k in range(beam_size):
            if beam_hyps.scores[b, k] == float('-inf'): continue

            context_tuples = beam_hyps.context_texts[b][k]
            for tuple_idx, tup in enumerate(context_tuples):
                if len(tup) != 3: continue

                history_id = tup[2]
                if history_id < 0: continue

                text = word_history.get_text(history_id)
                if not text: continue
                if text.rstrip() and text.rstrip()[-1] in '.?!': continue

                # Capitalize
                text_cap = text[0].upper() + text[1:] if text else text

                # Generate candidates
                for punct in eos_candidates:
                    full_text = text_cap + punct

                    if full_text not in unique_requests:
                        unique_requests[full_text] = []

                    unique_requests[full_text].append((b, k, tuple_idx, history_id, punct))

    if not unique_requests:
        return

    # Batch Score
    unique_texts = list(unique_requests.keys())
    unique_scores = score_texts_batch(
        lm_fusion.model, lm_fusion.tokenizer, unique_texts, lm_fusion.device, lm_fusion.scoring_chunk_size
    )

    llm_weight = lm_fusion.llm_weight
    ngram_weight = lm_fusion.ngram_weight

    # Organize results: (b, k, tuple_idx) -> List[(score, punct)]
    results_by_tuple: Dict[Tuple[int, int, int], List[Tuple[float, str]]] = {}

    for text, llm_score in zip(unique_texts, unique_scores):
        final_score = llm_weight * llm_score

        for (b, k, tuple_idx, _, punct) in unique_requests[text]:
            key = (b, k, tuple_idx)
            if key not in results_by_tuple:
                results_by_tuple[key] = []
            results_by_tuple[key].append((final_score, punct))

    # Update Hypotheses
    updated_beams = set()
    old_best_scores = {} # Capture on demand

    for (b, k, tuple_idx), options in results_by_tuple.items():
        # Find best punctuation
        best_score, best_punct = max(options, key=lambda x: x[0])

        if (b, k) not in old_best_scores:
            old_best_scores[(b, k)] = beam_hyps.context_texts[b][k][0][0]

        old_tup = beam_hyps.context_texts[b][k][tuple_idx]
        history_id = old_tup[2]

        # Add punct to history
        # Note: This might create duplicates in history if not handled in WordHistory,
        # but WordHistory.add() usually handles dedupe.
        new_hist_id = word_history.add(history_id, best_punct)

        # Combine N-gram and LLM scores
        old_score = old_tup[0]
        new_score = ngram_weight * old_score + best_score
        beam_hyps.context_texts[b][k][tuple_idx] = (new_score, old_tup[1], new_hist_id)
        updated_beams.add((b, k))

    # Final Sort
    for b, k in updated_beams:
        tuples = beam_hyps.context_texts[b][k]
        tuples.sort(key=lambda x: x[0], reverse=True)

        old = old_best_scores.get((b, k), tuples[0][0]) # Fallback just in case
        new = tuples[0][0]
        beam_hyps.scores[b, k] += (new - old)


def apply_llm_eos_scoring_with_pending(
    lm_fusion: "LLMRescorer | None",
    word_history: "WordHistory",
    beam_hyps: "BatchedBeamHyps",
    lexicon: "VectorizedLexiconConstraint",
    lexicon_states: "torch.Tensor",
) -> None:
    """
    Enhanced EOS scoring that also considers pending (pseudo-complete) words.

    For beams where the lexicon state is at a word-terminal (phonemes form a
    complete word but boundary token wasn't emitted), this function scores both:
    - current_text + punct (without pending word)
    - current_text + pending_word + punct (with pending word)

    And picks the best scoring option.

    Args:
        lm_fusion: LLM rescorer config
        word_history: Word history for text lookup/storage
        beam_hyps: Current beam hypotheses
        lexicon: Lexicon constraint (for get_words_at_state)
        lexicon_states: Current lexicon states [batch, beam]
    """
    if lm_fusion is None:
        return

    batch_size, beam_size = beam_hyps.scores.shape
    eos_candidates = [".", "?", "!"]

    # Strip variant suffix pattern (e.g., "record(2)" -> "record")
    variant_pattern = re.compile(r'\(\d+\)$')

    # Map: unique_text -> List of (b, k, tuple_idx, history_id, punct, pending_word_or_none)
    # pending_word_or_none is None for "without pending word" candidates, or the word string
    unique_requests: Dict[str, List[Tuple[int, int, int, int, str, str | None]]] = {}

    for b in range(batch_size):
        for k in range(beam_size):
            if beam_hyps.scores[b, k] == float('-inf'):
                continue

            context_tuples = beam_hyps.context_texts[b][k]

            # Get pending words at current lexicon state
            state = lexicon_states[b, k].item()
            pending_word_indices = lexicon.get_words_at_state(state)

            # Get unique pending words (strip variant suffixes, dedupe)
            pending_words = []
            if pending_word_indices:
                seen = set()
                for idx in pending_word_indices:
                    word = variant_pattern.sub('', lexicon.word_list[idx])
                    if word not in seen:
                        seen.add(word)
                        pending_words.append(word)

            for tuple_idx, tup in enumerate(context_tuples):
                if len(tup) != 3:
                    continue

                history_id = tup[2]
                if history_id < 0:
                    continue

                text = word_history.get_text(history_id)
                if not text:
                    continue
                if text.rstrip() and text.rstrip()[-1] in '.?!':
                    continue

                # Capitalize
                text_cap = caps_text(text)

                # Generate candidates WITHOUT pending word
                for punct in eos_candidates:
                    full_text = text_cap + punct

                    if full_text not in unique_requests:
                        unique_requests[full_text] = []
                    unique_requests[full_text].append((b, k, tuple_idx, history_id, punct, None))

                # Generate candidates WITH pending word (if any)
                for pending_word in pending_words:
                    text_with_pending = text_cap + " " + pending_word
                    text_with_pending_cap = caps_text(text_with_pending)

                    for punct in eos_candidates:
                        full_text = text_with_pending_cap + punct

                        if full_text not in unique_requests:
                            unique_requests[full_text] = []
                        unique_requests[full_text].append((b, k, tuple_idx, history_id, punct, pending_word))

    if not unique_requests:
        return

    # Batch Score all unique texts
    unique_texts = list(unique_requests.keys())
    unique_scores = score_texts_batch(
        lm_fusion.model, lm_fusion.tokenizer, unique_texts, lm_fusion.device, lm_fusion.scoring_chunk_size
    )

    llm_weight = lm_fusion.llm_weight
    ngram_weight = lm_fusion.ngram_weight

    # Organize results: (b, k, tuple_idx) -> List[(score, punct, pending_word_or_none)]
    results_by_tuple: Dict[Tuple[int, int, int], List[Tuple[float, str, str | None]]] = {}

    for text, llm_score in zip(unique_texts, unique_scores):
        final_score = llm_weight * llm_score

        for (b, k, tuple_idx, _, punct, pending_word) in unique_requests[text]:
            key = (b, k, tuple_idx)
            if key not in results_by_tuple:
                results_by_tuple[key] = []
            results_by_tuple[key].append((final_score, punct, pending_word))

    # Update Hypotheses
    updated_beams = set()
    old_best_scores = {}

    for (b, k, tuple_idx), options in results_by_tuple.items():
        # Find best option (highest score)
        best_score, best_punct, best_pending_word = max(options, key=lambda x: x[0])

        if (b, k) not in old_best_scores:
            old_best_scores[(b, k)] = beam_hyps.context_texts[b][k][0][0]

        old_tup = beam_hyps.context_texts[b][k][tuple_idx]
        history_id = old_tup[2]

        # Build new history: optionally add pending word, then add punct
        new_hist_id = history_id
        if best_pending_word is not None:
            new_hist_id = word_history.add(new_hist_id, best_pending_word)
        new_hist_id = word_history.add(new_hist_id, best_punct)

        # Combine N-gram and LLM scores
        old_score = old_tup[0]
        new_score = ngram_weight * old_score + best_score
        beam_hyps.context_texts[b][k][tuple_idx] = (new_score, old_tup[1], new_hist_id)
        updated_beams.add((b, k))

    # Final Sort
    for b, k in updated_beams:
        tuples = beam_hyps.context_texts[b][k]
        tuples.sort(key=lambda x: x[0], reverse=True)

        old = old_best_scores.get((b, k), tuples[0][0])
        new = tuples[0][0]
        beam_hyps.scores[b, k] += (new - old)
