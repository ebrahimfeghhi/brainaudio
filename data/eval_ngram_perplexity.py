#!/usr/bin/env python3
"""
Evaluate perplexity of N-gram LM (KenLM) on transcripts_all.txt.

Evaluates training and validation sets separately.
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import kenlm


# KenLM returns log10 probabilities, convert to natural log for perplexity
LOG10_TO_LN = 2.302585092994046


def load_transcripts(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Load transcripts and split into train/val based on marker comment.

    Returns:
        (train_sentences, val_sentences)
    """
    train_sentences = []
    val_sentences = []
    in_val_section = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for validation marker
            if line.startswith('#') and 'VALIDATION' in line.upper():
                in_val_section = True
                continue

            # Skip other comment lines
            if line.startswith('#'):
                continue

            if in_val_section:
                val_sentences.append(line)
            else:
                train_sentences.append(line)

    return train_sentences, val_sentences


def compute_perplexity(
    model: kenlm.Model,
    sentences: List[str],
    score_bos_eos: bool = True,
) -> Tuple[float, float, int]:
    """
    Compute perplexity using KenLM.

    Args:
        model: KenLM model
        sentences: List of sentences to evaluate
        score_bos_eos: Whether to include BOS/EOS scoring

    Returns:
        (perplexity, avg_log_prob_ln, total_words)
    """
    total_log_prob_log10 = 0.0
    total_words = 0

    for sentence in sentences:
        # Normalize: lowercase for LM lookup
        # (Many LMs are trained on lowercased text)
        sentence_lower = sentence.lower()

        # KenLM's score() returns log10 probability
        # bos=True/eos=True includes sentence boundary scoring
        log10_prob = model.score(sentence_lower, bos=score_bos_eos, eos=score_bos_eos)
        total_log_prob_log10 += log10_prob

        # Count words (for per-word perplexity)
        words = sentence_lower.split()
        word_count = len(words)
        if score_bos_eos:
            # Add 1 for </s> token
            word_count += 1
        total_words += word_count

    # Convert log10 to ln
    total_log_prob_ln = total_log_prob_log10 * LOG10_TO_LN

    # Average log probability (natural log)
    avg_log_prob_ln = total_log_prob_ln / total_words if total_words > 0 else 0

    # Perplexity = exp(-avg_log_prob)
    perplexity = math.exp(-avg_log_prob_ln)

    return perplexity, avg_log_prob_ln, total_words


def compute_perplexity_word_by_word(
    model: kenlm.Model,
    sentences: List[str],
) -> Tuple[float, float, int, int]:
    """
    Compute perplexity word-by-word for more detailed analysis.

    Returns:
        (perplexity, avg_log_prob_ln, total_words, oov_count)
    """
    total_log_prob_log10 = 0.0
    total_words = 0
    oov_count = 0

    for sentence in sentences:
        sentence_lower = sentence.lower()
        words = sentence_lower.split()

        # Initialize state
        state = kenlm.State()
        model.BeginSentenceWrite(state)

        for word in words:
            out_state = kenlm.State()
            log10_prob = model.BaseScore(state, word, out_state)
            total_log_prob_log10 += log10_prob
            total_words += 1

            # Check if OOV
            if word not in model:
                oov_count += 1

            state = out_state

        # Score EOS
        out_state = kenlm.State()
        log10_prob = model.BaseScore(state, "</s>", out_state)
        total_log_prob_log10 += log10_prob
        total_words += 1

    # Convert and compute perplexity
    total_log_prob_ln = total_log_prob_log10 * LOG10_TO_LN
    avg_log_prob_ln = total_log_prob_ln / total_words if total_words > 0 else 0
    perplexity = math.exp(-avg_log_prob_ln)

    return perplexity, avg_log_prob_ln, total_words, oov_count


def main():
    parser = argparse.ArgumentParser(description="Evaluate N-gram LM perplexity on transcripts")
    parser.add_argument(
        "--transcript-file",
        type=str,
        default="/home/ebrahim/brainaudio/data/transcripts_all.txt",
        help="Path to transcripts file",
    )
    parser.add_argument(
        "--lm-path",
        type=str,
        default="/data2/brain2text/lm/lm_dec19_huge_4gram.kenlm",
        help="Path to KenLM model",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed word-by-word analysis with OOV stats",
    )
    args = parser.parse_args()

    print(f"Loading N-gram LM: {args.lm_path}")
    model = kenlm.Model(args.lm_path)
    print(f"  Order: {model.order}")

    print(f"\nLoading transcripts: {args.transcript_file}")
    train_sentences, val_sentences = load_transcripts(args.transcript_file)
    print(f"  Train sentences: {len(train_sentences)}")
    print(f"  Val sentences: {len(val_sentences)}")

    if args.detailed:
        # Detailed word-by-word analysis
        print("\n" + "=" * 60)
        print("TRAINING SET (detailed)")
        print("=" * 60)
        train_ppl, train_avg_lp, train_words, train_oov = compute_perplexity_word_by_word(
            model, train_sentences
        )
        print(f"  Perplexity: {train_ppl:.2f}")
        print(f"  Avg Log Prob (ln): {train_avg_lp:.4f}")
        print(f"  Total Words: {train_words:,}")
        print(f"  OOV Words: {train_oov:,} ({100 * train_oov / train_words:.2f}%)")

        print("\n" + "=" * 60)
        print("VALIDATION SET (detailed)")
        print("=" * 60)
        val_ppl, val_avg_lp, val_words, val_oov = compute_perplexity_word_by_word(
            model, val_sentences
        )
        print(f"  Perplexity: {val_ppl:.2f}")
        print(f"  Avg Log Prob (ln): {val_avg_lp:.4f}")
        print(f"  Total Words: {val_words:,}")
        print(f"  OOV Words: {val_oov:,} ({100 * val_oov / val_words:.2f}%)")

    else:
        # Standard evaluation using KenLM's score() method
        print("\n" + "=" * 60)
        print("TRAINING SET")
        print("=" * 60)
        train_ppl, train_avg_lp, train_words = compute_perplexity(model, train_sentences)
        print(f"  Perplexity: {train_ppl:.2f}")
        print(f"  Avg Log Prob (ln): {train_avg_lp:.4f}")
        print(f"  Total Words: {train_words:,}")

        print("\n" + "=" * 60)
        print("VALIDATION SET")
        print("=" * 60)
        val_ppl, val_avg_lp, val_words = compute_perplexity(model, val_sentences)
        print(f"  Perplexity: {val_ppl:.2f}")
        print(f"  Avg Log Prob (ln): {val_avg_lp:.4f}")
        print(f"  Total Words: {val_words:,}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {args.lm_path}")
    print(f"Train Perplexity: {train_ppl:.2f}")
    print(f"Val Perplexity: {val_ppl:.2f}")


if __name__ == "__main__":
    main()
