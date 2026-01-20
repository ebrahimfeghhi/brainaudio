#!/usr/bin/env python3
"""
Interactive tool to query phoneme n-gram LM probabilities.

Usage:
    python creating_n_gram_lm/query_phoneme_lm.py
    python creating_n_gram_lm/query_phoneme_lm.py --lm-path /path/to/model.nemo

Enter phoneme sequences (space-separated) to see their probabilities.
Use '|' or 'SIL' for word boundaries.

Examples:
    > AY SIL AY
    > HH AH L OW
    > DH AH | K AE T
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brainaudio.inference.decoder.ngram_lm import NGramGPULanguageModel


# Phoneme vocabulary (matching units_pytorch.txt, excluding blank)
PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL'
]
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEMES)}
IDX_TO_PHONEME = {i: p for i, p in enumerate(PHONEMES)}
VOCAB_SIZE = len(PHONEMES)


def parse_sequence(text: str) -> list[int] | None:
    """Parse a space-separated phoneme sequence into token IDs."""
    tokens = text.strip().upper().split()
    if not tokens:
        return None

    ids = []
    for tok in tokens:
        # Treat '|' as SIL (word boundary)
        if tok == '|':
            tok = 'SIL'

        if tok not in PHONEME_TO_IDX:
            print(f"  Unknown phoneme: '{tok}'")
            print(f"  Valid phonemes: {', '.join(PHONEMES)}")
            return None
        ids.append(PHONEME_TO_IDX[tok])

    return ids


def format_sequence(ids: list[int]) -> str:
    """Format token IDs back to phoneme string."""
    return ' '.join(IDX_TO_PHONEME[i] for i in ids)


def score_sequence(lm: NGramGPULanguageModel, ids: list[int], device: str) -> dict:
    """Score a phoneme sequence and return detailed results."""
    labels = torch.tensor([ids], device=device)

    # Get per-position scores
    scores = lm.score_sentences(labels, bos=True, eos=False)
    scores_with_eos = lm.score_sentences(labels, bos=True, eos=True)

    per_position = scores[0].tolist()
    eos_score = scores_with_eos[0, -1].item()
    total_log_prob = scores.sum().item()
    total_with_eos = scores_with_eos.sum().item()

    # Track state orders to detect backoff
    state_orders = get_state_orders(lm, ids, device)

    return {
        'ids': ids,
        'per_position': per_position,
        'state_orders': state_orders,
        'eos_score': eos_score,
        'total_log_prob': total_log_prob,
        'total_with_eos': total_with_eos,
        'prob': torch.exp(torch.tensor(total_log_prob)).item(),
        'prob_with_eos': torch.exp(torch.tensor(total_with_eos)).item(),
    }


def get_state_orders(lm: NGramGPULanguageModel, ids: list[int], device: str) -> list[int]:
    """
    Track the state order (effective N-gram length) at each position.

    Returns list of (state_order_before, state_order_after) for each token.
    If state_order_before < position+1 (with BOS), it means backoff occurred.
    """
    state = lm.get_init_states(batch_size=1, bos=True)  # Start at BOS state
    state_orders = []

    for i, token_id in enumerate(ids):
        # Order of current state (before consuming this token)
        order_before = lm.state_order[state.item()].item()

        # Get next states for all vocab items
        _, next_states = lm.advance(state)

        # Get the next state for this specific token
        next_state = next_states[0, token_id].item()
        order_after = lm.state_order[next_state].item()

        # Expected order if no backoff: min(position + 2, max_order)
        # position 0 after BOS -> order 2 (bigram), etc.
        expected_order = min(i + 2, lm.max_order)

        state_orders.append({
            'position': i,
            'token': IDX_TO_PHONEME[token_id],
            'order_before': order_before,
            'order_after': order_after,
            'expected_order': expected_order,
            'backed_off': order_after < expected_order,
        })

        state = torch.tensor([next_state], device=device)

    return state_orders


def print_result(result: dict) -> None:
    """Pretty-print scoring results."""
    ids = result['ids']
    per_pos = result['per_position']
    state_orders = result.get('state_orders', [])

    print("\n  Position breakdown:")
    print(f"    {'Pos':<4} {'Context':<25} {'Token':<6} {'Log P':>10} {'N-gram':>8} {'Backoff?':>10}")
    print(f"    {'-'*4} {'-'*25} {'-'*6} {'-'*10} {'-'*8} {'-'*10}")

    for i, (tok_id, log_p) in enumerate(zip(ids, per_pos)):
        if i == 0:
            context = "<s>"
        else:
            context_ids = ids[:i]
            context = "<s> " + ' '.join(IDX_TO_PHONEME[j] for j in context_ids[-4:])
            if len(context_ids) > 4:
                context = "..." + context[3:]

        tok = IDX_TO_PHONEME[tok_id]

        # Get backoff info
        if state_orders and i < len(state_orders):
            order_info = state_orders[i]
            ngram_str = f"{order_info['order_after']}-gram"
            backed_off = "YES" if order_info['backed_off'] else "no"
        else:
            ngram_str = "?"
            backed_off = "?"

        print(f"    {i:<4} {context:<25} {tok:<6} {log_p:>10.4f} {ngram_str:>8} {backed_off:>10}")

    # Show backoff summary if any backoffs occurred
    if state_orders:
        backoffs = [s for s in state_orders if s['backed_off']]
        if backoffs:
            print(f"\n  BACKOFF DETECTED at {len(backoffs)} position(s):")
            for b in backoffs:
                print(f"    - Position {b['position']} ({b['token']}): "
                      f"expected {b['expected_order']}-gram, got {b['order_after']}-gram")

    print(f"\n  Total log-prob (no EOS):  {result['total_log_prob']:.4f}")
    print(f"  Total log-prob (with EOS): {result['total_with_eos']:.4f}")
    print(f"  Probability (no EOS):      {result['prob']:.2e}")
    print(f"  Probability (with EOS):    {result['prob_with_eos']:.2e}")


def compare_sequences(results: list[dict]) -> None:
    """Print comparison table for multiple sequences."""
    if len(results) < 2:
        return

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  {'Sequence':<35} {'Log P':>10} {'Probability':>12}")
    print(f"  {'-'*35} {'-'*10} {'-'*12}")

    # Sort by probability (descending)
    sorted_results = sorted(results, key=lambda r: r['total_log_prob'], reverse=True)

    for r in sorted_results:
        seq_str = format_sequence(r['ids'])
        if len(seq_str) > 33:
            seq_str = seq_str[:30] + "..."
        print(f"  {seq_str:<35} {r['total_log_prob']:>10.4f} {r['prob']:>12.2e}")


def main():
    parser = argparse.ArgumentParser(description="Query phoneme n-gram LM probabilities")
    parser.add_argument(
        "--lm-path",
        type=str,
        default="/home/ebrahim/brainaudio/creating_n_gram_lm/phoneme_6gram.nemo",
        help="Path to phoneme LM (.nemo or .arpa)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda:0/cuda:1/cpu)"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="*",
        help="Sequences to score (non-interactive mode). Separate phonemes with spaces, sequences with commas."
    )
    args = parser.parse_args()

    print(f"Loading LM from: {args.lm_path}")
    print(f"Device: {args.device}")

    lm = NGramGPULanguageModel.from_file(
        lm_path=args.lm_path,
        vocab_size=VOCAB_SIZE,
    )
    lm = lm.to(args.device)
    print(f"Loaded {lm.max_order}-gram LM with {VOCAB_SIZE} phonemes\n")

    # Non-interactive mode
    if args.sequences:
        results = []
        for seq_str in args.sequences:
            seq_str = seq_str.strip()
            if not seq_str:
                continue
            print(f"Sequence: {seq_str}")
            ids = parse_sequence(seq_str)
            if ids:
                result = score_sequence(lm, ids, args.device)
                print_result(result)
                results.append(result)

        if len(results) > 1:
            compare_sequences(results)
        return

    # Interactive mode
    print("Enter phoneme sequences (space-separated). Use '|' or 'SIL' for word boundaries.")
    print("Commands: 'compare' (compare last entries), 'clear' (reset), 'quit' (exit)")
    print("Example: AY | AY")
    print("-" * 60)

    results = []

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        if cmd == 'clear':
            results = []
            print("Cleared comparison buffer.")
            continue

        if cmd == 'compare':
            if results:
                compare_sequences(results)
            else:
                print("No sequences to compare. Enter some sequences first.")
            continue

        if cmd == 'help':
            print("\nValid phonemes:")
            for i in range(0, len(PHONEMES), 10):
                print(f"  {', '.join(PHONEMES[i:i+10])}")
            print("\nUse '|' or 'SIL' for word boundaries.")
            continue

        if cmd == 'vocab':
            print("\nPhoneme vocabulary:")
            for i, p in enumerate(PHONEMES):
                print(f"  {i:2d}: {p}")
            continue

        # Parse and score sequence
        ids = parse_sequence(user_input)
        if ids is None:
            continue

        print(f"Scoring: {format_sequence(ids)}")
        result = score_sequence(lm, ids, args.device)
        print_result(result)
        results.append(result)

        # Auto-compare if we have multiple
        if len(results) > 1:
            print("\n(Type 'compare' to see comparison table, 'clear' to reset)")


if __name__ == "__main__":
    main()
