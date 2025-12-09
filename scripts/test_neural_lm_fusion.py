"""Tiny end-to-end CTC beam-search check with neural LM fusion.

By default the script decodes validation trials 0 and 1 in a single
batch, applies the vectorized lexicon constraint, and fuses a
HuggingFace causal LM at word boundaries. Keeping the surface area small
still exercises the real decoder path (including batched contexts) so it
is easy to reason about `HuggingFaceLMFusion` changes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from brainaudio.inference.decoder import (
    BatchedBeamCTCComputer,
    VectorizedLexiconConstraint,
    HuggingFaceLMFusion,
)
from brainaudio.inference.decoder.beam_helpers import (
    decode_beam_texts,
    load_log_probs,
    load_phoneme_to_word_mapping,
    load_token_to_phoneme_mapping,
    pick_device,
)

#DEFAULT_LOGITS = "/data2/brain2text/b2t_25/logits/tm_transformer_b2t_24+25_large_wide_bidir_grad_clip_cosine_decay/logits_val.npz"
DEFAULT_LOGITS =  "/data2/brain2text/b2t_25/logits/tm_transformer_combined_reduced_reg_seed_0/logits_val_None_None.npz"
DEFAULT_TOKENS = "/data2/brain2text/lm/units_pytorch.txt"
DEFAULT_LEXICON = "/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CTC beam search + HF LM fusion smoke test")
    parser.add_argument(
        "--trials",
        type=int,
        nargs="+",
        default=[1],
        help="Validation trial indices to decode together (default: 0 1)",
    )
    parser.add_argument("--beam-size", type=int, default=50, help="CTC beam size (default: 3)")
    parser.add_argument(
        "--top-beams",
        type=int,
        default=1,
        help="Number of beams to print with scores (default: 1)",
    )
    
    parser.add_argument("--model", default="google/gemma-3-270m", help="HuggingFace causal LM checkpoint")
    parser.add_argument("--hf-token", default=None, help="Optional HF token for gated models")
    parser.add_argument("--lm-weight", type=float, default=1, help="Fusion weight passed to HuggingFaceLMFusion")
    parser.add_argument("--max-context-length", type=int, default=512, help="Token budget (including BOS)")
    parser.add_argument("--device", default=None, help="Torch device for CTC + LM (default: cuda if available)")
    parser.add_argument("--logits", type=Path, default=Path(DEFAULT_LOGITS), help="NPZ file containing validation logits")
    parser.add_argument("--tokens", type=Path, default=Path(DEFAULT_TOKENS), help="units_pytorch.txt file")
    parser.add_argument("--lexicon", type=Path, default=Path(DEFAULT_LEXICON), help="lexicon file")
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    log_probs, lengths = load_log_probs(args.logits, args.trials, device)

    lexicon = VectorizedLexiconConstraint.from_file_paths(
        tokens_file=str(args.tokens),
        lexicon_file=str(args.lexicon),
        device=device,
    )
    token_table = load_token_to_phoneme_mapping(args.tokens)
    phoneme_to_word = load_phoneme_to_word_mapping(args.lexicon)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        token=args.hf_token,
    ).to(device)

    lm_fusion = HuggingFaceLMFusion(
        model=model,
        tokenizer=tokenizer,
        weight=args.lm_weight,
        homophone_aggregation="max",
        device=device,
        max_context_length=args.max_context_length,
    )

    decoder = BatchedBeamCTCComputer(
        blank_index=lexicon.blank_index,
        beam_size=args.beam_size,
        lexicon=lexicon,
        lm_fusion=lm_fusion,
        allow_cuda_graphs=False,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
        
    decode_start = time.perf_counter()
    result = decoder(log_probs, lengths)
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    decode_elapsed = time.perf_counter() - decode_start
    top_k = max(0, min(args.top_beams, result.transcript_wb.shape[1]))
    decoded_beams = decode_beam_texts(
        beam_hyps=result,
        token_table=token_table,
        lexicon=lexicon,
        phoneme_to_word=phoneme_to_word,
        top_k=top_k,
    )
    decoded_texts = [texts[0] if texts else "<EMPTY>" for texts in decoded_beams]

    print("\n=== Neural LM Fusion Decode ===")
    print(
        f"Beam search wall time: {decode_elapsed * 1000:.1f} ms"
        f" ({decode_elapsed:.3f} s) for batch size {len(args.trials)}"
    )
    for batch_idx, (trial_idx, text) in enumerate(zip(args.trials, decoded_texts)):
        best_score = result.scores[batch_idx, 0].item()
        print(f"Trial {trial_idx:3d} | Beam size {args.beam_size} | Score {best_score:.4f}")
        print(f"   Best: {text}")

        if top_k == 0:
            continue

        print("   Top beams:")
        for beam_rank, beam_text in enumerate(decoded_beams[batch_idx]):
            beam_score = result.scores[batch_idx, beam_rank].item()
            print(f"     #{beam_rank:02d} | log {beam_score:.4f} | {beam_text}")


if __name__ == "__main__":
    main()
