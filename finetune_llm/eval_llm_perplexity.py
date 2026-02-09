#!/usr/bin/env python3
"""
Evaluate perplexity of Llama 3.2 3B on transcripts_all.txt.

Evaluates training and validation sets separately.
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


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


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    sentences: List[str],
    device: torch.device,
    batch_size: int = 32,
    desc: str = "Computing perplexity",
) -> Tuple[float, float, int]:
    """
    Compute perplexity using fused cross-entropy (memory efficient).

    Returns:
        (perplexity, avg_log_prob, total_tokens)
    """
    model.eval()

    total_log_prob = 0.0
    total_tokens = 0

    # Sort by length for efficient batching
    sorted_indices = sorted(range(len(sentences)), key=lambda k: len(sentences[k]))
    sorted_sentences = [sentences[i] for i in sorted_indices]

    for chunk_start in tqdm(range(0, len(sorted_sentences), batch_size), desc=desc):
        chunk_sentences = sorted_sentences[chunk_start:chunk_start + batch_size]

        # Tokenize with BOS token
        inputs = tokenizer(
            chunk_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        ).to(device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        # Fused cross-entropy (memory efficient)
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        token_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction='none',
            ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100,
        )

        token_losses = token_losses.view(shift_labels.shape)

        # Accumulate log probabilities (negative of loss)
        # Only count non-padded tokens
        batch_log_probs = -(token_losses * shift_mask).sum().item()
        batch_tokens = shift_mask.sum().item()

        total_log_prob += batch_log_probs
        total_tokens += batch_tokens

        # Clear memory
        del outputs, logits, shift_logits, shift_labels, token_losses
        del inputs, input_ids, attention_mask
        torch.cuda.empty_cache()

    # Perplexity = exp(-avg_log_prob)
    avg_log_prob = total_log_prob / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(-avg_log_prob)

    return perplexity, avg_log_prob, int(total_tokens)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM perplexity on transcripts")
    parser.add_argument(
        "--transcript-file",
        type=str,
        default= "/home/ebrahim/data2/brain2text/transcripts_merged_normalized.txt", #"../data/transcripts_merged_normalized.txt",
        help="Path to transcripts file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter (optional)",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
    )

    # Load LoRA adapter if specified
    if args.lora_path:
        print(f"Loading LoRA adapter: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading transcripts: {args.transcript_file}")
    train_sentences, val_sentences = load_transcripts(args.transcript_file)
    print(f"  Train sentences: {len(train_sentences)}")
    print(f"  Val sentences: {len(val_sentences)}")

    device = torch.device(args.device)

    # Evaluate training set
    print("\n" + "=" * 60)
    print("TRAINING SET")
    print("=" * 60)
    train_ppl, train_avg_lp, train_tokens = compute_perplexity(
        model, tokenizer, train_sentences, device, args.batch_size, "Train"
    )
    print(f"  Perplexity: {train_ppl:.2f}")
    print(f"  Avg Log Prob: {train_avg_lp:.4f}")
    print(f"  Total Tokens: {train_tokens:,}")

    # Evaluate validation set
    print("\n" + "=" * 60)
    print("VALIDATION SET")
    print("=" * 60)
    val_ppl, val_avg_lp, val_tokens = compute_perplexity(
        model, tokenizer, val_sentences, device, args.batch_size, "Val"
    )
    print(f"  Perplexity: {val_ppl:.2f}")
    print(f"  Avg Log Prob: {val_avg_lp:.4f}")
    print(f"  Total Tokens: {val_tokens:,}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    if args.lora_path:
        print(f"LoRA Adapter: {args.lora_path}")
    print(f"Train Perplexity: {train_ppl:.2f}")
    print(f"Val Perplexity: {val_ppl:.2f}")


if __name__ == "__main__":
    main()
