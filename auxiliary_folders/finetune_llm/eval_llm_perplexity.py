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
    sentences,
    batch_size=16,
    max_length=512,
    device="cuda",
    desc="Computing Perplexity",
):
    """
    Standard HF Perplexity calculation (matches finetune_llama.py).
    """
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(sentences), batch_size), desc=desc):
        batch = sentences[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        labels = inputs.input_ids.clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(**inputs, labels=labels, use_cache=False)

        num_valid_tokens = (labels != -100).sum().item()
        if num_valid_tokens > 0:
            total_loss += outputs.loss.item() * num_valid_tokens
            total_tokens += num_valid_tokens

        del inputs, outputs, labels
        torch.cuda.empty_cache()

    if total_tokens == 0:
        return float('inf'), float('-inf'), 0

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity, -avg_loss, total_tokens


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM perplexity on transcripts")
    parser.add_argument(
        "--transcript-file",
        type=str,
        default= "/home/ebrahim/brainaudio/data/transcripts_merged_normalized.txt", #"../data/transcripts_merged_normalized.txt",
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map=args.device,
    )

    # Load LoRA adapter if specified
    if args.lora_path:
        print(f"Loading LoRA adapter: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)

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
        model, tokenizer, train_sentences, batch_size=args.batch_size, device=device, desc="Train"
    )
    print(f"  Perplexity: {train_ppl:.2f}")
    print(f"  Avg Log Prob: {train_avg_lp:.4f}")
    print(f"  Total Tokens: {train_tokens:,}")

    # Evaluate validation set
    print("\n" + "=" * 60)
    print("VALIDATION SET")
    print("=" * 60)
    val_ppl, val_avg_lp, val_tokens = compute_perplexity(
        model, tokenizer, val_sentences, batch_size=args.batch_size, device=device, desc="Val"
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
