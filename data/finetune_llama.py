#!/usr/bin/env python3
"""
Fine-tune Llama 3.2 3B (Base) on brain-to-text sentences using Unsloth.

Optimized for RTX 5090:
- Uses Full Bfloat16 (no quantization)
- Uses Official Meta weights
- Single GPU pinning to avoid FSDP overhead on small data
"""

import argparse
import math
import os
import torch
import torch.nn.functional as F
from typing import List, Tuple
from datasets import Dataset
from tqdm import tqdm
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

# Limit to 1 GPU. Unsloth is optimized for single-card throughput.
# For small datasets (20k rows), multi-GPU communication overhead 
# often makes training slower, not faster.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_transcripts(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Load transcripts and split into train/val based on marker comment.
    """
    train_sentences = []
    val_sentences = []
    in_val_section = False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # Detect validation marker
                if line.startswith('#') and 'VALIDATION' in line.upper():
                    in_val_section = True
                    continue

                # Skip comments
                if line.startswith('#'):
                    continue

                if in_val_section:
                    val_sentences.append(line)
                else:
                    train_sentences.append(line)
                    
    except FileNotFoundError:
        print(f"Error: Transcript file not found at {file_path}")
        return [], []

    return train_sentences, val_sentences


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    sentences: List[str],
    batch_size: int = 8,
    max_length: int = 512,
    desc: str = "Computing perplexity",
) -> float:
    """Compute perplexity on a list of sentences."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    # Determine pad token id for ignore_index
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    # Process in batches
    for i in tqdm(range(0, len(sentences), batch_size), desc=desc):
        batch = sentences[i:i + batch_size]

        # Tokenize with special tokens
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        ).to(model.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        # Compute loss per token, ignoring padding
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
            ignore_index=pad_token_id,
        ).view(shift_labels.shape)

        # Accumulate only non-padded tokens
        masked_loss = (token_losses * shift_mask).sum()
        num_tokens = shift_mask.sum()

        total_loss += masked_loss.item()
        total_tokens += num_tokens.item()

        # Clear memory
        del outputs, logits, shift_logits, shift_labels, token_losses
        torch.cuda.empty_cache()

    # Perplexity = exp(avg cross-entropy loss)
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss)

    return perplexity


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.2 3B Base on transcripts"
    )
    # File Arguments
    parser.add_argument(
        "--transcript-files",
        type=str,
        nargs="+",
        default=["data/transcripts_merged.txt"],
        help="Paths to transcript files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./llama-3.2-3b-b2t-finetuned",
        help="Directory to save the final model",
    )
    
    # Model Arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="/home/ebrahim/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062",
        help="Hugging Face model ID or local path",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512, # 512 is plenty for single sentences
        help="Maximum sequence length",
    )
    
    # Training Arguments
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16) # Higher batch size for 5090
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    print("=" * 60)
    print(f"FINE-TUNING: {args.model_name}")
    print(f"PRECISION:   Bfloat16 (Full Weights)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------
    all_train = []
    all_val = []

    for f in args.transcript_files:
        print(f"Loading: {f}")
        t, v = load_transcripts(f)
        all_train.extend(t)
        all_val.extend(v)
        print(f"  -> Found {len(t)} train, {len(v)} val")

    if not all_train:
        raise ValueError("No training data found! Check your file paths.")

    print(f"\nTotal Training Samples:   {len(all_train)}")
    print(f"Total Validation Samples: {len(all_val)}")

    # ------------------------------------------------------------------
    # 2. Load Model (Unquantized)
    # ------------------------------------------------------------------
    print(f"\nLoading model architecture...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,          # Auto-detects bfloat16 for RTX 5090
        load_in_4bit=False,  # FALSE = Full Precision (Better for 5090)
    )

    # Set pad token (required for batched inference)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # 3. Evaluate Perplexity BEFORE LoRA (baseline)
    # ------------------------------------------------------------------
    ppl_before = None
    if all_val:
        print("\n" + "=" * 60)
        print("PERPLEXITY BEFORE TRAINING (Base Model)")
        print("=" * 60)
        FastLanguageModel.for_inference(model)
        ppl_before = compute_perplexity(model, tokenizer, all_val, desc="Val PPL (before)")
        print(f"Validation Perplexity: {ppl_before:.2f}")
        breakpoint()  # DEBUG: Check perplexity before continuing

    # ------------------------------------------------------------------
    # 4. Format Data with EOS Token
    # ------------------------------------------------------------------
    # CRITICAL for Base Models: We must tell the model where the sentence ends.
    # Otherwise, it will generate run-on text forever.
    EOS_TOKEN = tokenizer.eos_token

    def formatting_func(sentences):
        # Format: "Sentence text\n<|end_of_text|>"
        return [{"text": f"{s}\n{EOS_TOKEN}"} for s in sentences]

    train_dataset = Dataset.from_list(formatting_func(all_train))
    val_dataset = Dataset.from_list(formatting_func(all_val)) if all_val else None

    # ------------------------------------------------------------------
    # 4. Configure LoRA
    # ------------------------------------------------------------------
    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ------------------------------------------------------------------
    # 5. Training Config
    # ------------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False, # False is better for distinct, short sentences
        
        # Hyperparameters
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=10,
        
        # Logging & Saving
        logging_steps=1,
        save_strategy="no", # We save manually at end to save space
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=50 if val_dataset else None,
        report_to="none",
        
        # Precision (Auto-detect Bfloat16)
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 6. Initialize Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
    )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print("\nStarting Training...")
    trainer.train()

    # ------------------------------------------------------------------
    # 9. Evaluate Perplexity AFTER Training
    # ------------------------------------------------------------------
    ppl_after = None
    if all_val:
        print("\n" + "=" * 60)
        print("PERPLEXITY AFTER TRAINING")
        print("=" * 60)
        FastLanguageModel.for_inference(model)
        ppl_after = compute_perplexity(model, tokenizer, all_val, desc="Val PPL (after)")
        print(f"Validation Perplexity: {ppl_after:.2f}")
        if ppl_before:
            improvement = ppl_before - ppl_after
            pct_improvement = (improvement / ppl_before) * 100
            print(f"Improvement: {improvement:.2f} ({pct_improvement:.1f}%)")

    # ------------------------------------------------------------------
    # 10. Save Results to File
    # ------------------------------------------------------------------
    results_file = os.path.join(args.output_dir, "training_results.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Epochs: {args.num_epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Train Samples: {len(all_train)}\n")
        f.write(f"Val Samples: {len(all_val)}\n\n")
        if ppl_before is not None:
            f.write(f"Perplexity Before: {ppl_before:.2f}\n")
        if ppl_after is not None:
            f.write(f"Perplexity After: {ppl_after:.2f}\n")
        if ppl_before and ppl_after:
            improvement = ppl_before - ppl_after
            pct_improvement = (improvement / ppl_before) * 100
            f.write(f"Improvement: {improvement:.2f} ({pct_improvement:.1f}%)\n")
    print(f"\nResults saved to: {results_file}")

    # ------------------------------------------------------------------
    # 11. Save Model
    # ------------------------------------------------------------------
    print(f"\nSaving model to {args.output_dir}")
    # Save LoRA Adapters
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Optional: Save Full Merged Model (16-bit)
    # This creates a standalone model you can load without Unsloth later.
    merged_dir = os.path.join(args.output_dir, "merged_16bit")
    print(f"Saving full merged model to {merged_dir}...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("\nDone!")

if __name__ == "__main__":
    main()