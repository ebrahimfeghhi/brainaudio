#!/usr/bin/env python3
"""
Fine-tune Llama models using standard Hugging Face Transformers & PEFT.

Optimized for RTX 5090:
- Uses Bfloat16 for smaller models
- Uses 4-bit quantization for 70B models (auto-detected)
- Uses Official Meta weights
- Standard LoRA (Low-Rank Adaptation)
"""

import argparse
import math
import os
import torch
from typing import List, Tuple
from datasets import Dataset
from tqdm import tqdm

# Hugging Face imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


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
                if line.startswith('#') and 'VALIDATION' in line.upper():
                    in_val_section = True
                    continue
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
    batch_size: int = 16,
    max_length: int = 512,
    desc: str = "Computing perplexity",
) -> float:
    """
    Standard HF Perplexity calculation.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Ensure pad token is set for the tokenizer before this runs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(sentences), batch_size), desc=desc):
        batch = sentences[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        # Labels are input_ids, but masked where input is padding
        labels = inputs.input_ids.clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(**inputs, labels=labels)
        
        # CrossEntropyLoss averages over the batch tokens by default
        # We multiply by number of tokens to get total loss
        loss = outputs.loss
        
        # Calculate number of non-ignored tokens in this batch
        # (labels != -100).sum()
        num_valid_tokens = (labels != -100).sum().item()

        if num_valid_tokens > 0:
            total_loss += loss.item() * num_valid_tokens
            total_tokens += num_valid_tokens
        
        del inputs, outputs, labels, loss
        torch.cuda.empty_cache()

    if total_tokens == 0:
        return float('inf')
        
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


class PerplexityCallback(TrainerCallback):
    """Custom callback to evaluate perplexity and save best model."""

    def __init__(self, tokenizer, val_sentences, output_dir, batch_size=16, max_length=512):
        self.tokenizer = tokenizer
        self.val_sentences = val_sentences
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.best_perplexity = float('inf')

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Called after evaluation step."""
        if not self.val_sentences:
            return

        print(f"\n[Step {state.global_step}] Computing validation perplexity...")
        ppl = compute_perplexity(
            model, self.tokenizer, self.val_sentences,
            batch_size=self.batch_size,
            max_length=self.max_length,
            desc=f"Val PPL (step {state.global_step})"
        )
        print(f"[Step {state.global_step}] Perplexity: {ppl:.2f} (best: {self.best_perplexity:.2f})")

        if ppl < self.best_perplexity:
            self.best_perplexity = ppl
            print(f"[Step {state.global_step}] New best! Saving model to {self.output_dir}")
            model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)

        # Put model back in training mode
        model.train()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 3B Base via HF")
    parser.add_argument("--transcript-files", type=str, nargs="+", default=["/home/ebrahim/data2/brain2text/transcripts_merged_normalized.txt"])
    parser.add_argument("--output-dir", type=str, default="./llama-3.2-1b-hf-finetuned-normalized")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--eval-every", type=float, default=0.25, help="Evaluate every N epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="CUDA device(s) to use, e.g. '0', '0,1', or 'cpu'")

    args = parser.parse_args()

    print(f"Loading Model: {args.model_name}")

    # Detect if using a 70B model
    is_70b = "70b" in args.model_name.lower()
    if is_70b:
        print("Detected 70B model - enabling 4-bit quantization, multi-GPU, and memory optimizations")

    # Set CUDA device(s)
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    elif is_70b:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use 2 GPUs for 70B
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Single GPU for smaller models

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Llama 3 does not have a pad token by default. We use EOS.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for training stability

    # 2. Load Model
    if is_70b:
        # 4-bit quantization for 70B to fit in VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            use_cache=False,
        )
        model.gradient_checkpointing_enable()
    else:
        # Standard bfloat16 for smaller models
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            use_cache=False,
        )

    # 3. Load Data
    all_train = []
    all_val = []
    for f in args.transcript_files:
        t, v = load_transcripts(f)
        all_train.extend(t)
        all_val.extend(v)

    print(f"Train samples: {len(all_train)}, Val samples: {len(all_val)}")

    # 4. Compute Baseline Perplexity
    if all_val:
        print("Calculating baseline perplexity...")
        ppl_before = compute_perplexity(model, tokenizer, all_val, desc="Baseline PPL")
        print(f"Baseline Perplexity: {ppl_before:.2f}")
        breakpoint()

    # 5. Format Dataset
    # We add EOS token to every sample
    def formatting_func(example):
        return {"text": example["text"] + tokenizer.eos_token}

    train_ds = Dataset.from_dict({"text": all_train})
    train_ds = train_ds.map(formatting_func)
    train_ds = train_ds.shuffle(seed=args.seed)
    
    val_ds = None
    if all_val:
        val_ds = Dataset.from_dict({"text": all_val})
        val_ds = val_ds.map(formatting_func)

    # 6. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32, # alpha usually 2x rank
        lora_dropout=0.00,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 7. Adjust batch size, learning rate, and optimizer for 70B models
    if is_70b:
        effective_batch_size = args.batch_size
        per_device_batch_size = 1
        gradient_accumulation = effective_batch_size  # Accumulate to match original batch size
        learning_rate = 5e-5  # Lower LR for 70B
        optim = "paged_adamw_8bit"  # Paged 8-bit optimizer - handles memory spikes
        print(f"70B mode: per_device_batch_size=1, gradient_accumulation={gradient_accumulation}, lr={learning_rate}, optim={optim}")
    else:
        per_device_batch_size = args.batch_size
        gradient_accumulation = 1
        learning_rate = args.learning_rate
        optim = "adamw_torch"

    # 8. Calculate eval steps for 0.25 epoch intervals
    steps_per_epoch = len(train_ds) // (per_device_batch_size * gradient_accumulation)
    eval_steps = max(1, int(steps_per_epoch * args.eval_every))
    print(f"Steps per epoch: {steps_per_epoch}, Eval every {eval_steps} steps ({args.eval_every} epochs)")

    # 9. Training Arguments (using SFTConfig for trl 0.12.0+)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=learning_rate,
        optim=optim,
        weight_decay=0.001,
        logging_steps=10,
        bf16=True,  # Enable bfloat16
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=eval_steps,
        report_to="none",
        seed=args.seed,
        # SFT-specific parameters
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,  # Set to True if you want to pack multiple short sentences into one sequence
    )

    # 10. Create perplexity callback
    ppl_callback = PerplexityCallback(
        tokenizer=tokenizer,
        val_sentences=all_val,
        output_dir=args.output_dir,
        batch_size=4 if is_70b else 16,
        max_length=args.max_seq_length,
    )

    # 11. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        args=training_args,
        callbacks=[ppl_callback],
    )

    print("Starting Training...")
    trainer.train()

    # 12. Final summary
    if all_val:
        model.eval()
        print("\nCalculating final perplexity...")
        ppl_after = compute_perplexity(model, tokenizer, all_val, desc="Final PPL")
        print(f"Perplexity: {ppl_before:.2f} -> {ppl_after:.2f}")
        print(f"Best perplexity during training: {ppl_callback.best_perplexity:.2f}")

    print(f"\nBest model saved to {args.output_dir}")

if __name__ == "__main__":
    main()