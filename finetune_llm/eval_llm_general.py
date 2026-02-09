# #!/usr/bin/env python3
# """
# General-purpose perplexity evaluation for causal language models.

# This script uses the EXACT SAME perplexity computation as the training scripts,
# ensuring consistency between training and evaluation metrics.

# Works with any causal LM: LLaMA, Granite, Gemma, Mistral, etc.
# """

# import argparse
# import math
# import torch
# from typing import List, Tuple
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel


# def load_transcripts(file_path: str) -> Tuple[List[str], List[str]]:
#     """
#     Load transcripts and split into train/val based on marker comment.

#     Returns:
#         (train_sentences, val_sentences)
#     """
#     train_sentences = []
#     val_sentences = []
#     in_val_section = False

#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 if line.startswith('#') and 'VALIDATION' in line.upper():
#                     in_val_section = True
#                     continue
#                 if line.startswith('#'):
#                     continue

#                 if in_val_section:
#                     val_sentences.append(line)
#                 else:
#                     train_sentences.append(line)

#     except FileNotFoundError:
#         print(f"Error: Transcript file not found at {file_path}")
#         return [], []

#     return train_sentences, val_sentences


# @torch.no_grad()
# def compute_perplexity(
#     model,
#     tokenizer,
#     sentences: List[str],
#     batch_size: int = 16,
#     max_length: int = 512,
#     desc: str = "Computing perplexity",
#     add_eos: bool = True,
# ) -> Tuple[float, float, int]:
#     """
#     Compute perplexity using the SAME method as training scripts.

#     This uses the model's built-in loss computation, which is what
#     SFTTrainer uses during training.

#     Args:
#         model: The causal language model
#         tokenizer: The tokenizer
#         sentences: List of text sentences to evaluate
#         batch_size: Batch size for evaluation
#         max_length: Maximum sequence length
#         desc: Description for progress bar
#         add_eos: Whether to add EOS token (should match training setup)

#     Returns:
#         (perplexity, avg_loss, total_tokens)
#     """
#     model.eval()
#     total_loss = 0.0
#     total_tokens = 0

#     # Ensure pad token is set
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # Add EOS token to sentences if specified (to match training)
#     if add_eos:
#         sentences = [s + tokenizer.eos_token for s in sentences]

#     for i in tqdm(range(0, len(sentences), batch_size), desc=desc):
#         batch = sentences[i:i + batch_size]

#         inputs = tokenizer(
#             batch,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=max_length,
#         ).to(model.device)

#         # Labels are input_ids, but masked where input is padding
#         labels = inputs.input_ids.clone()
#         if tokenizer.pad_token_id is not None:
#             labels[labels == tokenizer.pad_token_id] = -100

#         outputs = model(**inputs, labels=labels)

#         # Model returns average loss over valid tokens in the batch
#         # We multiply by number of tokens to get total loss
#         loss = outputs.loss

#         # Calculate number of non-ignored tokens in this batch
#         num_valid_tokens = (labels != -100).sum().item()

#         if num_valid_tokens > 0:
#             total_loss += loss.item() * num_valid_tokens
#             total_tokens += num_valid_tokens

#         del inputs, outputs, labels, loss
#         torch.cuda.empty_cache()

#     if total_tokens == 0:
#         return float('inf'), float('inf'), 0

#     avg_loss = total_loss / total_tokens
#     perplexity = math.exp(avg_loss)

#     return perplexity, avg_loss, int(total_tokens)


# def main():
#     parser = argparse.ArgumentParser(
#         description="Evaluate perplexity of causal LMs (LLaMA, Granite, Gemma, etc.)",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Evaluate base model
#   python eval_llm_general.py --model-name meta-llama/Llama-3.2-1B

#   # Evaluate finetuned model with LoRA
#   python eval_llm_general.py \\
#     --model-name ibm-granite/granite-3.0-2b-base \\
#     --lora-path ./granite-3.0-2b-finetuned-normalized

#   # Use specific device
#   python eval_llm_general.py --model-name google/gemma-3-270m --device 0
#         """
#     )
#     parser.add_argument(
#         "--transcript-file",
#         type=str,
#         default="/home/ebrahim/data2/brain2text/transcripts_merged_normalized.txt",
#         help="Path to transcripts file",
#     )
#     parser.add_argument(
#         "--model-name",
#         type=str,
#         required=True,
#         help="HuggingFace model name (e.g., meta-llama/Llama-3.2-1B)",
#     )
#     parser.add_argument(
#         "--lora-path",
#         type=str,
#         default=None,
#         help="Path to LoRA adapter directory (optional)",
#     )
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         default=16,
#         help="Batch size for inference",
#     )
#     parser.add_argument(
#         "--max-length",
#         type=int,
#         default=512,
#         help="Maximum sequence length",
#     )
#     parser.add_argument(
#         "--device",
#         type=str,
#         default=None,
#         help="CUDA device to use (e.g., '0', '1'). If not specified, uses device 1 or auto.",
#     )
#     parser.add_argument(
#         "--no-eos",
#         action="store_true",
#         help="Don't add EOS token to sentences. Use ONLY if your training didn't add EOS.",
#     )
#     parser.add_argument(
#         "--trust-remote-code",
#         action="store_true",
#         help="Trust remote code (needed for some models like Granite)",
#     )

#     args = parser.parse_args()

#     # Set CUDA device if specified
#     if args.device is not None:
#         import os
#         os.environ["CUDA_VISIBLE_DEVICES"] = args.device
#         print(f"Using CUDA device: {args.device}")

#     print("=" * 70)
#     print("GENERAL-PURPOSE LLM PERPLEXITY EVALUATION")
#     print("=" * 70)
#     print(f"Model: {args.model_name}")
#     if args.lora_path:
#         print(f"LoRA Adapter: {args.lora_path}")
#     print(f"Transcript file: {args.transcript_file}")
#     print("=" * 70)

#     # Load tokenizer
#     print("\nLoading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model_name,
#         trust_remote_code=args.trust_remote_code
#     )

#     # Load model with bfloat16 (matching training scripts)
#     print("Loading model...")
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         use_cache=False,
#         trust_remote_code=args.trust_remote_code,
#     )

#     # Load LoRA adapter if specified
#     if args.lora_path:
#         print(f"Loading LoRA adapter from: {args.lora_path}")
#         model = PeftModel.from_pretrained(model, args.lora_path)
#         model = model.merge_and_unload()  # Merge for faster inference
#         print("LoRA adapter merged successfully")

#     # Set pad token if not set
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

#     # Load transcripts
#     print(f"\nLoading transcripts from: {args.transcript_file}")
#     train_sentences, val_sentences = load_transcripts(args.transcript_file)
#     print(f"  Train sentences: {len(train_sentences):,}")
#     print(f"  Val sentences: {len(val_sentences):,}")

#     add_eos = not args.no_eos
#     if add_eos:
#         print("  ✓ Adding EOS token to sentences (matching standard training setup)")
#     else:
#         print("  ⚠ NOT adding EOS token (--no-eos flag set)")

#     # Evaluate training set
#     if train_sentences:
#         print("\n" + "=" * 70)
#         print("TRAINING SET EVALUATION")
#         print("=" * 70)
#         train_ppl, train_loss, train_tokens = compute_perplexity(
#             model=model,
#             tokenizer=tokenizer,
#             sentences=train_sentences,
#             batch_size=args.batch_size,
#             max_length=args.max_length,
#             desc="Train PPL",
#             add_eos=add_eos,
#         )
#         print(f"  Perplexity: {train_ppl:.2f}")
#         print(f"  Average Loss: {train_loss:.4f}")
#         print(f"  Total Tokens: {train_tokens:,}")

#     # Evaluate validation set
#     if val_sentences:
#         print("\n" + "=" * 70)
#         print("VALIDATION SET EVALUATION")
#         print("=" * 70)
#         val_ppl, val_loss, val_tokens = compute_perplexity(
#             model=model,
#             tokenizer=tokenizer,
#             sentences=val_sentences,
#             batch_size=args.batch_size,
#             max_length=args.max_length,
#             desc="Val PPL",
#             add_eos=add_eos,
#         )
#         print(f"  Perplexity: {val_ppl:.2f}")
#         print(f"  Average Loss: {val_loss:.4f}")
#         print(f"  Total Tokens: {val_tokens:,}")

#     # Summary
#     print("\n" + "=" * 70)
#     print("SUMMARY")
#     print("=" * 70)
#     print(f"Model: {args.model_name}")
#     if args.lora_path:
#         print(f"LoRA Adapter: {args.lora_path}")
#     if train_sentences:
#         print(f"Train Perplexity: {train_ppl:.2f}")
#     if val_sentences:
#         print(f"Val Perplexity: {val_ppl:.2f}")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Evaluate perplexity of IBM Granite / Llama models.
Corrected for: BFloat16, EOS Token appending, and Padding Masking.
"""

import argparse
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from typing import List, Tuple

def load_transcripts(file_path: str) -> Tuple[List[str], List[str]]:
    train_sentences = []
    val_sentences = []
    in_val_section = False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('#') and 'VALIDATION' in line.upper():
                in_val_section = True
                continue
            if line.startswith('#'): continue

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
    ignore_eos_evaluation: bool = False,
    desc: str = "Computing perplexity",
) -> Tuple[float, float, int]:
    model.eval()
    total_log_prob = 0.0
    total_tokens = 0

    # Sort by length to minimize padding (faster inference)
    sorted_indices = sorted(range(len(sentences)), key=lambda k: len(sentences[k]))
    sorted_sentences = [sentences[i] for i in sorted_indices]

    for chunk_start in tqdm(range(0, len(sorted_sentences), batch_size), desc=desc):
        chunk_sentences = sorted_sentences[chunk_start:chunk_start + batch_size]

        # 1. APPEND EOS MANUALLY to match training behavior
        #    (The tokenizer usually adds BOS, but we must ensure EOS exists)
        chunk_sentences = [s + tokenizer.eos_token for s in chunk_sentences]

        # 2. Tokenize
        #    Note: padding_side="right" is standard for PPL calculation
        tokenizer.padding_side = "right"
        inputs = tokenizer(
            chunk_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True, # Adds BOS
        ).to(device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits

        # 3. Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        # 4. Flatten
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        # 5. Calculate Loss WITHOUT ignore_index
        #    We manually mask using the attention mask later.
        #    This prevents accidentally ignoring the EOS token if pad_token == eos_token
        token_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction='none'
        )
        token_losses = token_losses.view(shift_labels.shape)

        if ignore_eos_evaluation:
            # SIMULATE TRAINING SCRIPT BEHAVIOR:
            # If the label is EOS, ignore it (even if it's the real one)
            # We treat EOS ID as if it were a pad token ID
            special_mask = shift_labels.ne(tokenizer.eos_token_id)
            final_mask = shift_mask * special_mask
        else:
            # CORRECT BEHAVIOR:
            # Only ignore padding. Count the Real EOS.
            final_mask = shift_mask

        # 6. Apply Mask
        #    This zeros out loss for padding tokens, but KEEPS the loss for the EOS token
        masked_losses = token_losses * shift_mask

        batch_log_probs = -masked_losses.sum().item()
        batch_tokens = shift_mask.sum().item()

        total_log_prob += batch_log_probs
        total_tokens += batch_tokens

        del outputs, logits, shift_logits, shift_labels, token_losses, masked_losses
        del inputs, input_ids, attention_mask
        torch.cuda.empty_cache()

    avg_log_prob = total_log_prob / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(-avg_log_prob)

    return perplexity, avg_log_prob, int(total_tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript-file", type=str, default="/home/ebrahim/data2/brain2text/transcripts_merged_normalized.txt")
    parser.add_argument("--model-name", type=str, default="ibm-granite/granite-3.0-2b-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--ignore-eos", action="store_true", help="Mimic the training script's masking bug")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Ensure Pad Token Logic matches training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LOAD IN BFLOAT16 (Crucial for Granite/Llama 3)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map=args.device,
        trust_remote_code=True
    )

    if args.lora_path:
        print(f"Loading LoRA adapter: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)

    print(f"Loading transcripts: {args.transcript_file}")
    train_sentences, val_sentences = load_transcripts(args.transcript_file)
    device = torch.device(args.device)

    # Evaluate training set
    print("\n" + "=" * 60)
    print("TRAINING SET")
    print("=" * 60)
    train_ppl, train_avg_lp, train_tokens = compute_perplexity(
        model, tokenizer, train_sentences, device, args.batch_size, ignore_eos_evaluation=args.ignore_eos, 
    )
    print(f"  Perplexity: {train_ppl:.2f}")
    print(f"  Avg Log Prob: {train_avg_lp:.4f}")
    print(f"  Total Tokens: {train_tokens:,}")

    # Evaluate validation set
    print("\n" + "=" * 60)
    print("VALIDATION SET")
    print("=" * 60)
    val_ppl, val_avg_lp, val_tokens = compute_perplexity(
        model, tokenizer, val_sentences, device, args.batch_size, ignore_eos_evaluation=args.ignore_eos, 
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