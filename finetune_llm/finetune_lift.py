import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import torch
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments


# ──────────────────────────────────────────────────────────────────
# 1) Configuration & Model Loading
# ──────────────────────────────────────────────────────────────────
YEAR = "b2t_25" # or "b2t_25"

OUTPUT_DIR = "/home/ebrahim/data2/brain2text/finetuned_llms/"
EPOCH_NUM = 1
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct" 
SEED = 3416 # 3407-3416
# unsloth/gemma-3-12b-it-unsloth-bnb-4bit
# unsloth/Llama-3.2-1B-Instruct
# unsloth/Llama-3.2-3B-Instruct
# unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit

# LLM-specific configurations
max_seq_length = 512 # Adjust based on your VRAM and sequence lengths
dtype = torch.bfloat16          # None for auto-detection (Bfloat16 for Ampere+, Float16 for older)
load_in_4bit = True   # Set to True for 4-bit quantization to save memory



# Load the base model and tokenizer
# (Change to "unsloth/Llama-3.2-3B-Instruct" for the 3B version)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME, 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# ──────────────────────────────────────────────────────────────────
# 2) Dataset Loading & LLaMA-3 Chat Template Formatting
# ──────────────────────────────────────────────────────────────────
# Configure Unsloth to use the standard LLaMA-3 chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.3", # llama-3.3 works identical for 3.1, 3.2, and 3.3
)

def formatting_prompts_func(examples):
    """
    Takes the 'prompt' and 'completion' arrays from our JSONL, concatenates 
    them into a standard conversational list, and applies the LLaMA-3 chat template.
    """
    prompts = examples["prompt"]
    completions = examples["completion"]
    
    texts = []
    # Zip through the batched examples
    for prompt_turn, completion_turn in zip(prompts, completions):
        # Combine user prompt and assistant completion into a single conversation
        conversation = prompt_turn + completion_turn
        
        # apply_chat_template translates this into the <|start_header_id|> format
        text = tokenizer.apply_chat_template(
            conversation, 
            tokenize = False, 
            add_generation_prompt = False
        )
        texts.append(text)
        
    return { "text" : texts }

# Load the JSONL datasets generated from the previous step
train_data_files = "./jsonl_files/train_b2t_24_bi.jsonl" if YEAR == "b2t_24" else "./jsonl_files/train_b2t_25.jsonl"
val_data_files = "./jsonl_files/val_b2t_24_bi.jsonl" if YEAR == "b2t_24" else "./jsonl_files/val_b2t_25.jsonl"

train_dataset = load_dataset("json", data_files=train_data_files, split="train")
val_dataset = load_dataset("json", data_files=val_data_files, split="train")

# Map the formatting function across the dataset
train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
val_dataset = val_dataset.map(formatting_prompts_func, batched = True)

# Create the stacked dataset: val -> train -> val
stacked_train_dataset = concatenate_datasets([val_dataset, train_dataset, val_dataset])


# # combined
# train_dataset_24 = load_dataset("json", data_files="./jsonl_files/train_b2t_24.jsonl", split="train")
# val_dataset_24 = load_dataset("json", data_files= "./jsonl_files/val_b2t_24.jsonl", split="train")
# train_dataset_25 = load_dataset("json", data_files="./jsonl_files/train_b2t_25.jsonl", split="train")
# val_dataset_25 = load_dataset("json", data_files= "./jsonl_files/val_b2t_25.jsonl", split="train")
# train_dataset_24 = train_dataset_24.map(formatting_prompts_func, batched = True)
# val_dataset_24 = val_dataset_24.map(formatting_prompts_func, batched = True)
# train_dataset_25 = train_dataset_25.map(formatting_prompts_func, batched = True)
# val_dataset_25 = val_dataset_25.map(formatting_prompts_func, batched = True)
# stacked_train_dataset = concatenate_datasets([val_dataset_24, train_dataset_24, val_dataset_24, val_dataset_25, train_dataset_25, val_dataset_25])


print(f"Total training examples in stacked dataset: {len(stacked_train_dataset)}")
# ──────────────────────────────────────────────────────────────────
# 3) Trainer Setup
# ──────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = stacked_train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Set to True to speed up training for shorter sequences
    args = TrainingArguments(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 1,
        warmup_steps = 10,
        num_train_epochs = EPOCH_NUM,
        learning_rate = 2e-4,
        fp16 = False,
        bf16 = True,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = SEED, 
        output_dir = os.path.join(OUTPUT_DIR, f"checkpoints_{YEAR}"),
    ),
)


if __name__ == "__main__":
    # Optional but Highly Recommended: Mask the user prompt during loss calculation
    # This forces the model to only calculate loss on its own assistant-role generations (the ground truths).
    trainer = train_on_responses_only(
        trainer,
        # gemma
        # instruction_part = "<start_of_turn>user\n",
        # response_part = "<start_of_turn>model\n",
        # llama
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",

    )

    # ──────────────────────────────────────────────────────────────────
    # 4) Execute Training & Save
    # ──────────────────────────────────────────────────────────────────
    trainer_stats = trainer.train()

    # Save the LoRA adapters locally
    # --- FIXED DYNAMIC SAVE PATH ---
    model_shortname = MODEL_NAME.split('/')[-1]
    # final_save_path = os.path.join(OUTPUT_DIR, f"{model_shortname}_final_lora_combined") # saving combined finetuning
    final_save_path = os.path.join(OUTPUT_DIR, f"{model_shortname}_final_lora_{YEAR}_seed_{SEED}")

    # Save the LoRA adapters locally to the correct absolute directory
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    # If you want to merge to 16bit or GGUF format later:
    # model.save_pretrained_merged("model_16bit", tokenizer, save_method = "merged_16bit")