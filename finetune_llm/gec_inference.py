import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel

# ──────────────────────────────────────────────────────────────────
# 1) Setup Device & Paths
# ──────────────────────────────────────────────────────────────────
SEED = 3416
YEAR = "b2t_25"
MODE = "uni" 
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-bnb-4bit" # Llama-3.2-1B-Instruct   Meta-Llama-3.1-8B-Instruct-bnb-4bit  gemma-3-12b-it-unsloth-bnb-4bit
OUTPUT_DIR = "/home/ebrahim/data2/brain2text/finetuned_llms/"
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_final_lora_{YEAR}_wfst_{MODE}_seed_{SEED}")
#FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_final_lora_combined")

TEST_DATA_PATH = f"./jsonl_files/test_{YEAR}_wfst_{MODE}.jsonl"
OUTPUT_TXT_PATH = f"generative_results/test_predictions_{YEAR}_{MODEL_NAME}_wfst_{MODE}_seed_{SEED}.txt"

# ──────────────────────────────────────────────────────────────────
# 2) Load Model & Tokenizer
# ──────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = FINAL_MODEL_PATH, # Loading your completed LoRA
    max_seq_length = 512,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)

# Enable Unsloth's native 2x faster inference engine
FastLanguageModel.for_inference(model)

# --- CRITICAL FOR BATCHED INFERENCE ---
# We must pad on the left so the <|start_header_id|>assistant<|end_header_id|>
# generation trigger is always the very last token in the tensor.
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
# --------------------------------------

# ──────────────────────────────────────────────────────────────────
# 3) Load Test Data & Apply Chat Template
# ──────────────────────────────────────────────────────────────────
test_dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")

def format_for_inference(examples):
    prompts = examples["prompt"]
    texts = []
    for prompt in prompts:
        # add_generation_prompt=True automatically appends the assistant trigger
        text = tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True 
        )
        texts.append(text)
    return {"text": texts}

test_dataset = test_dataset.map(format_for_inference, batched=True)

# ──────────────────────────────────────────────────────────────────
# 4) Batched Generation Loop
# ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 1 # Increase to 32 or 64 if you have the VRAM
results = []

# Define the hard stop tokens for Llama 3
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Define the hard stop tokens for Gemma 3
# terminators = [
#     tokenizer.tokenizer.eos_token_id,                          # Resolves to <eos>
#     tokenizer.tokenizer.convert_tokens_to_ids("<end_of_turn>") # Resolves to <end_of_turn>
# ]

print("🚀 Starting batched inference on test dataset...")
for i in range(0, len(test_dataset), BATCH_SIZE):
    batch_texts = test_dataset["text"][i : i + BATCH_SIZE]
    
    # Tokenize the batch and push to GPU
    # inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    # gemma-specific format
    inputs = tokenizer(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=64, # Adjust based on how long your target sentences are
            use_cache=True,
            eos_token_id=terminators, # Stops generation the moment the turn ends
        )
    
    # Slice off the input prompt to get ONLY the newly generated tokens
    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[:, prompt_length:]
    
    # Decode to text
    decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    for raw_text in decoded_batch:
        # 1. Split by newline and grab ONLY the very first line generated
        first_line = raw_text.split('\n')[0].strip()
        # 2. Strip out the stray "assistant" string just in case it bled into line 1
        clean_sentence = first_line.replace("assistant", "").strip()
        
        results.append(clean_sentence)
    # results.extend([text.strip() for text in decoded_batch])
    
    print(f"Processed {min(i + BATCH_SIZE, len(test_dataset))} / {len(test_dataset)}")

# ──────────────────────────────────────────────────────────────────
# 5) Save to TXT
# ──────────────────────────────────────────────────────────────────
with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
    for sentence in results:
        # Force the output to be strictly one line per sentence
        clean_sentence = sentence.replace("\n", " ").strip()
        f.write(clean_sentence + "\n")

print(f"✅ Finished! Predictions saved line-by-line to {OUTPUT_TXT_PATH}")