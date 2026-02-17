# LLM Generative Corrector Dataset Generation

This directory contains scripts to generate a finetuning dataset for a larger LLM that acts as a "generative corrector" in your brain-to-text pipeline.

## Overview

The corrector LLM receives the top N beam hypotheses from your CTC beam decoder and generates an improved final transcription. This approach leverages:
1. The beam search's exploration of multiple likely hypotheses
2. A larger LLM's language understanding to select/correct the best output

## Pipeline Architecture

```
Brain Signals → Encoder (Transformer) → Logits → Beam Search (top N beams) → Corrector LLM → Final Transcription
```

## Dataset Generation Process

### Step 1: Generate Training Set Logits (if not already available)

If you don't have logits saved for the training sets, you need to generate them first.

```bash
# For b2t_24
python scripts/save_logits.py --model best_chunked_transformer_24 --partition train

# For b2t_25
python scripts/save_logits.py --model best_chunked_transformer_25 --partition train
```

Expected output locations:
- `/home/ebrahim/data2/brain2text/b2t_24/logits/best_chunked_transformer_24/logits_train.npz`
- `/home/ebrahim/data2/brain2text/b2t_25/logits/best_chunked_transformer_25/logits_train.npz`

### Step 2: Generate Corrector Dataset

Run the dataset generation script to extract top N beams for each training sample:

```bash
# Use the provided shell script (easiest)
./finetune_llm/generate_corrector_dataset.sh

# Or run manually for each dataset
python finetune_llm/generate_data_lift.py \
  --logits-path /path/to/logits_train.npz \
  --transcripts-path /path/to/transcripts_train.pkl \
  --output-path finetune_llm/data/corrector_train.jsonl \
  --tokens /home/ebrahim/data2/brain2text/lm/units_pytorch.txt \
  --lexicon /home/ebrahim/data2/brain2text/lm/vocab_lower_100k_pytorch_phoneme_with_variants.txt \
  --word-lm-path /home/ebrahim/data2/brain2text/lm/lm_dec19_huge_4gram.kenlm \
  --beam-size 300 \
  --top-n 100 \
  --device cuda:0
```

### Step 3: Dataset Format

The generated JSONL file contains entries like:

```json
{
  "trial_id": 0,
  "input_beams": [
    "hello how are you",
    "hello how r you",
    "helo how are you",
    ...
  ],
  "ground_truth": "hello how are you",
  "instruction": "Given the following 100 candidate transcriptions from a brain-to-text decoder, generate the most accurate and coherent transcription:",
  "formatted_input": "Given the following 100 candidate transcriptions...\n\n1. hello how are you\n2. hello how r you\n..."
}
```

## Finetuning the Corrector LLM

### Recommended Model

Consider using:
- **Llama 3.2 3B** or **Llama 3.1 8B** for good performance
- **Llama 3.2 1B** if you need faster inference
- **Qwen 2.5 7B** for strong language understanding

### Data Preparation

Convert JSONL to instruction-tuning format:

```python
import json

def convert_to_instruction_format(input_jsonl, output_jsonl):
    """Convert dataset to instruction-tuning format."""
    with open(input_jsonl, 'r') as f_in, open(output_jsonl, 'w') as f_out:
        for line in f_in:
            entry = json.loads(line)

            # Create instruction-response pair
            instruction_entry = {
                "instruction": entry["formatted_input"],
                "output": entry["ground_truth"],
                "input": "",  # No additional input needed
            }
            f_out.write(json.dumps(instruction_entry) + '\n')

# Run conversion
convert_to_instruction_format(
    'finetune_llm/data/b2t_24_25_corrector_train.jsonl',
    'finetune_llm/data/corrector_train_alpaca.jsonl'
)
```

### Finetuning with HuggingFace

Example using `transformers` and LoRA:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset('json', data_files='finetune_llm/data/corrector_train_alpaca.jsonl')

# Training arguments
training_args = TrainingArguments(
    output_dir="./corrector_llm_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    tokenizer=tokenizer,
    max_seq_length=2048,
)
trainer.train()
```

## Hyperparameters

### Beam Search Parameters
- `--beam-size 300`: Large beam to explore diverse hypotheses
- `--top-n 100`: Number of beams to provide to the corrector LLM
- `--acoustic-scale 0.4`: Scale acoustic scores (adjust based on your model)
- `--alpha-ngram 1.0`: N-gram LM weight during beam search

### Recommendation
Start with top-100 beams. If the LLM struggles with long inputs or the beams are too similar, reduce to top-20 or top-50.

## Inference with Corrector LLM

After finetuning, integrate into your decoding pipeline:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + LoRA
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
model = PeftModel.from_pretrained(base_model, "./corrector_llm_checkpoints")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

def correct_beams(beams: list[str]) -> str:
    """Generate corrected transcription from beam hypotheses."""
    instruction = f"Given the following {len(beams)} candidate transcriptions..."
    beam_text = "\n".join([f"{i+1}. {beam}" for i, beam in enumerate(beams)])
    prompt = f"{instruction}\n\n{beam_text}\n\nGenerate the corrected transcription:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the generated part (after prompt)
    return corrected.split("Generate the corrected transcription:")[-1].strip()

# Usage in decoding
beams = run_beam_search(logits, top_k=100)
final_output = correct_beams(beams)
```

## Expected Performance

With a well-trained corrector LLM, you should see:
- **WER improvement**: 5-15% relative improvement over best beam
- **Grammar correction**: Better handling of articles, verb tenses
- **Rare words**: Better recovery of uncommon words seen in top beams but not in best beam
- **Consistency**: More coherent long-form outputs

## Troubleshooting

### Issue: Out of memory during dataset generation
- Reduce `--beam-size` or process fewer samples at once
- Use `--max-samples 100` for testing

### Issue: All beams are identical
- The beam search is collapsing. Try:
  - Increasing `--beam-prune-threshold`
  - Reducing `--acoustic-scale`
  - Increasing beam diversity during search

### Issue: Logits files don't exist
- Generate them using `scripts/save_logits.py` with `PARTITION='train'`
- Make sure you're using the correct model checkpoint

## Files in this Directory

- `generate_data_lift.py`: Main dataset generation script
- `generate_corrector_dataset.sh`: Convenience script to generate datasets for both b2t_24 and b2t_25
- `README_corrector.md`: This file
- `data/`: Output directory for generated datasets (created automatically)
