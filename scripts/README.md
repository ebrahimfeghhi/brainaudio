## Scripts

This directory contains the end-to-end pipeline for training brain-to-text acoustic models and running the Lightbeam CTC decoder.

### Pipeline Overview

```
1. Format data        →  scripts/dataset/
2. Train model        →  scripts/train.py
3. Finetune LLM       →  scripts/finetune_llm.py
4. Generate + Decode  →  scripts/batch_decode.py
```

---

### Directory Structure

```
scripts/
├── README.md                     # This file
├── decoder_config.py             # Default hyperparameters and paths for the decoder (edit before first run)
├── custom_config.yaml            # Custom training config (edit before running train.py)
├── train.py                      # Train a CTC acoustic encoder (Transformer or GRU)
├── generate_logits.py            # Run inference and save encoder logits to disk (called by batch_decode.py)
├── run_decoder.py                # Run the Lightbeam CTC beam search decoder (called by batch_decode.py)
├── batch_decode.py               # Generate logits and decode across multiple models/seeds
├── finetune_llm.py               # SFT fine-tune an LLM with LoRA on transcript data
├── save_transcript.py            # Save model predictions as transcript files
├── save_config.py                # Save a training configuration to YAML
├── dataset/
│   ├── brain2text_2025.py        # Download and format the B2T-25 dataset (HDF5 → pickle)
│   ├── brain2text_2024.py        # Download and format the B2T-24 dataset (HDF5 → pickle)
│   └── lazyload_format.py        # Convert datasets to lazy-loading format for memory-efficient training
├── aux_scripts/
│   ├── compare_predictions.py    # Compare decoder predictions against a baseline and compute WER
│   └── check_vocab.py            # Check for out-of-vocabulary words in a transcript file
└── _archive/                     # Deprecated scripts kept for reference — not intended for use
```

---

### Configuration: `decoder_config.py`

`decoder_config.py` holds beam search and LLM hyperparameters tuned per dataset. You should not need to edit it directly — all user-facing variables (dataset, model mode, base path) are passed via CLI flags to `batch_decode.py` and threaded through as environment variables.

Hyperparameter defaults can be overridden at runtime via command-line flags to `run_decoder.py`.

---

### Step 1 — Format Data

Download and convert a dataset to the internal pickle format used by the training pipeline.

```bash
# Format the B2T-25 dataset
uv run scripts/dataset/brain2text_2025.py

# Format the B2T-24 dataset
uv run scripts/dataset/brain2text_2024.py
```

Each script downloads raw data from respective sources and converts it to a uniform pickle format. Edit the path constants at the top of the script before running.

---

### Step 2 — Train the Encoder

`train.py` trains a CTC acoustic encoder (Transformer or GRU) over multiple random seeds.

**Before running**, edit the constants at the top of `train.py`:

```python
config_path = "your_config.yaml"   # YAML config in src/brainaudio/training/utils/custom_configs/
device = "cuda:0"                  # GPU to train on
```

Training configs are YAML files in `src/brainaudio/training/utils/custom_configs/`. Key fields:

| Field | Description |
|---|---|
| `modelType` | `"transformer"` or `"gru"` |
| `modelName` | Base name for saved checkpoints |
| `seeds` | List of random seeds (e.g. `[0, 1, 2]`) |
| `nClasses` | Number of output phoneme classes |
| `dropout`, `input_dropout` | Regularization |
| `learning_rate` | Peak learning rate |
| `normalize_ctc_len` | Normalize CTC loss by sequence length (`true` for GRU-24 and Transformer, `false` for GRU-25) |
| `model.gru.shared_input` | If `true`, all days share a single input transform instead of day-specific weights |

```bash
uv run scripts/train.py
```

Checkpoints and metrics are saved to the path in the config and logged to Weights & Biases.

---

### Step 3 — Finetune the LLM

`finetune_llm.py` SFT fine-tunes a causal LLM with LoRA on transcript data. The resulting adapter is used by the decoder for shallow fusion rescoring.

```bash
uv run scripts/finetune_llm.py \
    --model-name meta-llama/Llama-3.2-3B \
    --transcript-files /path/to/transcripts_merged_normalized.txt \
    --output-dir /path/to/save/adapter \
    --num-epochs 3 \
    --batch-size 16 \
    --device 0
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--model-name` | — | HuggingFace model ID (e.g. `meta-llama/Llama-3.2-3B`, `google/gemma-3-270m`) |
| `--transcript-files` | — | One or more transcript `.txt` files (train/val split by `# VALIDATION` marker) |
| `--output-dir` | — | Where to save the best LoRA adapter checkpoint |
| `--num-epochs` | `3` | Number of training epochs |
| `--eval-every` | `0.25` | Evaluate perplexity every N epochs |
| `--batch-size` | `16` | Per-device batch size |
| `--learning-rate` | `2e-4` | Learning rate |
| `--device` | auto | CUDA device(s), e.g. `0`, `0,1` |

The best checkpoint (lowest validation perplexity) is saved to `--output-dir`. Point `PATHS["lora_adapter_*"]` in `decoder_config.py` to this directory.

---

### Step 4 — Generate Logits + Decode

`batch_decode.py` is the main entry point for inference. It generates logits for each seed and immediately decodes them, looping over all seeds in sequence. Any additional arguments are passed through to `run_decoder.py`.

```bash
# Generate logits + decode (default — omit --logits-base)
uv run scripts/batch_decode.py \
    --dataset b2t_24 \
    --model-mode gru \
    --base-path /home/user \
    --brain2text-dir /home/user/data2 \
    --model-template "gru_b2t_24_baseline_brainaudio_seed_{seed}" \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --val \
    --device cuda:0

# Decode only (logits already exist — pass --logits-base to skip generation)
uv run scripts/batch_decode.py \
    --dataset b2t_24 \
    --model-mode gru \
    --base-path /home/user \
    --brain2text-dir /home/user/data2 \
    --logits-base /home/user/data2/brain2text/b2t_24/logits \
    --model-template "gru_b2t_24_baseline_brainaudio_seed_{seed}" \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --val \
    --device cuda:0

# Override decoder hyperparameters for all runs
uv run scripts/batch_decode.py \
    --dataset b2t_25 \
    --model-mode transformer \
    --base-path /home/user \
    --brain2text-dir /home/user/data2 \
    --model-template "my_transformer_seed_{seed}" \
    --seeds 0 1 2 \
    --val --test \
    --beam-size 1200 --disable-llm
```

**Flags:**

| Flag | Required | Description |
|---|---|---|
| `--dataset` | yes | `"b2t_24"` or `"b2t_25"` |
| `--model-mode` | yes | `"transformer"` or `"gru"` |
| `--base-path` | yes | Base directory for results and adapter paths (e.g. `/home/user`) |
| `--brain2text-dir` | yes | Directory containing `brain2text/` folder |
| `--model-template` | yes | Model name template with `{seed}` placeholder |
| `--seeds` | yes | Seeds to run |
| `--logits-base` | no | If provided, skips logit generation and decodes from this path |
| `--device` | no | Device for both logit generation and decoding (e.g. `cuda:0`) |

**Outputs** (saved to `results_dir` from `decoder_config.py`):
- `<model>_<timestamp>.csv` — predicted transcripts (`id`, `text`)
- `<model>_<timestamp>.json` — hyperparameters and aggregate metrics (WER, RTF, VRAM)
- `<model>_<timestamp>_beams.json` — top-K beam hypotheses per trial

---

### Notes

- **`train.py`** is configured by editing constants at the top of the file, not via command-line flags.
- **`generate_logits.py`** and **`run_decoder.py`** are called automatically by `batch_decode.py` but can also be run standalone — use `--help` for options.
- **`decoder_config.py`** provides defaults for the decoder; any value can be overridden with a flag at runtime.
- Experiment tracking uses [Weights & Biases](https://wandb.ai/). Pass `--no-wandb` to disable it.
