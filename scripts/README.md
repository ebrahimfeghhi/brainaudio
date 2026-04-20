## Scripts

This directory contains the end-to-end pipeline for training brain-to-text acoustic models and running the Lightbeam CTC decoder.

### Pipeline Overview

```
1. Format data        →  scripts/dataset/
2. Train model        →  scripts/train.py
3. Generate logits    →  scripts/generate_logits.py
4. Finetune LLM       →  scripts/finetune_llm.py
5. Run decoder        →  scripts/run_decoder.py
```

---

### Directory Structure

```
scripts/
├── README.md                     # This file
├── decoder_config.py             # Default hyperparameters and paths for the decoder (edit before first run)
├── custom_config.yaml            # Custom training config (edit before running train.py)
├── train.py                      # Train a CTC acoustic encoder (Transformer or GRU)
├── generate_logits.py            # Run inference and save encoder logits to disk
├── run_decoder.py                # Run the Lightbeam CTC beam search decoder
├── batch_decode.py               # Run the decoder over multiple models/seeds in sequence
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

### Step 3 — Generate Logits

`generate_logits.py` loads trained encoder checkpoints (Transformer or GRU) and saves the raw CTC logits to `.npz` files, which are the input to the decoder.

**Before running**, edit the constants at the top of `generate_logits.py`:

```python
MODEL_NAME_TEMPLATES = ["your_model_name_seed_{seed}"]
SEEDS = [0, 1, 2]                  # Seeds to generate logits for
local_model_folder = "b2t_25"      # "b2t_24" or "b2t_25"
modelWeightsFilesList = ["modelWeights_PER_25"]
PARTITION = "val"                  # "val" or "test"
DEVICE = "cuda:0"
MODEL_TYPE = "transformer"         # "transformer" or "gru"
```

```bash
uv run scripts/generate_logits.py
```

Logits are saved as `logits_{partition}.npz` inside a subdirectory named after the model. PER (Phoneme Error Rate) results are saved to `results/per_results/`.

---

### Step 4 — Finetune the LLM

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

### Step 5 — Run the Decoder

`run_decoder.py` takes saved logits and runs the Lightbeam CTC beam search decoder with an optional LLM shallow fusion stage.

```bash
# Decode validation set with default settings from decoder_config.py
uv run scripts/run_decoder.py \
    --logits-val-path /path/to/logits/my_model/logits_val.npz

# Decode without LLM (n-gram LM only, much faster)
uv run scripts/run_decoder.py \
    --logits-val-path /path/to/logits/my_model/logits_val.npz \
    --disable-llm

# Decode val + test in one run
uv run scripts/run_decoder.py \
    --logits-val-path /path/to/logits/my_model/logits_val.npz \
    --logits-test-path /path/to/logits/my_model/logits_test.npz

# Override key hyperparameters
uv run scripts/run_decoder.py \
    --logits-val-path /path/to/logits/my_model/logits_val.npz \
    --beam-size 1200 \
    --acoustic-scale 0.5 \
    --alpha-ngram 1.0 \
    --llm-weight 1.5
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--logits-val-path` | — | Path to val logits `.npz` |
| `--logits-test-path` | — | Path to test logits `.npz` |
| `--logits-train-path` | — | Path to train logits `.npz` |
| `--disable-llm` | off | Skip LLM shallow fusion; use n-gram LM only |
| `--beam-size` | dataset-dependent | CTC beam width |
| `--acoustic-scale` | dataset-dependent | Scale applied to log-probs after softmax |
| `--alpha-ngram` | dataset-dependent | N-gram LM weight during beam search |
| `--llm-weight` | 1.2 | LLM score weight during rescoring |
| `--model` | `Llama-3.2-3B` | HuggingFace model ID for LLM fusion |
| `--lora-adapter` | config default | Path to fine-tuned LoRA adapter |
| `--no-adapter` | off | Use base LLM without a LoRA adapter |
| `--load-in-4bit` | off | Load LLM in 4-bit quantization (reduces VRAM) |
| `--random N` | — | Randomly sample N trials (seed=42) |
| `--trial-indices 0 5 10` | — | Decode only specific trial indices |
| `--device` | `cuda:0` | Torch device |
| `--no-wandb` | off | Disable W&B logging |
| `--verbose` | off | Print per-beam details for each trial |

Run `uv run scripts/run_decoder.py --help` for the full list of options.

**Outputs** (saved to `results_dir` from `decoder_config.py`):
- `<model>_<timestamp>.csv` — predicted transcripts (`id`, `text`)
- `<model>_<timestamp>.json` — hyperparameters and aggregate metrics (WER, RTF, VRAM)
- `<model>_<timestamp>_beams.json` — top-K beam hypotheses per trial

---

### Step 5 (batch) — Batch Decode Multiple Models

`batch_decode.py` loops over a list of logit paths and calls `run_decoder.py` for each. Any additional arguments are passed through to the decoder.

All user-facing options are CLI flags:

```bash
# Run on val set across seeds 0–4
uv run scripts/batch_decode.py \
    --dataset b2t_25 \
    --model-mode transformer \
    --base-path /home/user \
    --logits-base /data2/brain2text/b2t_25/logits \
    --model-template "my_model_seed_{seed}" \
    --seeds 0 1 2 3 4 \
    --val

# Override decoder hyperparameters for all runs
uv run scripts/batch_decode.py \
    --dataset b2t_25 \
    --model-mode gru \
    --base-path /home/user \
    --logits-base /data2/brain2text/b2t_25/logits \
    --model-template "my_gru_seed_{seed}" \
    --seeds 0 1 2 \
    --val --test \
    --beam-size 1200 --disable-llm
```

**Required flags:**

| Flag | Description |
|---|---|
| `--dataset` | `"b2t_24"` or `"b2t_25"` |
| `--model-mode` | `"transformer"` or `"gru"` |
| `--base-path` | Base directory for results and adapter paths (e.g. `/home/user`) |
| `--logits-base` | Directory containing per-model logits folders |
| `--model-template` | Folder name template with `{seed}` placeholder |
| `--seeds` | Seeds to decode |

---

### Notes

- **`train.py` and `generate_logits.py`** are configured by editing constants at the top of the file, not via command-line flags.
- **`run_decoder.py`** and **`compare_predictions.py`** are fully CLI-driven — run with `--help` to see all options.
- **`decoder_config.py`** provides defaults for the decoder; any value can be overridden with a flag at runtime.
- Experiment tracking uses [Weights & Biases](https://wandb.ai/). Pass `--no-wandb` to disable it.
