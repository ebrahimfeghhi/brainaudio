## Scripts

This directory contains the end-to-end pipeline for training brain-to-text acoustic models and running the Lightbeam CTC decoder.

### Pipeline Overview

```
1. Format data      →  scripts/dataset/
2. Train model      →  scripts/train.py
3. Generate logits  →  scripts/generate_logits.py
4. Run decoder      →  scripts/run_decoder.py
5. Compare results  →  scripts/compare_predictions.py
```

---

### Directory Structure

```
scripts/
├── README.md                     # This file
├── decoder_config.py             # Default hyperparameters and paths for the decoder (edit before first run)
├── train.py                      # Train a CTC acoustic encoder (Transformer or GRU)
├── generate_logits.py            # Run inference and save encoder logits to disk
├── run_decoder.py                # Run the Lightbeam CTC beam search decoder
├── batch_decode.py               # Run the decoder over multiple models/seeds in sequence
├── compare_predictions.py        # Compare decoder predictions against a baseline and compute WER
├── check_vocab.py                # Check for out-of-vocabulary words in a transcript file
├── save_transcript.py            # Save model predictions as transcript files
├── save_config.py                # Save a training configuration to YAML
├── dataset/
│   ├── brain2text_2025.py        # Download and format the B2T-25 dataset (HDF5 → pickle)
│   ├── brain2text_2024.py        # Download and format the B2T-24 dataset (HDF5 → pickle)
│   └── lazyload_format.py        # Convert datasets to lazy-loading format for memory-efficient training
├── hpo/
│   ├── search_train_hparams.py   # Optuna-based HPO for encoder training hyperparameters
│   ├── search_decode_hparams.py  # Optuna-based HPO for Lightbeam decoder hyperparameters
│   ├── hpo_trainer.py            # Single trial runner called by search_train_hparams.py
│   ├── save_hpo_configs.py       # Generate and save HPO configuration files
│   ├── adapt_hpo_configs.py      # Adapt saved HPO configs for different settings
│   └── compare_trial_configs.py  # Compare configurations across HPO trials
└── _archive/                     # Deprecated scripts kept for reference — not intended for use
```

---

### Configuration: `decoder_config.py`

Before running the decoder, edit `decoder_config.py` to set your local paths and select the dataset:

```python
DATASET = "b2t_25"   # Switch between "b2t_24" and "b2t_25"

base_path = "/your/data/root"   # Root directory for data files
```

Key path fields to update:

| Field | Description |
|---|---|
| `PATHS["tokens"]` | Phoneme units file (`units_pytorch.txt`) |
| `PATHS["lexicon"]` | Phoneme lexicon file |
| `PATHS["word_lm"]` | KenLM 4-gram language model (`.kenlm`) |
| `PATHS["transcripts_val"]` | Ground truth val transcripts (`.pkl`) |
| `PATHS["lora_adapter_*"]` | Fine-tuned LLM adapter paths |
| `PATHS["results_dir"]` | Where decoder output CSVs are saved |

Beam search and LLM hyperparameters in `decoder_config.py` are tuned per dataset and serve as defaults that can be overridden via command-line flags at runtime.

---

### Step 1 — Format Data

Download and convert a dataset to the internal pickle format used by the training pipeline.

```bash
# Format the B2T-25 dataset
python scripts/dataset/brain2text_2025.py

# Format the B2T-24 dataset
python scripts/dataset/brain2text_2024.py
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

```bash
python scripts/train.py
```

Checkpoints and metrics are saved to the path in the config and logged to Weights & Biases.

---

### Step 3 — Generate Logits

`generate_logits.py` loads trained encoder checkpoints and saves the raw CTC logits to `.npz` files, which are the input to the decoder.

**Before running**, edit the constants at the top of `generate_logits.py`:

```python
MODEL_NAME_TEMPLATE = "your_model_name_seed_{seed}"
SEEDS = [0, 1, 2]           # Seeds to generate logits for
DEVICE = "cuda:0"
PARTITION = "val"            # "train", "val", or "test"
MANIFEST_PATHS = ["/path/to/manifest.json"]
SAVE_PATHS = {0: "/path/to/save/logits/"}
```

```bash
python scripts/generate_logits.py
```

Logits are saved as `logits_{partition}.npz` inside a subdirectory named after the model. PER (Phoneme Error Rate) results are saved to `results/per_results/`.

---

### Step 4 — Run the Decoder

`run_decoder.py` takes saved logits and runs the Lightbeam CTC beam search decoder with an optional LLM shallow fusion stage.

```bash
# Decode validation set with default settings from decoder_config.py
python scripts/run_decoder.py \
    --logits-val-path /path/to/logits/my_model/logits_val.npz

# Decode without LLM (n-gram LM only, much faster)
python scripts/run_decoder.py \
    --logits-val-path /path/to/logits/my_model/logits_val.npz \
    --disable-llm

# Decode val + test in one run
python scripts/run_decoder.py \
    --logits-val-path /path/to/logits/my_model/logits_val.npz \
    --logits-test-path /path/to/logits/my_model/logits_test.npz

# Override key hyperparameters
python scripts/run_decoder.py \
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

Run `python scripts/run_decoder.py --help` for the full list of options.

**Outputs** (saved to `results_dir` from `decoder_config.py`):
- `<model>_<timestamp>.csv` — predicted transcripts (`id`, `text`)
- `<model>_<timestamp>.json` — hyperparameters and aggregate metrics (WER, RTF, VRAM)
- `<model>_<timestamp>_beams.json` — top-K beam hypotheses per trial

---

### Step 4 (batch) — Batch Decode Multiple Models

`batch_decode.py` loops over a list of logit paths and calls `run_decoder.py` for each. Any additional arguments are passed through to the decoder.

**Before running**, edit the lists at the top of `batch_decode.py`:

```python
seed_list  = [0, 1, 2, 3]
VAL_PATHS  = [f"/path/to/logits/my_model_seed_{i}/logits_val.npz" for i in seed_list]
TEST_PATHS = [None] * len(seed_list)    # Set to None to skip test
TRAIN_PATHS = [None] * len(seed_list)
SAVE_NAMES = [f"run_{i}" for i in seed_list]   # Output filename prefix per run
```

```bash
# Run with default decoder settings
python scripts/batch_decode.py

# Override decoder settings for all runs
python scripts/batch_decode.py --beam-size 1200 --disable-llm
```

---

### Step 5 — Compare Predictions

`compare_predictions.py` compares decoder output against a baseline CSV, reporting WER improvements and regressions at the sentence level.

```bash
python scripts/compare_predictions.py \
    --results-csv my_model_03_15_1200.csv \
    --baseline /path/to/baseline_predictions.csv \
    --transcripts /path/to/transcripts_val_cleaned.pkl \
    --show-all-diff
```

**Key flags:**

| Flag | Description |
|---|---|
| `--results-csv` | CSV filename from the decoder (looked up in `results/`) |
| `--baseline` | Path to baseline predictions CSV |
| `--transcripts` | Path to ground truth transcripts pickle |
| `--show-better` | Print sentences where the decoder outperforms the baseline |
| `--show-worse` | Print sentences where the decoder underperforms the baseline |
| `--show-all-diff` | Print all sentences with differing predictions (default) |

**Output:** Prints a per-error-type breakdown (substitutions, insertions, deletions) and saves a comparison CSV to `results/`.

---

### Hyperparameter Search (`hpo/`)

The `hpo/` directory contains Optuna-based hyperparameter optimization scripts.

```bash
# Search for best encoder training hyperparameters
python scripts/hpo/search_train_hparams.py

# Search for best decoder hyperparameters
python scripts/hpo/search_decode_hparams.py
```

Both scripts save trial results and best configs locally. Edit the configuration constants at the top of each script before running.

---

### Notes

- **`train.py` and `generate_logits.py`** are configured by editing constants at the top of the file, not via command-line flags.
- **`run_decoder.py`** and **`compare_predictions.py`** are fully CLI-driven — run with `--help` to see all options.
- **`decoder_config.py`** provides defaults for the decoder; any value can be overridden with a flag at runtime.
- Experiment tracking uses [Weights & Biases](https://wandb.ai/). Pass `--no-wandb` to disable it.
