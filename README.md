# BrainAudio

**[LightBeam: An Accurate and Memory-Efficient CTC Decoder for Speech Neuroprostheses](https://arxiv.org/abs/2603.14002)**

---

## General Installation Instructions

The following are general installation instructions for the main package. Instructions for the WFST-based language model environment (used in Willett et al., 2023 and Card et al., 2024) can be found at the bottom of this file.

1. Install the `uv` package. [Instructions can be found here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

2. `cd` into the outer `brainaudio` directory and run `uv sync`.

3. `uv sync` creates a python virtual environment, which can be activated through `source .venv/bin/activate`

### Test Installation
Ensure your installation is correct by running the following commands:
(a bash script )

---

## Dataset Preparation

The training pipeline expects data in a trial-level directory format with a manifest JSON. The scripts below convert the raw competition data into this format.

### B2T '24

Raw data is provided as `.mat` files from the competition. Edit the paths at the top of `scripts/dataset/brain2text_2024.py` (`DATA_DIR`, `SAVE_DIR`), then run:

```bash
uv run python scripts/dataset/brain2text_2024.py
```

This loads each session's `.mat` file, extracts threshold crossings and spike band power (256 features: 2 per electrode × 128 electrodes), applies block-wise z-score normalization, converts transcriptions to phoneme IDs, and saves the result as a single `.pkl` file.

### B2T '25

Raw data is downloaded from Dryad (DOI: `10.5061/dryad.dncjsxm85`) and provided as `.hdf5` files. Edit `DATA_DIR` and `OUT_DIR` at the top of `scripts/dataset/brain2text_2025.py`, uncomment `download_dataset()` if needed, then run:

```bash
uv run python scripts/dataset/brain2text_2025.py
```

This reformats 45 recording sessions into the same `.pkl` structure used by B2T '24. B2T '25 has 512 features (2 per electrode × 256 electrodes) and block-wise normalization is handled at load time rather than preprocessing time.

### Creating the trial-level manifest

Both benchmarks use the same final step. Edit the `data_paths` and `output_dirs` at the top of `scripts/dataset/lazyload_format.py`, then run:

```bash
uv run python scripts/dataset/lazyload_format.py
```

This splits the `.pkl` into one directory per trial (`sentenceDat.npy`, `text.npy`, `meta.json`) and writes a `manifest.json` pointing to all trial paths. The manifest path is what goes into the training config under `manifest_paths`.

---

## Training CTC Encoders

### Overview

Encoders are trained with `scripts/train.py`. The script reads a YAML config file, builds the model, and calls `trainModel` for each seed listed in the config. Configs are stored in `src/brainaudio/training/utils/custom_configs/`.

### Workflow

**Step 1 — Create or edit a config**

Copy an existing config or edit `scripts/custom_config.yaml` directly. The key fields to set are:

| Field | Description |
|---|---|
| `modelName` | Name used for output directories and checkpoint filenames |
| `modelType` | `transformer` or `gru` |
| `seeds` | List of random seeds to train (one run per seed) |
| `outputDir` | Directory where checkpoints and logs are saved |
| `manifest_paths` | Path(s) to the dataset manifest JSON |
| `model.transformer` / `model.gru` | Architecture hyperparameters |

**Step 2 — Save the config** *(optional, if editing `custom_config.yaml`)*

```bash
cd scripts
uv run python save_config.py
```

This saves `custom_config.yaml` into `src/brainaudio/training/utils/custom_configs/<modelName>.yaml`.

**Step 3 — Launch training**

Edit the `config_path` and `device` variables at the top of `scripts/train.py`, then run:

```bash
cd scripts
uv run python train.py
```

The script iterates over all seeds in `config["seeds"]`, appending `_seed_<N>` to `modelName` for each run. For architecture-level config details see [`src/brainaudio/models/README.md`](src/brainaudio/models/README.md).

Configs for the four models used in the paper — baseline GRUs and time-masked Transformers for both B2T '24 and B2T '25 — are provided in `src/brainaudio/training/utils/custom_configs/`.

### Model types

Set `modelType` to one of:

| `modelType` | Class | Config section |
|---|---|---|
| `gru` (year `'2024'`) | `GRU_24` | `model.gru` |
| `gru` (year `'2025'`) | `GRU_25` | `model.gru` |
| `transformer` | `TransformerModel` (chunking) | `model.transformer` |

---

## Generating Logits

After training, run `scripts/generate_logits.py` to load a trained encoder and save phoneme logits to disk. These logits are the input to the decoder.

Edit the config block at the top of the script:

| Field | Description |
|---|---|
| `MODEL_NAME_TEMPLATES` | List of model name templates with a `{seed}` placeholder |
| `SEEDS` | Seeds to run |
| `local_model_folder` | `"b2t_24"` or `"b2t_25"` — determines data paths and eval chunk config |
| `modelWeightsFilesList` | Checkpoint filename(s) to load (e.g. `modelWeights_PER_25`) |
| `PARTITION` | `"val"` or `"test"` |
| `DEVICE` | CUDA device |

Then run:

```bash
cd scripts
uv run python generate_logits.py
```

Logits are saved to `data2/brain2text/<benchmark>/logits/<model_name>/`. For Transformer models, the eval chunk size and context window are set automatically based on `local_model_folder` (1s chunk / 7.5s context for B2T '24; 1s chunk / 20s context for B2T '25). These settings are not used for GRU models.

---

## Decoding

Decoding uses the Lightbeam decoder (see paper link above), which runs CTC beam search with word-level n-gram LM fusion and optional LLM shallow fusion.

### Decoder config

Hyperparameters are set in `scripts/decoder_config.py`. Tuned configs for both benchmarks are already saved — the correct one is selected automatically via the `--dataset` flag. The LLM used for shallow fusion can also be changed there. The token list and lexicon required for decoding are included in the `shallow_fusion/` directory.

### Finetuning the LLM

The Lightbeam decoder uses a LoRA-finetuned Llama 3.2-1B for shallow fusion. The model used in the paper was finetuned on the B2T training transcripts using `finetune_llm/finetune_llm.py`.

```bash
uv run python finetune_llm/finetune_llm.py \
    --model-name meta-llama/Llama-3.2-1B \
    --transcript-files data/transcripts_merged_normalized.txt \
    --output-dir /path/to/save/adapter \
    --num-epochs 3 \
    --batch-size 16 \
    --learning-rate 2e-4 \
    --device 0
```

The transcript file should contain one sentence per line, with a `# VALIDATION` comment separating the train and val splits. The best checkpoint (lowest validation perplexity) is saved to `--output-dir`. Once trained, point `lora_adapter_1b` in `scripts/decoder_config.py` to this directory.

### Generative Error Correction (GEC)

`finetune_llm/finetune_gec.py` fine-tunes an instruction-tuned LLM (Llama 3.1-8B by default) to correct beam search hypotheses. This is an optional post-processing step applied after decoding. It uses [Unsloth](https://github.com/unslothai/unsloth) for 4-bit quantized LoRA fine-tuning.

Edit the config block at the top of the script:

| Field | Description |
|---|---|
| `YEAR` | `"b2t_24"` or `"b2t_25"` |
| `MODE` | `"uni"` or `"bi"` — selects the JSONL files to use |
| `MODEL_NAME` | Unsloth model identifier (default: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`) |
| `OUTPUT_DIR` | Directory where the LoRA adapter is saved |
| `EPOCH_NUM` | Number of training epochs |

Input data should be JSONL files in `finetune_llm/jsonl_files/` named `train_<year>_wfst_<mode>.jsonl` and `val_<year>_wfst_<mode>.jsonl`. Each record has a `prompt` field (the ASR hypothesis) and a `completion` field (the corrected text). Then run:

```bash
uv run python finetune_llm/finetune_gec.py
```

The adapter is saved to `OUTPUT_DIR/<model_shortname>_final_lora_<year>_wfst_<mode>_seed_<seed>`. Loss is computed only on the assistant responses (not the input hypothesis).

### Running the decoder

Use `scripts/batch_decode.py` to decode logits across multiple seeds:

```bash
cd scripts
uv run python batch_decode.py \
    --dataset b2t_25 \
    --model-mode transformer \
    --logits-base /data2/brain2text/b2t_25/logits \
    --model-template "neurips_b2t_25_causal_transformer_seed_{seed}_PER_25" \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --val \
    --device cuda:0
```

| Argument | Description |
|---|---|
| `--dataset` | `b2t_24` or `b2t_25` — selects decoder hyperparameters and results directory |
| `--model-mode` | `gru` or `transformer` — selects results subdirectory |
| `--logits-base` | Base directory containing per-model logit folders |
| `--model-template` | Folder name template with `{seed}` placeholder |
| `--seeds` | Seeds to decode |
| `--val` / `--test` | Partition(s) to decode |
| `--device` | CUDA device |

Results are written to `results/<gru_24|transformer_24|gru_25|transformer_25>/`.

---

## Ongoing Work

- **GRU '25 replication:** We are still verifying that our `GRU_25` implementation matches the NEJM codebase. Note that the models reported in the paper were trained using the NEJM codebase, not this one.
- **Hardcoded file paths:** Several scripts still have hardcoded paths (e.g. `generate_logits.py`, `decoder_config.py`). We are working on making these more flexible for users on different systems.
- **4-gram language model:** Lightbeam requires the Huge 4-gram language model, which is not included in this repo. It can be downloaded in ARPA format from [imagineville.org](https://imagineville.org/software/lm/dec19/). Once downloaded, update the `word_lm` path in `scripts/decoder_config.py`.

---

## Installation Instructions for Language Model (WFST-based)

Required only for reproducing the WFST-based decoding results from Willett et al., 2023 and Card et al., 2024. This uses a separate Python environment from the main package.

1. `cd` into the outer `brainaudio` directory and run `uv venv .wfst -p 3.9`.

2. Activate the environment with `source .wfst/bin/activate`

3. Run `uv pip install -r requirements.txt`

4. Clone the following repository outside this repository: [NEJM repo](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text).

5. Create a directory called `third_party`. After creating this directory, your project structure should look like this:

```
    brainaudio/
    ├── src/
    │   └── brainaudio/
    └── third_party/ <-- Create this folder
```

6. Copy the `language_model` directory from the NEJM repo into `third_party`: `cp -r nejm-brain-to-text/language_model brainaudio/third_party`

7. Run `cd third_party/language_model/runtime/server/x86` and then `python setup.py install`. Make sure this command is run in the `.wfst` venv.
