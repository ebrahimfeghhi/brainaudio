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

## Repository Structure

| Directory | Description |
|---|---|
| [`src/brainaudio/`](src/brainaudio/README.md) | Core Python package — models, training, inference, datasets |
| [`scripts/`](scripts/README.md) | End-to-end pipeline scripts |
| `auxiliary_folders/` | Supporting tools (LLM finetuning, GEC, shallow fusion assets) |
| `results/` | Decoder output CSVs and PER/WER summaries |

---

## Pipeline

The full pipeline is documented in [`scripts/README.md`](scripts/README.md). A summary of each step is below.

### 1. Format Data

Convert raw competition data into the trial-level pickle format expected by the training pipeline.

```bash
uv run scripts/dataset/brain2text_2025.py   # or brain2text_2024.py
uv run scripts/dataset/lazyload_format.py
```

### 2. Train the Encoder


Train a CTC acoustic encoder (Transformer or GRU) over multiple seeds. Edit `config_path` and `device` at the top of `scripts/train.py`, then:

```bash
uv run scripts/train.py
```

Configs are in `src/brainaudio/training/utils/custom_configs/`. Configs for the four models used in the paper (baseline GRUs and Transformers for B2T '24 and '25) are provided.

### 3. Finetune the LLM

SFT fine-tune a causal LLM with LoRA on transcript data. The resulting adapter is used by the decoder for shallow fusion rescoring.

```bash
uv run scripts/finetune_llm.py \
    --model-name meta-llama/Llama-3.2-1B \
    --transcript-files /path/to/transcripts_merged_normalized.txt \
    --output-dir /path/to/save/adapter
```

The transcript file should have one sentence per line with a `# VALIDATION` comment separating train and val splits.

### 4. Generate Logits + Decode

Generate logits and decode with the Lightbeam CTC beam search decoder across multiple seeds in one call:

```bash
uv run scripts/batch_decode.py \
    --dataset b2t_25 \
    --model-mode transformer \
    --base-path /home/user \
    --brain2text-dir /home/user/data2 \
    --model-template "neurips_b2t_25_causal_transformer_seed_{seed}" \
    --seeds 0 1 2 3 4 \
    --val \
    --device cuda:0
```

Logits are generated automatically before each decode. If logits already exist, pass `--logits-base` to skip generation. Hyperparameters are in `scripts/decoder_config.py` — tuned defaults for both benchmarks are already set. The token list and lexicon required for decoding are in `auxiliary_folders/shallow_fusion/`.

> **Note:** Lightbeam requires the Huge 4-gram LM, not included in this repo. Download in ARPA format from [imagineville.org](https://imagineville.org/software/lm/dec19/) and update `word_lm` in `scripts/decoder_config.py`.

### Generative Error Correction (GEC) *(optional)*

Fine-tune an instruction-tuned LLM to correct beam search hypotheses as a post-processing step. Edit the config block at the top of `auxiliary_folders/finetune_llm/finetune_gec.py`, then:

```bash
uv run auxiliary_folders/finetune_llm/finetune_gec.py
```

---

## Ongoing Work

- **GRU '25 replication:** We are still verifying that our `GRU_25` implementation matches the NEJM codebase. Note that the models reported in the paper were trained using the NEJM codebase, not this one.

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
