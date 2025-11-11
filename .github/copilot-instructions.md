<!-- Copilot / AI agent guidance for contributors and automated agents -->
# Copilot instructions — brainaudio

This file captures concise, project-specific knowledge to make an AI coding agent immediately productive.

- Repo layout (important places)
  - `src/brainaudio/` — main package. Prefer edits here for runtime code.
  - `scripts/` — utility scripts and dataset conversion tools. Example: `scripts/dataset_scripts/brain2text_2025.py` shows how raw HDF5 data is reformatted.
  - `src/brainaudio/datasets/README.md` — canonical description of the dataset format (keys: `train`, `val`, `test` and per-day dictionaries with `sentenceDat`, `transcriptions`, `text`, `timeSeriesLen`, `textLens`, `forced_alignments`). Use this when editing data converters or loader logic.
  - `notebooks/` — exploratory notebooks and reproducibility checks. Useful for quick data inspections and examples.
  - `wandb/` — contains offline run artifacts and logs. When changing training or logging behavior, update or inspect these folders for examples.

- Quick start / environment notes
  - This project uses `uv` for lightweight environment management in README.md. Common commands from the top-level README:
    - `uv sync` — creates and sets up a venv at `.venv` (then `source .venv/bin/activate`).
    - WFST language model uses a separate venv: `uv venv .wfst -p 3.9` then `source .wfst/bin/activate` and `uv pip install -r requirements.txt`.
  - The WFST-based language model must be copied into `third_party/language_model` (see README). When working on language model integration, preserve the copy step and installation path under `.wfst`.

- Data & format conventions (very important)
  - The canonical format is a hierarchical dict with top-level keys `train`, `val`, `test`.
  - Each top-level value is a list of per-day dictionaries (ordered in time). A missing day is represented by `None`.
  - Per-day dictionary keys (examples found in `src/brainaudio/datasets/README.md` and `scripts/dataset_scripts/brain2text_2025.py`):
    - `sentenceDat`: list of 2D numpy arrays (T x N) — neural features per trial.
    - `transcriptions`: list of strings (ground-truth sentence); for test set this may be a filler string.
    - `text`: list of integer arrays (phoneme ids padded to M).
    - `timeSeriesLen`: list of integers (length of neural trial in time points).
    - `textLens`: list of integers (phoneme lengths per trial).
    - `forced_alignments`: optional list where each element is a dict mapping word end frame -> word.
  - Dataset conversion scripts follow this structure. When creating or modifying loaders, return the same dictionary shape.

- Important scripts & entry points
  - `scripts/dataset_scripts/brain2text_2025.py` — canonical example of reformatting raw hdf5 into the project format; references `DATA_DIR` and `OUT_DIR` constants near the top.
  - `scripts/save_transcript.py`, `scripts/save_logits.py`, `scripts/save_config.py` — utility scripts used during experiments/training. Inspect them for I/O patterns.
  - `scripts/call_trainer.py` and `debugging/e2e_trainer.py` — places to look for training pipeline calls and debug hooks.

- Project-specific patterns & conventions
  - Typed package: `src/brainaudio/py.typed` exists — prefer adding type hints and keeping public APIs typed.
  - Data-processing scripts may use absolute paths or module-level constants for `DATA_DIR` / `OUT_DIR` (see `brain2text_2025.py`). Agents should search for and preserve or parameterize these constants rather than hardcoding new paths.
  - Notebooks are used as living documentation — keep them up-to-date when changing data formats or training behavior.
  - Many scripts are permissive about missing files and will warn and continue (see `brain2text_2025.py` session directory checks). Follow existing defensive patterns.

- Integration points & external dependencies
  - WFST-based LM: external repo `nejm-brain-to-text` must be copied into `third_party/language_model` and built inside the `.wfst` venv (see README).
  - wandb is used for logging; training scripts write runs into `wandb/` folder. Avoid removing or overwriting runs unless intentionally cleaning experiments.

- Developer workflows (handy commands)
  - Create / sync main venv and activate:
    - `uv sync`
    - `source .venv/bin/activate`
  - WFST LM specific (only when working with WFST):
    - `uv venv .wfst -p 3.9`
    - `source .wfst/bin/activate`
    - `uv pip install -r requirements.txt`
  - Reformat dataset example:
    - Edit `DATA_DIR` / `OUT_DIR` in `scripts/dataset_scripts/brain2text_2025.py` then run `python scripts/dataset_scripts/brain2text_2025.py`.

- How the agent should behave when editing code
  - Preserve the project data shape exactly when changing dataset code; point to `src/brainaudio/datasets/README.md` for the contract.
  - Prefer updating `src/brainaudio/` package modules rather than ad-hoc scripts when implementing core behavior.
  - Keep changes small and well-scoped; run the notebooks or a small script to verify I/O when modifying data pipelines.

- Where to look for examples
  - Data-conversion example: `scripts/dataset_scripts/brain2text_2025.py` (HDF5 -> pickled format matching `src/brainaudio/datasets/README.md`).
  - Training/debug: `scripts/call_trainer.py`, `debugging/e2e_trainer.py`.

If anything is unclear or you want the instructions to emphasize other areas (tests, CI hooks, or specific modules), tell me which area and I will iterate.
