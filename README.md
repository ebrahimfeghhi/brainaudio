## General Installation Instructions

The following are general installation instructions for the main package. The WFST-based language model
used in Willett et al., 2023 and Card et al., 2024 requires a different environment, and instructions can be found in the `Installation Instructions for Language Model (WFST-based)` section.

1. Install the `uv` package. [Instructions can be found here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

2. `cd` into the outer `brainaudio` directory and run `uv sync`.

3. `uv sync` creates a python virtual environment, which can be activated through `source .venv/bin/activate`

### Installation Instructions for Language Model (WFST-based)

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

6. Copy the `language_model` directory from the `NEJM_repo` into the `third_party` directory: `cp -r nejm-brain-to-text/language_model brainaudio/third_party`

7. Run `cd third_party/language_model/runtime/server/x86` and then `python setup.py install`. Make sure this command is run in the `.wfst` venv

### Environment Variables Configuration

1. Run `pip install -e .` in the terminal or `python -m pip install -e . --no-deps` if there are dependency issues from `pyproject.toml`

2. Import constants (hard-coded addresses) from `brainaudio.utils.config` into individual scripts + notebooks
