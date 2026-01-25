import os
import json
import subprocess
import re
from pathlib import Path
import optuna
from optuna.samplers import QMCSampler
from datetime import datetime
import wandb

# ===================================================================
#                       CONFIGURATION
# ===================================================================

HPO_PROJECT_NAME = "neural-decoder-qmc"
DEVICE = "cuda:0"
N_TRIALS = 50

# Path to run_decoder.py script
RUN_DECODER_SCRIPT = Path(__file__).parent.parent / "run_decoder.py"

# Encoder model and data paths
ENCODER_MODEL_NAME = "pretrained_RNN"
RESULTS_DIR = Path("/data2/brain2text/hpo/decoder_hpo")

# Define hyperparameter search space
# Format: "param_name": (min, max) for floats, or (min, max) for ints
# Fixing the acoustic scale, fixing the beam-prune-threshold
HPO_RANGES = {
    "alpha-ngram": (0.4, 1.2),                    # N-gram LM weight    <default: 0.8>
    "lm-weight": (0.5, 1.5),                      # Neural LM fusion weight  <default: 1.0>
    "beam-beta": (0.5, 2),                      # Bonus for extending beams    <default: 1.5>
    "word-boundary-bonus": (0.5, 1.5),            # Bonus for word boundary token   <default: 1.0>
}

# Fixed parameters (not being optimized)
FIXED_PARAMS = {
    "encoder-model-name": ENCODER_MODEL_NAME,
    "device": DEVICE,
    "model": "meta-llama/Llama-3.2-3B",
    "lora-adapter": "/home/ebrahim/brainaudio/llama-3.2-3b-hf-finetuned",
    "load-in-4bit": True,
    "word-lm-path": "/data2/brain2text/lm/lm_dec19_huge_4gram.kenlm",
    "beam-size": 300,
    "logit-scale": 0.4,  # Fixed to default value to decouple interactions
    "word-insertion-bonus": 0.0,
    "top-k": 1,
    "lm-rescore-interval": 10,
    "beta-ngram": 0,
    "beam-prune-threshold": 12.0,
    "homophone-prune-threshold": 4.0,
    "num-homophone-beams": 3
}

# ===================================================================
#                       DECODER OBJECTIVE FUNCTION
# ===================================================================

def run_decoder_with_params(trial):
    """
    Run run_decoder.py with trial hyperparameters and extract WER from output.
    """
    # Sample hyperparameters
    params = {}
    params["alpha-ngram"] = trial.suggest_float("alpha-ngram", *HPO_RANGES["alpha-ngram"])
    params["lm-weight"] = trial.suggest_float("lm-weight", *HPO_RANGES["lm-weight"])
    params["beam-beta"] = trial.suggest_float("beam-beta", *HPO_RANGES["beam-beta"])
    params["word-boundary-bonus"] = trial.suggest_float("word-boundary-bonus", *HPO_RANGES["word-boundary-bonus"])

    # Generate unique results filename for this trial
    results_filename = f"hpo_trial_{trial.number}"
    params["results-filename"] = results_filename

    # Build command line arguments
    # Use uv run (without python) to execute the script in the virtual environment
    cmd = ["uv", "run", str(RUN_DECODER_SCRIPT)]

    # Add fixed parameters
    for key, value in FIXED_PARAMS.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{key}", str(value)])

    # Add trial hyperparameters
    for key, value in params.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{key}", str(value)])

    print(f"\n=== Trial {trial.number} ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"Hyperparameters:")
    for key, value in params.items():
        if key != "results-filename":
            print(f"  {key}: {value}")

    # Run decoder
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 60 minute timeout
        )

        # Parse WER from output
        # Look for line like "Aggregate WER: 0.1234"
        wer_match = re.search(r"Aggregate WER:\s+([\d.]+)", result.stdout)

        if wer_match:
            wer = float(wer_match.group(1))
            print(f"  WER: {wer:.4f}")
        else:
            print(f"Warning: Could not parse WER from output")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            wer = 1.0  # Return high WER if parsing failed

        # Store full output for debugging
        trial.set_user_attr("stdout", result.stdout)
        trial.set_user_attr("stderr", result.stderr)
        trial.set_user_attr("return_code", result.returncode)

        return wer

    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out")
        return 1.0  # Return high WER on timeout
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1.0  # Return high WER on error


def objective(trial):
    """
    Optuna objective function.
    """
    try:
        wer = run_decoder_with_params(trial)

        # Store WER in user attributes
        trial.set_user_attr("wer", wer)

        # Log to wandb
        wandb.log({
            "trial_number": trial.number,
            "wer": wer,
            **trial.params
        })

        # Prune trials with very bad WER
        if wer > 0.8:
            print(f"Trial {trial.number} pruned due to high WER: {wer:.4f}")
            raise optuna.exceptions.TrialPruned()

        return wer

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned()


# ===================================================================
#                       MAIN HPO SCRIPT
# ===================================================================

if __name__ == "__main__":
    print(f"Starting Decoder Hyperparameter Optimization")
    print(f"Project: {HPO_PROJECT_NAME}")
    print(f"Decoder script: {RUN_DECODER_SCRIPT}")
    print(f"N_trials: {N_TRIALS}")
    print(f"Results directory: {RESULTS_DIR}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify run_decoder.py exists
    if not RUN_DECODER_SCRIPT.exists():
        raise FileNotFoundError(f"run_decoder.py not found at {RUN_DECODER_SCRIPT}")

    # Initialize wandb
    wandb.init(
        project=HPO_PROJECT_NAME,
        config={
            "n_trials": N_TRIALS,
            "encoder_model": ENCODER_MODEL_NAME,
            "fixed_params": FIXED_PARAMS,
            "search_ranges": HPO_RANGES
        },
        name=f"decoder_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Create Optuna study with QMC sampler
    print(f"Creating Optuna study with QMC sampler and {N_TRIALS} trials...\n")
    sampler = QMCSampler()
    study = optuna.create_study(sampler=sampler, direction='minimize')

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)

    # Save results
    print("\n" + "="*60)
    print("HPO COMPLETE")
    print("="*60)

    best_trial = study.best_trial
    print(f"\nBest Trial: {best_trial.number}")
    print(f"  WER: {best_trial.value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Save best hyperparameters including fixed params
    best_params_with_fixed = {
        "best_trial_number": best_trial.number,
        "wer": best_trial.value,
        "optimized_params": best_trial.params,
        "fixed_params": {k: v for k, v in FIXED_PARAMS.items() if not isinstance(v, bool) or v}
    }

    best_params_file = RESULTS_DIR / "best_decoder_params.json"
    with open(best_params_file, "w") as f:
        json.dump(best_params_with_fixed, f, indent=2)
    print(f"\nSaved best hyperparameters (with fixed params) to {best_params_file}")

    # Save all trials
    all_trials_file = RESULTS_DIR / "all_trials.json"
    all_trials_data = []
    for trial in study.get_trials(states=[optuna.trial.TrialState.COMPLETE]):
        trial_data = {
            "trial_number": trial.number,
            "wer": trial.user_attrs.get("wer", trial.value),
            "optimized_params": trial.params,
            "fixed_params": {k: v for k, v in FIXED_PARAMS.items() if not isinstance(v, bool) or v},
            "return_code": trial.user_attrs.get("return_code", None)
        }
        all_trials_data.append(trial_data)

    # Sort by WER
    all_trials_data.sort(key=lambda x: x["wer"])

    # Save with metadata
    all_trials_output = {
        "hpo_config": {
            "project_name": HPO_PROJECT_NAME,
            "n_trials": N_TRIALS,
            "encoder_model": ENCODER_MODEL_NAME,
            "fixed_params": {k: v for k, v in FIXED_PARAMS.items() if not isinstance(v, bool) or v},
            "search_ranges": HPO_RANGES
        },
        "trials": all_trials_data
    }

    with open(all_trials_file, "w") as f:
        json.dump(all_trials_output, f, indent=2)
    print(f"Saved all trials to {all_trials_file}")

    # Print top 5 trials
    print(f"\nTop 5 Trials by WER:")
    for i, trial_data in enumerate(all_trials_data[:5]):
        print(f"{i+1}. Trial {trial_data['trial_number']}: WER={trial_data['wer']:.4f}")
        print(f"   Optimized params: {trial_data['optimized_params']}")

    # Log summary to wandb
    wandb.log({
        "best_wer": best_trial.value,
        "best_trial_number": best_trial.number
    })

    # Create wandb table with all trials
    trials_table = wandb.Table(
        columns=["trial_number", "wer"] + list(HPO_RANGES.keys())
    )
    for trial_data in all_trials_data:
        row = [
            trial_data["trial_number"],
            trial_data["wer"]
        ] + [trial_data["optimized_params"].get(key, None) for key in HPO_RANGES.keys()]
        trials_table.add_data(*row)

    wandb.log({"all_trials": trials_table})

    # Log best parameters
    wandb.summary["best_trial_number"] = best_trial.number
    wandb.summary["best_wer"] = best_trial.value
    for key, value in best_trial.params.items():
        wandb.summary[f"best_{key}"] = value

    # Finish wandb run
    wandb.finish()

    print("\n" + "="*60)