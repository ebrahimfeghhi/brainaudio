# Save this file as run_optuna_search.py
# (in the same directory as call_trainer.py)

import yaml
import optuna
from optuna.samplers import QMCSampler
from hpo_trainer import run_single_trial
from brainaudio.training.utils import track_best_models, save_best_hparams, print_hpo_summary
import copy
import random
import os
import glob


# ===================================================================
#                       1. CONFIGURATION
# ===================================================================

CONFIGS_DIR = "/data2/brain2text/hpo/hpo_configs/baseline_hpo"  # Pre-saved HPO configs
HPO_PROJECT_NAME = "transformer-qmc-search"
MODEL_NAME = None  # Will be extracted from first config


# ===================================================================
#                       2. DEFINE OPTUNA OBJECTIVE
# ===================================================================

def objective(trial, config_file):
    """
    Run a trial using a pre-saved config file.
    """
    # Load the config from file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract model name (used for results directory)
    global MODEL_NAME
    if MODEL_NAME is None:
        MODEL_NAME = config['modelName'].split('_trial_')[0]
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Config file: {config_file}")
    print(f"Model name: {config['modelName']}")
    
    try:
        # --- Run the trial ---
        final_mean_wer, final_mean_per, best_wer_by_participant, best_per_by_participant = run_single_trial(config)
        
        trial.set_user_attr("final_per", final_mean_per)
        trial.set_user_attr("by_participant_wer", best_wer_by_participant)
        trial.set_user_attr("by_participant_per", best_per_by_participant)
        
        # Check for bad results (e.g., model diverged)
        if final_mean_wer is None or final_mean_wer > 1:
            print(f"Trial {trial.number} pruned due to bad metric: {final_mean_wer}")
            raise optuna.exceptions.TrialPruned()
            
        return final_mean_wer
        
    except Exception as e:
        # Handle crashes (e.g., OOM) gracefully
        print(f"Trial {trial.number} failed with exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Prune the trial so Optuna doesn't count it as a success
        raise optuna.exceptions.TrialPruned()

# ===================================================================
#                       3. RUN HPO WITH PRESAVED CONFIGS
# ===================================================================
if __name__ == "__main__":
    # Find all presaved config files (can be in subdirectories or directly in CONFIGS_DIR)
    config_files = sorted(glob.glob(os.path.join(CONFIGS_DIR, "trial_*_config.yaml")))
    
    # If not found, look in subdirectories
    if not config_files:
        config_files = sorted(glob.glob(os.path.join(CONFIGS_DIR, "*", "trial_*_config.yaml")))
    
    if not config_files:
        print(f"ERROR: No config files found in {CONFIGS_DIR} or {CONFIGS_DIR}/*/")
        exit(1)
    
    n_trials = len(config_files)
    print(f"Found {n_trials} presaved configs in {CONFIGS_DIR}")
    
    # Create study
    sampler = QMCSampler()
    study = optuna.create_study(sampler=sampler, direction='minimize')
    
    # Extract model name from first config
    with open(config_files[0], 'r') as f:
        first_config = yaml.safe_load(f)
        MODEL_NAME = first_config['modelName'].split('_trial_')[0]
    
    # Create results directory
    hpo_output_dir = f"/data2/brain2text/hpo/{MODEL_NAME}"
    os.makedirs(hpo_output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {hpo_output_dir}")
    print(f"Starting Optuna with {n_trials} presaved configs...\n")
    
    # Create wrapper that uses config files
    def objective_with_config(trial):
        config_idx = trial.number
        if config_idx >= len(config_files):
            raise ValueError(f"Trial {config_idx} but only {len(config_files)} configs available")
        return objective(trial, config_files[config_idx])
    
    # Run the study
    study.optimize(objective_with_config, n_trials=n_trials)
    
    # --- Post-process: Track best hparams ---
    print("\n--- HPO Search Complete ---")
    best_metrics = None
    
    for trial in study.get_trials(states=[optuna.trial.TrialState.COMPLETE]):
        mean_wer = trial.value
        mean_per = trial.user_attrs.get('final_per')
        by_participant_wer = trial.user_attrs.get('by_participant_wer', {})
        by_participant_per = trial.user_attrs.get('by_participant_per', {})
        
        current_metrics = {
            'mean_wer': mean_wer,
            'mean_per': mean_per,
            'by_participant_wer': by_participant_wer,
            'by_participant_per': by_participant_per
        }
        
        is_best, best_metrics = track_best_models(current_metrics, best_metrics)
        
        # Save best WER hparams
        if is_best['mean_wer'] or any(is_best['by_participant_wer'].values()):
            save_best_hparams(
                hpo_output_dir,
                trial.params,
                metric_type='wer',
                is_best_mean=is_best['mean_wer'],
                is_best_participant=is_best['by_participant_wer'] if is_best['by_participant_wer'] else None
            )
        
        # Save best PER hparams
        if is_best['mean_per'] or any(is_best['by_participant_per'].values()):
            save_best_hparams(
                hpo_output_dir,
                trial.params,
                metric_type='per',
                is_best_mean=is_best['mean_per'],
                is_best_participant=is_best['by_participant_per'] if is_best['by_participant_per'] else None
            )
    
    # Print summary
    print_hpo_summary(best_metrics, hpo_output_dir)
    
    # === Print Best Trial for Mean WER ===
    best_wer_trial = study.best_trial
    
    if best_wer_trial is not None:
        print("\n--- Best Mean WER Trial ---")
        print(f"Trial Number: {best_wer_trial.number}")
        print(f"  WER: {best_wer_trial.value:.4f}")
        print(f"  PER: {best_wer_trial.user_attrs['final_per']:.4f}")
        print("  Params:")
        for key, value in best_wer_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("Could not find best PER trial (were any trials completed?).")