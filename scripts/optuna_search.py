# Save this file as run_optuna_search.py
# (in the same directory as call_trainer.py)

import yaml
import optuna
from optuna.samplers import TPESampler # Tree-structured Parzen Estimator sampler
from hpo_trainer import run_single_trial # Our new function
from brainaudio.training.utils import track_best_models, save_best_hparams, print_hpo_summary
import copy
import random
import os


# ===================================================================
#                       1. DEFINE HPO RANGES
# ===================================================================
# Format: "param_name": (type, [list_of_values], {kwargs})
# All parameters are now categorical for more controlled search
# Learning rate and l2_decay use log spacing (geometric progression)
HPO_RANGES = {
    # Optimizer params (log scale categorical)
    "learning_rate": ("categorical", [1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2], {}),
    "l2_decay": ("categorical", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3], {}),
    
    # Regularization (time masking provides augmentation; input_dropout, white noise, baseline shift removed)
    "dropout": ("categorical", [0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.40], {}),
        
    # Transformer Architecture
    "n_heads": ("categorical", [6, 7, 8, 9], {}),
    "depth": ("categorical", [5, 6, 7, 8], {}),
    
    # Time Masking (core augmentation strategy)
    "total_mask_intensity": ("categorical", [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], {})
}


BASE_CONFIG_PATH = "../src/brainaudio/training/utils/custom_configs/baseline_hpo.yaml"
N_TRIALS = 1 # Total number of HPO runs
HPO_PROJECT_NAME = "transformer-qmc-search" # W&B project name for this search

# ===================================================================
#                       2. DEFINE OPTUNA OBJECTIVE
# ===================================================================

def objective(trial):
    """
    This function is called by Optuna for each trial.
    """
    # Load the base config file every time
    with open(BASE_CONFIG_PATH, 'r') as f:
        # Use deepcopy to avoid modifying the base config in memory
        config = yaml.safe_load(f)

    # --- 1. Get new hyperparameters from the trial ---
    hparams = {}
    for name, (ptype, prange, kwargs) in HPO_RANGES.items():
        if ptype == "float":
            hparams[name] = trial.suggest_float(name, prange[0], prange[1], **kwargs)
        elif ptype == "int":
            hparams[name] = trial.suggest_int(name, prange[0], prange[1], **kwargs)
        elif ptype == "categorical":
            hparams[name] = trial.suggest_categorical(name, prange)

    # --- Decompose total_mask_intensity into max_mask_pct and num_masks ---
    # New approach: sample num_masks as an integer, then derive max_mask_pct from total_intensity
    # Constraints: 5 <= num_masks <= 30, 0.02 <= max_mask_pct <= 0.15
    total_intensity = hparams['total_mask_intensity']
    
    min_num_masks, max_num_masks = 5, 30
    
    # Sample num_masks uniformly from the valid range
    num_masks = random.randint(min_num_masks, max_num_masks)
    
    # Derive max_mask_pct from total_intensity: max_mask_pct = total_intensity / num_masks
    max_mask_pct = total_intensity / num_masks
    
    # Round max_mask_pct to 2 significant digits
    max_mask_pct = float(f"{max_mask_pct:.2g}")
    
    hparams['max_mask_pct'] = max_mask_pct
    hparams['num_masks'] = num_masks

    # --- 2. Mutate the config dict with new hparams ---
    
    # Shared params
    config['learning_rate'] = hparams['learning_rate']
    config['l2_decay'] = hparams['l2_decay']
    config['dropout'] = hparams['dropout']
    
    # Transformer-specific params
    if 'model' not in config or 'transformer' not in config.get('model', {}):
        print(f"ERROR: Config structure missing 'model' or 'model.transformer'. Config keys: {config.keys()}")
        raise ValueError("Config missing required model.transformer section")
    
    model_args = config['model']['transformer']
    model_args['n_heads'] = hparams['n_heads']
    model_args['depth'] = hparams['depth']

    # Time masking (augmentation) params
    config['max_mask_pct'] = hparams['max_mask_pct']
    config['num_masks'] = hparams['num_masks']

    # --- 3. Set up trial-specific info ---
    
    # Use only the first seed for HPO. 
    # (You can run the best config on all seeds *after* the search)
    config['seed'] = config['seeds'][0] 
    
    # Set a unique model name
    trial_name = f"trial_{trial.number}"
    config['modelName'] = f"{config['modelName']}_{trial_name}"
    
    # Set W&B info
    config['wandb']['project'] = HPO_PROJECT_NAME
    config['wandb']['group'] = "QMC_Search"
    config['wandb']['name'] = trial_name
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: {trial.params}")
    
    try:
        # --- 4. Run the trial ---
        final_mean_wer, final_mean_per, best_wer_by_participant, best_per_by_participant = run_single_trial(config)
        
        trial.set_user_attr("final_per", final_mean_per)
        trial.set_user_attr("by_participant_wer", best_wer_by_participant)
        trial.set_user_attr("by_participant_per", best_per_by_participant)
        
        # Check for bad results (e.g., model diverged)
        if final_mean_wer is None or final_mean_wer > 0.5:
            print(f"Trial {trial.number} pruned due to bad metric: {final_mean_wer}")
            raise optuna.exceptions.TrialPruned()
            
        return final_mean_wer
        
    except Exception as e:
        # Handle crashes (e.g., OOM) gracefully
        print(f"Trial {trial.number} failed with exception: {e}")
        # Prune the trial so Optuna doesn't count it as a success
        raise optuna.exceptions.TrialPruned()
# ===================================================================
#                       3. RUN THE HPO STUDY
# ===================================================================
if __name__ == "__main__":
    print("Starting Optuna TPE Search...")

    # We use TPESampler (Tree-structured Parzen Estimator) for Bayesian optimization
    sampler = TPESampler()
    
    # We want to MINIMIZE the metric (e.G., WER or loss)
    study = optuna.create_study(
        sampler=sampler, 
        direction='minimize'
    )
    
    # Track best models across all trials
    best_metrics = None
    hpo_output_dir = "./hpo_results"
    os.makedirs(hpo_output_dir, exist_ok=True)
    
    # Run the search
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[
        lambda study, trial: _update_best_hparams(study, trial, best_metrics, hpo_output_dir)
    ] if False else None)  # Callback approach (optional; we'll use post-processing instead)

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
    
    print("\n--- Best Mean WER Trial ---")
    print(f"Trial Number: {best_wer_trial.number}")
    print(f"  WER: {best_wer_trial.value:.4f}")
    print(f"  PER: {best_wer_trial.user_attrs['final_per']:.4f}")
    print("  Params:")
    for key, value in best_wer_trial.params.items():
        print(f"    {key}: {value}")
    else:
        print("Could not find best PER trial (were any trials completed?).")