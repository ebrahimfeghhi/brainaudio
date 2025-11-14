# Save this file as run_optuna_search.py
# (in the same directory as call_trainer.py)

import yaml
import optuna
from optuna.samplers import QMCSampler # Quasi-random sampler
from hpo_trainer import run_single_trial # Our new function
import copy

# ===================================================================
#                       1. DEFINE HPO RANGES
# ===================================================================
# Format: "param_name": (type, [range_low, range_high], {kwargs})
HPO_RANGES = {
    # Optimizer params
    "learning_rate": ("float", [1e-5, 5e-3], {"log": True}),
    "l2_decay": ("float", [1e-7, 1e-3], {"log": True}),
    
    # Regularization
    "dropout": ("float", [0.1, 0.5], {}),
        
    # Transformer Architecture
    "depth": ("int", [5, 8], {}),
    "d_model": ("categorical", [384, 512], {}), 
    
    "max_mask_pct": ("float", [0.02, 0.10], {})
    
}

# Path to your base config file
BASE_CONFIG_PATH = "../src/brainaudio/training/utils/custom_configs/tm_transformer_combined_chunking_reduced_reg_smaller.yaml"
N_TRIALS = 50 # Total number of HPO runs
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

    # --- 2. Mutate the config dict with new hparams ---
    
    # Shared params
    config['learning_rate'] = hparams['learning_rate']
    config['l2_decay'] = hparams['l2_decay']
    config['dropout'] = hparams['dropout']
    config['input_dropout'] = hparams['input_dropout']
    
    # Transformer-specific params
    model_args = config['model']['transformer']
    model_args['depth'] = hparams['depth']
    model_args['d_model'] = hparams['d_model']
    model_args['mlp_dim_ratio'] = hparams['mlp_dim_ratio']
    
    # Chunking params
    model_args['chunked_attention']['chunk_size_min'] = hparams['chunk_size_min']
    model_args['chunked_attention']['chunk_size_max'] = hparams['chunk_size_max']
    model_args['chunked_attention']['context_sec_min'] = hparams['context_sec_min']
    model_args['chunked_attention']['context_sec_max'] = hparams['context_sec_max']

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
        final_wer, final_per = run_single_trial(config)
        
        trial.set_user_attr("final_per", final_per)
        
        # Check for bad results (e.g., model diverged)
        if final_wer is None or final_wer > 0.5:
            print(f"Trial {trial.number} pruned due to bad metric: {final_wer}")
            raise optuna.exceptions.TrialPruned()
            
        return final_wer
        
    except Exception as e:
        # Handle crashes (e.g., OOM) gracefully
        print(f"Trial {trial.number} failed with exception: {e}")
        # Prune the trial so Optuna doesn't count it as a success
        raise optuna.exceptions.TrialPruned()
# ===================================================================
#                       3. RUN THE HPO STUDY
# ===================================================================
if __name__ == "__main__":
    print("Starting Optuna Quasi-Random Search...")

    # We use QMCSampler for quasi-random search (Sobol sequence)
    sampler = QMCSampler(scikit_learn_integers=True)
    
    # We want to MINIMIZE the metric (e.G., WER or loss)
    study = optuna.create_study(
        sampler=sampler, 
        direction='minimize'
    )
    
    # Run the search
    study.optimize(objective, n_trials=N_TRIALS)

    # --- Print final results ---
    print("\n--- HPO Search Complete ---")

    # === 1. Find Best Trial for Metric 1 (final_wer) ===
    # Optuna does this for you automatically
    best_wer_trial = study.best_trial
    
    print("\n--- Best WER Trial ---")
    print(f"Trial Number: {best_wer_trial.number}")
    print(f"  WER: {best_wer_trial.value:.4f}")
    # *** MODIFIED: Print PER ***
    print(f"  PER: {best_wer_trial.user_attrs['final_per']:.4f}")
    print("  Params:")
    for key, value in best_wer_trial.params.items():
        print(f"    {key}: {value}")

    # === 2. Manually Find Best Trial for Metric 2 (final_per) ===
    # *** MODIFIED: All variables changed from 'latency' to 'per' ***
    best_per_trial = None
    best_per = float('inf')

    # We must loop through all trials that finished successfully
    for trial in study.get_trials(states=[optuna.trial.TrialState.COMPLETE]):
        # Get the PER we stored
        per = trial.user_attrs.get('final_per')
        
        if per is not None and per < best_per:
            best_per = per
            best_per_trial = trial

    print("\n--- Best PER Trial ---")
    if best_per_trial:
        print(f"Trial Number: {best_per_trial.number}")
        print(f"  WER: {best_per_trial.value:.4f}") # This is the WER for this trial
        print(f"  PER: {best_per_trial.user_attrs['final_per']:.4f}")
        print("  Params:")
        for key, value in best_per_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("Could not find best PER trial (were any trials completed?).")