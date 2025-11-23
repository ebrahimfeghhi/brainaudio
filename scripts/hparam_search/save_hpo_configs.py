"""
Script to generate and save HPO configs without training.
Useful for previewing the hyperparameter combinations that will be tested.
"""

import yaml
import optuna
from optuna.samplers import QMCSampler
import os
import random


# ===================================================================
#                       1. DEFINE HPO RANGES
# ===================================================================
HPO_RANGES = {
    # Optimizer params (log scale)
    "learning_rate": ("float", [5e-4, 3e-3], {"log": False}),
    "l2_decay": ("float", [1e-6, 1e-4], {"log": True}),
    
    # Regularization (time masking provides augmentation; input_dropout, white noise, baseline shift removed)
    "dropout": ("float", [0.1, 0.4], {}),
    
    # White Noise
    "whiteNoiseSD": ("float", [0.1, 0.3], {}),
    "constantOffsetSD": ("float", [0.0, 0.1], {}),

    # Transformer Architecture
    "dim_head": ("int", [48,64], {}),  # Model dimension (will derive n_heads from this)
    "n_heads": ("int", [6, 9], {}),
    "depth": ("int", [5, 8], {}),

    # Chunked attention:
    "chunkwise_prob": ("float", [0.4, 0.8], {}),
    
    # Time Masking (core augmentation strategy)
    "total_mask_intensity": ("float", [0.5, 2], {})
}

BASE_CONFIG_PATH = "../src/brainaudio/training/utils/custom_configs/baseline_hpo_combined.yaml"
N_TRIALS = 50
HPO_PROJECT_NAME = "transformer-qmc-search"

# ===================================================================
#                       2. GENERATE CONFIGS
# =======================ß============================================

def generate_configs(n_trials=N_TRIALS):
    """Generate and save HPO configs without training."""
    
    # Load base config
    with open(BASE_CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Extract base config filename (without extension) to create subdirectory
    base_config_name = os.path.splitext(os.path.basename(BASE_CONFIG_PATH))[0]
    
    # Create output directory with config name to avoid overwrites
    config_output_dir = f"/data2/brain2text/hpo/hpo_configs/{base_config_name}"
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Create study to generate hyperparameter combinations
    sampler = QMCSampler()
    study = optuna.create_study(sampler=sampler, direction='minimize')
    
    # Generate trials without actually running them
    # We use a dummy objective that just returns 0
    def dummy_objective(trial):
        hparams = {}
        for name, (ptype, prange, kwargs) in HPO_RANGES.items():
            if ptype == "float":
                hparams[name] = trial.suggest_float(name, prange[0], prange[1], **kwargs)
            elif ptype == "int":
                hparams[name] = trial.suggest_int(name, prange[0], prange[1], **kwargs)
            elif ptype == "categorical":
                hparams[name] = trial.suggest_categorical(name, prange)
        return 0  # Dummy value
    
    # Generate the trials
    study.optimize(dummy_objective, n_trials=n_trials)
    
    # Save configs for each trial
    configs_saved = []
    for trial in study.get_trials():
        trial_number = trial.number
        hparams = trial.params

        # Make a copy of base config
        config = yaml.safe_load(yaml.dump(base_config))

        # --- Extract and process hyperparameters ---
        total_intensity = hparams['total_mask_intensity']
        n_heads = hparams['n_heads']
        dim_head = hparams['dim_head']
        depth = hparams['depth']
        chunkwise_prob = hparams['chunkwise_prob']
        whiteNoiseSD = hparams['whiteNoiseSD']
        constantOffsetSD = hparams['constantOffsetSD']

        # Sample num_masks and derive max_mask_pct
        min_num_masks, max_num_masks = 5, 30
        num_masks = random.randint(min_num_masks, max_num_masks)
        max_mask_pct = total_intensity / num_masks
        max_mask_pct = float(f"{max_mask_pct:.2g}")

        # --- Update config with hparams ---
        config['learning_rate'] = hparams['learning_rate']
        config['l2_decay'] = hparams['l2_decay']
        config['dropout'] = hparams['dropout']
        config['max_mask_pct'] = max_mask_pct
        config['num_masks'] = num_masks
        config['seed'] = config['seeds'][0]
        config['whiteNoiseSD'] = whiteNoiseSD
        config['constantOffsetSD'] = constantOffsetSD


        # Update model args
        model_args = config['model']['transformer']
        model_args['n_heads'] = n_heads
        model_args['dim_head'] = dim_head
        model_args['depth'] = depth
        model_args["chunked_attention"]["chunkwise_prob"] = chunkwise_prob

        # Set trial-specific info
        trial_name = f"trial_{trial_number}"
        config['modelName'] = f"{config['modelName']}_{trial_name}"
        config['wandb']['project'] = HPO_PROJECT_NAME
        config['wandb']['group'] = "QMC_Search"
        config['wandb']['name'] = trial_name

        # Save config
        config_file = os.path.join(config_output_dir, f"{trial_name}_config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        configs_saved.append(config_file)

        # Print summary
        print(f"\nTrial {trial_number}:")
        print(f"  Learning Rate: {hparams['learning_rate']:.2e}")
        print(f"  L2 Decay: {hparams['l2_decay']:.2e}")
        print(f"  Dropout: {hparams['dropout']:.4f}")
        print(f"  n_heads: {n_heads}")
        print(f"  dim_head: {dim_head}")
        print(f"  Depth: {depth}")
        print(f"  Total Mask Intensity: {total_intensity:.4f}")
        print(f"  Max Mask %: {max_mask_pct:.2g}")
        print(f"  Num Masks: {num_masks}")
        print(f"  Config saved to: {config_file}")
    print(f"\n{'='*60}")
    print(f"Saved {len(configs_saved)} configs to {config_output_dir}")
    print(f"{'='*60}\n")
    return configs_saved


if __name__ == "__main__":
    generate_configs(N_TRIALS)
