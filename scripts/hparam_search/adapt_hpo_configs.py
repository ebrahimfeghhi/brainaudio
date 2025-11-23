"""
Adapt existing HPO configs to use different datasets (e.g., B2T 25 only).
Copies configs from source directory and modifies dataset-specific settings.
"""

import yaml
import os
import glob
import shutil
from pathlib import Path


def adapt_configs_for_dataset(
    source_configs_dir,
    output_configs_dir,
    manifest_paths=None,
    device=None,
    config_modifications=None
):
    """
    Adapt HPO configs for a different dataset while keeping hyperparameters the same.
    
    Args:
        source_configs_dir: Directory containing original trial_*_config.yaml files
        output_configs_dir: Directory to save adapted configs
        manifest_paths: List of new manifest paths (e.g., for B2T 25 only)
        device: GPU device to use (e.g., 'cuda:0' or 'cuda:1'). If None, keeps original.
        config_modifications: Dict of additional config changes to make
    """
    
    # Create output directory
    os.makedirs(output_configs_dir, exist_ok=True)
    
    # Find all source configs
    config_files = sorted(glob.glob(os.path.join(source_configs_dir, "trial_*_config.yaml")))
    
    if not config_files:
        print(f"ERROR: No config files found in {source_configs_dir}")
        return
    
    print(f"Found {len(config_files)} configs to adapt")
    
    # Default B2T 25 manifest if not provided
    if manifest_paths is None:
        manifest_paths = ["/data2/brain2text/b2t_25/trial_level_data/manifest.json"]
    
    # Default modifications if not provided
    if config_modifications is None:
        config_modifications = {}
    
    # Adapt each config
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Modify dataset-related settings
        config['manifest_paths'] = manifest_paths
        
        # Apply any additional modifications
        for key, value in config_modifications.items():
            if isinstance(key, str) and '.' in key:
                # Handle nested keys like 'model.transformer.n_heads'
                keys = key.split('.')
                d = config
                for k in keys[:-1]:
                    d = d[k]
                d[keys[-1]] = value
            else:
                config[key] = value
        
        # Update model name to reflect dataset
        original_name = config['modelName']
        config['modelName'] = original_name.replace('combined', 'b2t_25')
        
        # Save adapted con
        # fig
        trial_name = os.path.basename(config_file)
        output_file = os.path.join(output_configs_dir, trial_name)
        
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"  Adapted {trial_name} → {config['modelName']}")
    
    print(f"\nSaved {len(config_files)} adapted configs to {output_configs_dir}")
    return config_files


if __name__ == "__main__":
    # Example: Adapt configs for B2T 25 only
    source_dir = "/data2/brain2text/hpo/hpo_configs/baseline_hpo_combined"
    output_dir = "/data2/brain2text/hpo/hpo_configs/baseline_hpo_b2t_25"
    
    adapt_configs_for_dataset(
        source_configs_dir=source_dir,
        output_configs_dir=output_dir,
        manifest_paths=["/data2/brain2text/b2t_25/trial_level_data/manifest.json"],
        config_modifications={}
    )
