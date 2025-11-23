"""
Compare trial configs between combined and B2T 25 datasets.
Loads the same trial number from both and displays the differences.
"""

import yaml
import sys
from pathlib import Path


def load_config(config_path):
    """Load a trial config from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compare_configs(trial_num, combined_dir, b2t25_dir):
    """
    Load and compare the same trial config from both datasets.
    
    Args:
        trial_num: Trial number to compare (e.g., 0)
        combined_dir: Path to baseline_hpo directory
        b2t25_dir: Path to baseline_hpo_b2t25 directory
    """
    
    combined_path = Path(combined_dir) / f"trial_{trial_num}_config.yaml"
    b2t25_path = Path(b2t25_dir) / f"trial_{trial_num}_config.yaml"
    
    if not combined_path.exists():
        print(f"ERROR: Combined config not found: {combined_path}")
        return
    
    if not b2t25_path.exists():
        print(f"ERROR: B2T 25 config not found: {b2t25_path}")
        return
    
    # Load configs
    combined_config = load_config(combined_path)
    b2t25_config = load_config(b2t25_path)
    
    print(f"\n{'='*80}")
    print(f"Comparing Trial {trial_num}")
    print(f"{'='*80}\n")
    
    # Display hyperparameters (should be identical)
    print("HYPERPARAMETERS (should be identical):")
    print(f"{'='*80}")
    
    hpo_keys = ['learning_rate', 'l2_decay', 'dropout', 'n_heads', 'dim_head', 'depth', 'total_mask_intensity']
    
    for key in hpo_keys:
        combined_val = combined_config.get(key, "MISSING")
        b2t25_val = b2t25_config.get(key, "MISSING")
        match = "✓" if combined_val == b2t25_val else "✗ MISMATCH"
        print(f"{key:25} | Combined: {str(combined_val):15} | B2T25: {str(b2t25_val):15} | {match}")
    
    # Display dataset-specific settings
    print(f"\n\nDATASET-SPECIFIC SETTINGS (expected to differ):")
    print(f"{'='*80}")
    
    dataset_keys = ['modelName', 'device', 'manifest_paths']
    
    for key in dataset_keys:
        combined_val = combined_config.get(key, "MISSING")
        b2t25_val = b2t25_config.get(key, "MISSING")
        
        if key == 'manifest_paths':
            print(f"\n{key}:")
            print(f"  Combined: {combined_val}")
            print(f"  B2T25:    {b2t25_val}")
        else:
            print(f"{key:25} | Combined: {str(combined_val):30} | B2T25: {str(b2t25_val):30}")
    
    # Check for any other differences
    print(f"\n\nOTHER DIFFERENCES:")
    print(f"{'='*80}")
    
    all_keys = set(combined_config.keys()) | set(b2t25_config.keys())
    found_diffs = False
    
    for key in sorted(all_keys):
        if key not in hpo_keys and key not in dataset_keys:
            combined_val = combined_config.get(key, "MISSING")
            b2t25_val = b2t25_config.get(key, "MISSING")
            
            if combined_val != b2t25_val:
                print(f"\n{key}:")
                print(f"  Combined: {combined_val}")
                print(f"  B2T25:    {b2t25_val}")
                found_diffs = True
    
    if not found_diffs:
        print("No other differences found (good!)")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    
    # Default directories
    combined_dir = "/data2/brain2text/hpo/hpo_configs/baseline_hpo_combined"
    b2t25_dir = "/data2/brain2text/hpo/hpo_configs/baseline_hpo_b2t25"
    
    # Get trial number from command line or use default
    trial_num = 0
    if len(sys.argv) > 1:
        try:
            trial_num = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python compare_trial_configs.py [trial_num]")
            print(f"Example: python compare_trial_configs.py 5")
            sys.exit(1)
    
    # Compare configs
    compare_configs(trial_num, combined_dir, b2t25_dir)
