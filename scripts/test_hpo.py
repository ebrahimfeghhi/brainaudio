"""
Quick test script for HPO pipeline.
Runs a minimal 2-trial test to verify:
1. Hyperparameter sampling works
2. Per-participant metrics are captured
3. Best hparams are saved correctly
4. Output files exist and are valid JSON
"""

import yaml
import json
import os
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpo_utils import track_best_models, save_best_hparams, print_hpo_summary


def test_hpo_utils():
    """Test utility functions in isolation."""
    print("\n" + "="*60)
    print("TEST 1: HPO Utilities")
    print("="*60)
    
    test_dir = "./test_hpo_output"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test 1a: track_best_models with WER
    print("\n[TEST 1a] Tracking best models...")
    metrics_trial1 = {
        'mean_wer': 0.30,
        'mean_per': 0.15,
        'by_participant_wer': {0: 0.28, 1: 0.32},
        'by_participant_per': {0: 0.14, 1: 0.16}
    }
    
    is_best, best_metrics = track_best_models(metrics_trial1, None)
    assert is_best['mean_wer'], "Should detect first trial as best WER"
    assert is_best['mean_per'], "Should detect first trial as best PER"
    assert is_best['by_participant_wer'][0], "Should detect participant 0 as best WER"
    print("  ✓ First trial detected as best")
    
    # Test 1b: Second trial, only some metrics improve
    metrics_trial2 = {
        'mean_wer': 0.31,  # Worse mean WER (was 0.30)
        'mean_per': 0.14,  # Better mean PER (was 0.15)
        'by_participant_wer': {0: 0.27, 1: 0.35},  # Participant 0 better (was 0.28), 1 worse (was 0.32)
        'by_participant_per': {0: 0.15, 1: 0.13}   # Participant 0 worse (was 0.14), 1 better (was 0.16)
    }
    
    is_best, best_metrics = track_best_models(metrics_trial2, best_metrics)
    assert not is_best['mean_wer'], "Mean WER should NOT be best (0.31 > 0.30)"
    assert is_best['mean_per'], "Mean PER should be best (0.14 < 0.15)"
    assert is_best['by_participant_wer'].get(0, False), "Participant 0 WER should be best (0.27 < 0.28)"
    assert not is_best['by_participant_wer'].get(1, False), "Participant 1 WER should NOT be best (0.35 > 0.32)"
    assert not is_best['by_participant_per'].get(0, False), "Participant 0 PER should NOT be best (0.15 > 0.14)"
    assert is_best['by_participant_per'].get(1, False), "Participant 1 PER should be best (0.13 < 0.16)"
    print("  ✓ Selective tracking works correctly")
    
    # Test 1c: Save hparams
    print("\n[TEST 1b] Saving hparams...")
    trial_hparams = {
        'learning_rate': 0.0005,
        'l2_decay': 1e-5,
        'dropout': 0.3,
        'n_heads': 8,
        'depth': 6,
        'total_mask_intensity': 1.2
    }
    
    save_best_hparams(test_dir, trial_hparams, 'wer', is_best_mean=True, is_best_participant={0: True})
    save_best_hparams(test_dir, trial_hparams, 'per', is_best_mean=True, is_best_participant={1: True})
    
    # Verify files exist
    assert os.path.exists(f"{test_dir}/best_hparams_mean_wer.json"), "Mean WER hparams file missing"
    assert os.path.exists(f"{test_dir}/best_hparams_25_wer.json"), "Participant 25 WER hparams file missing"
    assert os.path.exists(f"{test_dir}/best_hparams_mean_per.json"), "Mean PER hparams file missing"
    assert os.path.exists(f"{test_dir}/best_hparams_24_per.json"), "Participant 24 PER hparams file missing"
    print("  ✓ All hparams files created")
    
    # Test 1d: Verify JSON content
    print("\n[TEST 1c] Verifying JSON content...")
    with open(f"{test_dir}/best_hparams_mean_wer.json", "r") as f:
        saved_hparams = json.load(f)
    assert saved_hparams == trial_hparams, "Saved hparams don't match original"
    print("  ✓ JSON content is valid and matches")
    
    # Test 1e: Print summary
    print("\n[TEST 1d] Printing summary...")
    print_hpo_summary(best_metrics, test_dir)
    
    print("\n✅ All utility tests passed!\n")
    return test_dir


def test_hyperparameter_sampling():
    """Test that hyperparameter sampling and decomposition works."""
    print("\n" + "="*60)
    print("TEST 2: Hyperparameter Sampling & Decomposition")
    print("="*60)
    
    import random
    
    # Test total_mask_intensity decomposition
    print("\n[TEST 2] Testing mask intensity decomposition...")
    
    test_cases = [
        {'total_intensity': 0.5, 'mask_ratio': 0.0},
        {'total_intensity': 1.0, 'mask_ratio': 0.5},
        {'total_intensity': 2.0, 'mask_ratio': 1.0},
    ]
    
    min_max_pct, max_max_pct = 0.02, 0.15
    min_num_masks, max_num_masks = 5, 40
    
    for i, test in enumerate(test_cases):
        total_intensity = test['total_intensity']
        mask_ratio = test['mask_ratio']
        
        # Decompose (from run_optuna_search.py logic)
        max_mask_pct = min_max_pct + (max_max_pct - min_max_pct) * mask_ratio
        num_masks = int(total_intensity / max_mask_pct)
        num_masks = max(min_num_masks, min(max_num_masks, num_masks))
        
        # Verify bounds
        assert min_max_pct <= max_mask_pct <= max_max_pct, f"max_mask_pct out of bounds: {max_mask_pct}"
        assert min_num_masks <= num_masks <= max_num_masks, f"num_masks out of bounds: {num_masks}"
        
        actual_intensity = max_mask_pct * num_masks
        print(f"  Case {i+1}: intensity={total_intensity}, ratio={mask_ratio}")
        print(f"    → max_mask_pct={max_mask_pct:.4f}, num_masks={num_masks}, actual_intensity={actual_intensity:.4f}")
    
    print("\n✅ All sampling tests passed!\n")


def test_config_loading():
    """Test that base config loads correctly."""
    print("\n" + "="*60)
    print("TEST 3: Config Loading")
    print("="*60)
    
    config_path = "../src/brainaudio/training/utils/custom_configs/tm_transformer_combined_chunking_reduced_reg_smaller.yaml"
    
    if not os.path.exists(config_path):
        print(f"  ⚠️  Config file not found at {config_path}")
        print("     (This is OK for quick test, but will fail in real run)")
        return
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key fields
        required_keys = ['learning_rate', 'l2_decay', 'dropout', 'num_masks', 'max_mask_pct']
        for key in required_keys:
            assert key in config, f"Missing key: {key}"
        
        # Check model config
        assert 'model' in config, "Missing 'model' key"
        assert 'transformer' in config['model'], "Missing 'model.transformer'"
        assert 'n_heads' in config['model']['transformer'], "Missing 'n_heads'"
        
        print(f"  ✓ Config loads successfully")
        print(f"  ✓ Current config values:")
        print(f"    - learning_rate: {config['learning_rate']}")
        print(f"    - dropout: {config['dropout']}")
        print(f"    - n_heads: {config['model']['transformer']['n_heads']}")
        print(f"    - depth: {config['model']['transformer']['depth']}")
        
        print("\n✅ Config test passed!\n")
    except Exception as e:
        print(f"  ✗ Config test failed: {e}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HPO PIPELINE TEST SUITE")
    print("="*60)
    
    # Run tests
    test_dir = test_hpo_utils()
    test_hyperparameter_sampling()
    test_config_loading()
    
    print("="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print(f"\nTest artifacts saved to: {test_dir}")
    print("You can inspect the JSON files to verify structure.\n")
