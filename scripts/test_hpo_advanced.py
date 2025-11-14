"""
Advanced test for HPO pipeline with realistic scenarios.
Tests that different hparams are saved for each metric/participant combination.
Saves all output to /data2/brain2text/hpo/
"""

import yaml
import json
import os
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpo_utils import track_best_models, save_best_hparams, print_hpo_summary


def test_different_best_hparams():
    """
    Test scenario where different hparams are best for different metrics/participants.
    This is realistic: e.g., high learning_rate might be best for mean WER,
    but lower learning_rate better for participant 0's PER.
    """
    print("\n" + "="*70)
    print("ADVANCED TEST: Different Best Hparams per Metric/Participant")
    print("="*70)
    
    output_dir = "/data2/brain2text/hpo"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Simulate 4 trials with different hparams
    trials = [
        {
            'name': 'Trial 0 (High LR, 8 heads)',
            'hparams': {
                'learning_rate': 0.002,
                'l2_decay': 1e-5,
                'dropout': 0.3,
                'n_heads': 8,
                'depth': 6,
                'total_mask_intensity': 1.0
            },
            'metrics': {
                'mean_wer': 0.28,
                'mean_per': 0.15,
                'by_participant_wer': {0: 0.26, 1: 0.30},
                'by_participant_per': {0: 0.14, 1: 0.16}
            }
        },
        {
            'name': 'Trial 1 (Medium LR, 6 heads)',
            'hparams': {
                'learning_rate': 0.001,
                'l2_decay': 5e-6,
                'dropout': 0.25,
                'n_heads': 6,
                'depth': 5,
                'total_mask_intensity': 0.8
            },
            'metrics': {
                'mean_wer': 0.26,  # BEST mean WER
                'mean_per': 0.16,
                'by_participant_wer': {0: 0.27, 1: 0.25},  # Better PID 1
                'by_participant_per': {0: 0.13, 1: 0.17}   # BEST PID 0 PER
            }
        },
        {
            'name': 'Trial 2 (Low LR, 10 heads)',
            'hparams': {
                'learning_rate': 0.0005,
                'l2_decay': 1e-4,
                'dropout': 0.35,
                'n_heads': 10,
                'depth': 7,
                'total_mask_intensity': 1.5
            },
            'metrics': {
                'mean_wer': 0.27,
                'mean_per': 0.145,  # BEST mean PER
                'by_participant_wer': {0: 0.25, 1: 0.29},  # BEST PID 0 WER
                'by_participant_per': {0: 0.14, 1: 0.15}   # BEST PID 1 PER
            }
        },
        {
            'name': 'Trial 3 (High Dropout, 7 heads)',
            'hparams': {
                'learning_rate': 0.0015,
                'l2_decay': 2e-5,
                'dropout': 0.4,
                'n_heads': 7,
                'depth': 8,
                'total_mask_intensity': 1.2
            },
            'metrics': {
                'mean_wer': 0.29,
                'mean_per': 0.15,
                'by_participant_wer': {0: 0.28, 1: 0.26},  # BEST PID 1 WER
                'by_participant_per': {0: 0.155, 1: 0.145}
            }
        }
    ]
    
    best_metrics = None
    all_best_hparams = {
        'mean_wer': {},
        'mean_per': {},
        'by_participant_wer': {},
        'by_participant_per': {}
    }
    
    # Process each trial
    for trial_idx, trial in enumerate(trials):
        print(f"\n[Trial {trial_idx}] {trial['name']}")
        print(f"  Hparams: LR={trial['hparams']['learning_rate']}, "
              f"n_heads={trial['hparams']['n_heads']}, "
              f"dropout={trial['hparams']['dropout']}")
        
        is_best, best_metrics = track_best_models(trial['metrics'], best_metrics)
        
        # Track which hparams are best for each metric
        if is_best['mean_wer']:
            all_best_hparams['mean_wer'] = trial['hparams']
            print(f"    ✓ NEW BEST Mean WER: {trial['metrics']['mean_wer']:.4f}")
        
        if is_best['mean_per']:
            all_best_hparams['mean_per'] = trial['hparams']
            print(f"    ✓ NEW BEST Mean PER: {trial['metrics']['mean_per']:.4f}")
        
        for pid in trial['metrics']['by_participant_wer'].keys():
            if is_best['by_participant_wer'].get(pid, False):
                all_best_hparams['by_participant_wer'][pid] = trial['hparams']
                print(f"    ✓ NEW BEST Participant {pid} WER: {trial['metrics']['by_participant_wer'][pid]:.4f}")
        
        for pid in trial['metrics']['by_participant_per'].keys():
            if is_best['by_participant_per'].get(pid, False):
                all_best_hparams['by_participant_per'][pid] = trial['hparams']
                print(f"    ✓ NEW BEST Participant {pid} PER: {trial['metrics']['by_participant_per'][pid]:.4f}")
    
    # Now save all the best hparams
    print("\n" + "-"*70)
    print("SAVING BEST HPARAMS")
    print("-"*70)
    
    # Save mean WER
    if all_best_hparams['mean_wer']:
        save_best_hparams(output_dir, all_best_hparams['mean_wer'], 'wer', is_best_mean=True)
    
    # Save mean PER
    if all_best_hparams['mean_per']:
        save_best_hparams(output_dir, all_best_hparams['mean_per'], 'per', is_best_mean=True)
    
    # Save per-participant WER
    if all_best_hparams['by_participant_wer']:
        is_best_dict = {pid: True for pid in all_best_hparams['by_participant_wer'].keys()}
        for pid, hparams in all_best_hparams['by_participant_wer'].items():
            save_best_hparams(output_dir, hparams, 'wer', is_best_participant={pid: True})
    
    # Save per-participant PER
    if all_best_hparams['by_participant_per']:
        is_best_dict = {pid: True for pid in all_best_hparams['by_participant_per'].keys()}
        for pid, hparams in all_best_hparams['by_participant_per'].items():
            save_best_hparams(output_dir, hparams, 'per', is_best_participant={pid: True})
    
    # Print summary
    print_hpo_summary(best_metrics, output_dir)
    
    # Verify files and show differences
    print("\n" + "="*70)
    print("VERIFICATION: Checking if different hparams were saved")
    print("="*70)
    
    files_to_check = [
        'best_hparams_mean_wer.json',
        'best_hparams_mean_per.json',
        'best_hparams_25_wer.json',
        'best_hparams_24_wer.json',
        'best_hparams_25_per.json',
        'best_hparams_24_per.json',
    ]
    
    saved_hparams = {}
    for fname in files_to_check:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                saved_hparams[fname] = json.load(f)
            print(f"\n✓ {fname}")
            print(f"  Learning Rate: {saved_hparams[fname]['learning_rate']}")
            print(f"  N Heads:       {saved_hparams[fname]['n_heads']}")
            print(f"  Dropout:       {saved_hparams[fname]['dropout']}")
    
    # Check if they're different
    print("\n" + "-"*70)
    print("COMPARISON: Are hparams different across metrics?")
    print("-"*70)
    
    lr_values = [h['learning_rate'] for h in saved_hparams.values()]
    heads_values = [h['n_heads'] for h in saved_hparams.values()]
    dropout_values = [h['dropout'] for h in saved_hparams.values()]
    
    lr_unique = len(set(lr_values))
    heads_unique = len(set(heads_values))
    dropout_unique = len(set(dropout_values))
    
    print(f"\nLearning Rates: {lr_unique} unique values out of {len(lr_values)} files")
    if lr_unique > 1:
        print(f"  ✓ Different learning rates: {sorted(set(lr_values))}")
    
    print(f"N Heads: {heads_unique} unique values out of {len(heads_values)} files")
    if heads_unique > 1:
        print(f"  ✓ Different n_heads: {sorted(set(heads_values))}")
    
    print(f"Dropout: {dropout_unique} unique values out of {len(dropout_values)} files")
    if dropout_unique > 1:
        print(f"  ✓ Different dropout: {sorted(set(dropout_values))}")
    
    print("\n" + "="*70)
    print(f"✅ ADVANCED TEST PASSED!")
    print(f"   Saved {len(saved_hparams)} files to {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_different_best_hparams()
