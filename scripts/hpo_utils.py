"""
Utility functions for Optuna HPO experiments.
"""

import json
import os
from pathlib import Path


def save_best_hparams(output_dir, trial_params, metric_type, is_best_mean=False, is_best_participant=None):
    """
    Save hyperparameters for best models.
    
    Args:
        output_dir: Directory to save results
        trial_params: Dict of hyperparameters for this trial
        metric_type: Either 'wer' or 'per'
        is_best_mean: If True, save as best mean model hparams
        is_best_participant: If dict {pid: is_best}, save for those participants
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best mean hparams
    if is_best_mean:
        mean_file = os.path.join(output_dir, f"best_hparams_mean_{metric_type}.json")
        with open(mean_file, "w") as f:
            json.dump(trial_params, f, indent=2)
        print(f"Saved best mean {metric_type.upper()} hparams to {mean_file}")
    
    # Save best per-participant hparams
    if is_best_participant:
        suffix_map = {0: "_25", 1: "_24"}
        for pid, is_best in is_best_participant.items():
            if is_best:
                suffix = suffix_map.get(pid, f"_{pid}")
                pid_file = os.path.join(output_dir, f"best_hparams{suffix}_{metric_type}.json")
                with open(pid_file, "w") as f:
                    json.dump(trial_params, f, indent=2)
                print(f"Saved best {metric_type.upper()} hparams for participant {pid}{suffix} to {pid_file}")


def track_best_models(current_metrics, best_metrics):
    """
    Track best models across trials for both WER and PER.
    
    Args:
        current_metrics: Dict with keys 'mean_wer', 'mean_per', 'by_participant_wer', 'by_participant_per'
        best_metrics: Dict tracking best values so far
    
    Returns:
        Dict with is_best indicators for each metric type
    """
    if not best_metrics:
        best_metrics = {
            'mean_wer': float('inf'),
            'mean_per': float('inf'),
            'by_participant_wer': {},
            'by_participant_per': {}
        }
    
    is_best = {
        'mean_wer': False,
        'mean_per': False,
        'by_participant_wer': {},
        'by_participant_per': {}
    }
    
    # Check if best mean WER
    if current_metrics['mean_wer'] < best_metrics['mean_wer']:
        best_metrics['mean_wer'] = current_metrics['mean_wer']
        is_best['mean_wer'] = True
    
    # Check if best mean PER
    if current_metrics['mean_per'] < best_metrics['mean_per']:
        best_metrics['mean_per'] = current_metrics['mean_per']
        is_best['mean_per'] = True
    
    # Check if best by participant WER
    for pid, wer in current_metrics['by_participant_wer'].items():
        if pid not in best_metrics['by_participant_wer'] or wer < best_metrics['by_participant_wer'][pid]:
            best_metrics['by_participant_wer'][pid] = wer
            is_best['by_participant_wer'][pid] = True
    
    # Check if best by participant PER
    for pid, per in current_metrics['by_participant_per'].items():
        if pid not in best_metrics['by_participant_per'] or per < best_metrics['by_participant_per'][pid]:
            best_metrics['by_participant_per'][pid] = per
            is_best['by_participant_per'][pid] = True
    
    return is_best, best_metrics


def load_best_hparams(hparams_file):
    """Load saved hyperparameters from file."""
    with open(hparams_file, "r") as f:
        return json.load(f)


def print_hpo_summary(best_metrics, output_dir):
    """Print summary of HPO results."""
    print("\n" + "="*60)
    print("HPO SUMMARY")
    print("="*60)
    
    print("\n--- WER Metrics ---")
    print(f"Best Mean WER: {best_metrics['mean_wer']:.4f}")
    print(f"Best By Participant WER: {best_metrics['by_participant_wer']}")
    
    suffix_map = {0: "_25", 1: "_24"}
    for pid, wer in best_metrics['by_participant_wer'].items():
        suffix = suffix_map.get(pid, f"_{pid}")
        hparams_file = os.path.join(output_dir, f"best_hparams{suffix}_wer.json")
        if os.path.exists(hparams_file):
            print(f"  Participant {pid}{suffix}: WER={wer:.4f}, Saved to {hparams_file}")
    
    mean_file = os.path.join(output_dir, "best_hparams_mean_wer.json")
    if os.path.exists(mean_file):
        print(f"Best Mean WER Hparams: {mean_file}")
    
    print("\n--- PER Metrics ---")
    print(f"Best Mean PER: {best_metrics['mean_per']:.4f}")
    print(f"Best By Participant PER: {best_metrics['by_participant_per']}")
    
    for pid, per in best_metrics['by_participant_per'].items():
        suffix = suffix_map.get(pid, f"_{pid}")
        hparams_file = os.path.join(output_dir, f"best_hparams{suffix}_per.json")
        if os.path.exists(hparams_file):
            print(f"  Participant {pid}{suffix}: PER={per:.4f}, Saved to {hparams_file}")
    
    mean_file = os.path.join(output_dir, "best_hparams_mean_per.json")
    if os.path.exists(mean_file):
        print(f"Best Mean PER Hparams: {mean_file}")
    
    print("="*60 + "\n")
