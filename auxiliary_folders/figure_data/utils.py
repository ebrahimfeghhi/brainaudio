import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def get_wer_values(results_folder, str_identifier):
    """Extract WER values from JSON files whose filenames contain str_identifier."""
    wer_values = []
    file_names = []
    for fname in sorted(os.listdir(results_folder)):
        if fname.endswith('.json') and str_identifier in fname:
            fpath = os.path.join(results_folder, fname)
            with open(fpath, 'r') as f:
                data = json.load(f)
            wer_values.append(data['aggregate']['wer'])
            file_names.append(fname)
    return wer_values, file_names


def paired_t_test(values_a, values_b, label_a="A", label_b="B"):
    """Paired t-test between two arrays. Lower is better."""
    values_a = np.array(values_a)
    values_b = np.array(values_b)
    diff = values_a - values_b
    t_stat, p_value = stats.ttest_rel(values_a, values_b)

    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    mean_diff = np.mean(diff)
    sem_a = stats.sem(values_a)
    sem_b = stats.sem(values_b)

    print(f"--- Paired T-Test Results ---")
    print(f"Mean {label_a}: {mean_a:.4f}, SEM {sem_a:.4f}")
    print(f"Mean {label_b}: {mean_b:.4f}, SEM {sem_b:.4f}")
    print(f"Mean difference ({label_a} - {label_b}): {mean_diff:.4f}")
    print(f"T-statistic:    {t_stat:.5f}")
    print(f"P-value:        {p_value:.5f}")

    alpha = 0.05
    if p_value < alpha:
        print("\nConclusion: The difference is statistically significant (Reject H0).")
        if mean_a < mean_b:
            print(f"{label_a} is significantly better than {label_b}.")
        else:
            print(f"{label_b} is significantly better than {label_a}.")
    else:
        print("\nConclusion: The difference is NOT statistically significant (Fail to reject H0).")

    # Bar plot with error bars
    sig_marker = "*" if p_value < alpha else "n.s."
    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar([label_a, label_b], [mean_a, mean_b], yerr=[sem_a, sem_b],
                  capsize=5, color=["#4C72B0", "#DD8452"], edgecolor="black", width=0.5)
    # Significance bracket
    y_max = max(mean_a + sem_a, mean_b + sem_b)
    bracket_y = y_max * 1.05
    ax.plot([0, 0, 1, 1], [bracket_y, bracket_y * 1.02, bracket_y * 1.02, bracket_y],
            color="black", linewidth=1.2)
    ax.text(0.5, bracket_y * 1.03, sig_marker, ha="center", va="bottom", fontsize=13)
    ax.set_ylabel("WER")
    ax.set_ylim(bottom=0, top=bracket_y * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

    return t_stat, p_value


def one_sample_t_test(baseline, comp_values):

    # Perform One-Sample T-Test
    # This checks if the mean of 'ours_data' is significantly different from 9.76
    t_stat, p_value = stats.ttest_1samp(comp_values, baseline)

    # Calculate descriptive statistics for context
    mean_ours = np.mean(comp_values)
    std_ours = np.std(comp_values, ddof=1) # Sample standard deviation

    print(f"--- Results ---")
    print(f"Mean across seeds: {mean_ours:.4f} (vs Baseline: {baseline})")
    print(f"T-statistic:    {t_stat:.5f}")
    print(f"P-value:        {p_value:.5f}")

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("\nConclusion: The difference is statistically significant (Reject H0).")
        if mean_ours > baseline:
            print("Your model is significantly worse than baseline.")
        else:
            print("Your model is significantly better than baseline.")
    else:
        print("\nConclusion: The difference is NOT statistically significant (Fail to reject H0).")