import numpy as np
from scipy import stats


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