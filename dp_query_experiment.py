"""
DP Query-Level Experiment: Laplace Mechanism on Counts and Means
==================================================================
Complements dp_experiment_template.py (which trains whole models under
DP) by testing the *original*, simplest form of differential privacy:
answering a single statistical query (a count or a mean) with noise
added directly to the true answer.

This uses the classic Laplace mechanism, implemented directly here
(not via diffprivlib) so the mechanism is fully transparent and easy
to explain in your report:

    noisy_answer = true_answer + Laplace(0, sensitivity / epsilon)

- For a COUNT query: sensitivity = 1
  (adding or removing one person changes a count by at most 1)
- For a MEAN query: sensitivity = (upper_bound - lower_bound) / n
  (one person's value can shift the mean by at most this much,
   given known bounds on the value and a fixed dataset size n)

Run this AFTER prepare_data.py has produced the four standardized
CSVs (german_credit.csv, mammographic_mass.csv, student_performance.csv,
bank_marketing.csv) in the same folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPSILONS = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
N_REPEATS = 500  # cheap query, so we can afford many repeats for smooth curves

# ------------------------------------------------------------------
# One query definition per dataset: a COUNT of the minority class,
# and a MEAN of a meaningful numeric column *within* the minority
# class (the group most affected by small subgroup size).
# ------------------------------------------------------------------
QUERIES = {
    "german_credit": {
        "csv_path": "german_credit.csv",
        "target_col": "target",
        "minority_label": "bad",
        "mean_col": "credit_amount",
        "mean_bounds": (0, 20000),  # plausible min/max for this column
    },
    "mammographic_mass": {
        "csv_path": "mammographic_mass.csv",
        "target_col": "target",
        "minority_label": "malignant",
        "mean_col": "age",
        "mean_bounds": (18, 96),
    },
    "student_performance": {
        "csv_path": "student_performance.csv",
        "target_col": "target",
        "minority_label": "fail",
        "mean_col": "absences",
        "mean_bounds": (0, 93),
    },
    "bank_marketing": {
        "csv_path": "bank_marketing.csv",
        "target_col": "target",
        "minority_label": "yes",
        "mean_col": "balance",
        "mean_bounds": (-8000, 100000),
    },
}


def laplace_noise(scale, size):
    """Draw noise from a Laplace(0, scale) distribution."""
    return np.random.laplace(loc=0.0, scale=scale, size=size)


def noisy_count_trials(true_count, epsilon, n_repeats):
    """Sensitivity of a count query is 1."""
    scale = 1.0 / epsilon
    noise = laplace_noise(scale, n_repeats)
    return true_count + noise


def noisy_mean_trials(true_mean, c, epsilon, bounds, n_repeats):
    """Sensitivity of a mean query is (upper - lower) / c, where c is the
    size of the group the mean is computed over."""
    lower, upper = bounds
    sensitivity = (upper - lower) / c

def run_query_experiment():
    all_results = []

    for dataset_name, q in QUERIES.items():
        df = pd.read_csv(q["csv_path"])
        minority = df[df[q["target_col"]] == q["minority_label"]]

        true_count = len(minority)
        true_mean = minority[q["mean_col"]].mean()
        n = len(minority)

        print(f"{dataset_name}: minority group n={true_count}, "
              f"true mean {q['mean_col']}={true_mean:.1f}")

        for eps in EPSILONS:
            counts = noisy_count_trials(true_count, eps, N_REPEATS)
            means = noisy_mean_trials(true_mean, n, eps, q["mean_bounds"], N_REPEATS)

            count_rel_err = np.abs(counts - true_count) / true_count * 100
            mean_rel_err = np.abs(means - true_mean) / abs(true_mean) * 100

            all_results.append({
                "dataset": dataset_name,
                "epsilon": eps,
                "true_count": true_count,
                "count_rel_err_mean": count_rel_err.mean(),
                "count_rel_err_std": count_rel_err.std(),
                "mean_rel_err_mean": mean_rel_err.mean(),
                "mean_rel_err_std": mean_rel_err.std(),
            })

    results = pd.DataFrame(all_results)
    results.to_csv("results_query_experiment.csv", index=False)
    print("Saved results_query_experiment.csv")
    return results


def plot_query_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for dataset_name in QUERIES:
        subset = results[results["dataset"] == dataset_name]
        true_n = subset["true_count"].iloc[0]
        label = f"{dataset_name} (minority n={true_n})"

        axes[0].errorbar(subset["epsilon"], subset["count_rel_err_mean"],
                         yerr=subset["count_rel_err_std"], marker="o",
                         label=label, capsize=3)
        axes[1].errorbar(subset["epsilon"], subset["mean_rel_err_mean"],
                         yerr=subset["mean_rel_err_std"], marker="o",
                         label=label, capsize=3)

    for ax, title in zip(axes, ["Count query: minority class size",
                                 "Mean query: within minority class"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("epsilon (privacy budget)")
        ax.set_ylabel("relative error (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.axhline(10, color="gray", linestyle=":", linewidth=1)  # 10% error reference line

    plt.tight_layout()
    plt.savefig("results_query_experiment.png", dpi=150)
    print("Saved results_query_experiment.png")


if __name__ == "__main__":
    results = run_query_experiment()
    plot_query_results(results)