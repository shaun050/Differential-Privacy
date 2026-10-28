import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression as SKLogReg
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score)
from imblearn.over_sampling import SMOTE
import diffprivlib.models as dp_models


CONFIG = {
    "dataset_name": "german_credit",       # used in output filenames
    "csv_path": "german_credit.csv",        # path to downloaded CSV
    "target_col": "target",                 # column with class label
    "positive_label": "bad",                # the minority/rare class
    "numeric_cols": ["duration", "credit_amount", "age"],
    "categorical_cols": ["purpose", "housing", "job"],
    "test_size": 0.2,
    "random_state": 42,
    "n_repeats": 50,   
    "epsilons": [0.1, 0.5, 1, 2, 5, 10, float("inf")],  # inf = non-private baseline
}



CONFIG_MAMMOGRAPHIC_MASS = {
    "dataset_name": "mammographic_mass",
    "csv_path": "mammographic_mass.csv",
    "target_col": "target",
    "positive_label": "malignant",
    "numeric_cols": ["age", "birads"],
    "categorical_cols": ["shape", "margin", "density"],
    "test_size": 0.2,
    "random_state": 42,
    "n_repeats": 50,   
    "epsilons": [0.1, 0.5, 1, 2, 5, 10, float("inf")],
}

CONFIG_STUDENT_PERFORMANCE = {
    "dataset_name": "student_performance",
    "csv_path": "student_performance.csv",
    "target_col": "target",
    "positive_label": "fail",   # minority class per the 1.8:1 pass:fail ratio
    "numeric_cols": ["age", "studytime", "failures", "absences", "goout", "health"],
    "categorical_cols": ["school", "sex", "address", "famsize", "internet"],
    "test_size": 0.2,
    "random_state": 42,
    "n_repeats": 50,   
    "epsilons": [0.1, 0.5, 1, 2, 5, 10, float("inf")],
}


CONFIG_ADULT_INCOME = {
    "dataset_name": "adult_income",
    "csv_path": "adult_income.csv",
    "target_col": "target",
    "positive_label": "high",   # minority class (~24% earn >50K)
    "numeric_cols": ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"],
    "categorical_cols": ["workclass", "marital_status", "occupation", "sex"],
    "test_size": 0.2,
    "random_state": 42,
    "n_repeats": 50,   
    "epsilons": [0.1, 0.5, 1, 2, 5, 10, float("inf")],
}


def load_and_preprocess(cfg):
    """Load CSV, encode categoricals, scale numerics, return X, y, and
    the data_norm bound required by diffprivlib mechanisms."""
    df = pd.read_csv(cfg["csv_path"])
    y = (df[cfg["target_col"]] == cfg["positive_label"]).astype(int).values
    num = df[cfg["numeric_cols"]].copy()
    cat = df[cfg["categorical_cols"]].copy()   
    enc = OrdinalEncoder()
    cat_encoded = enc.fit_transform(cat)
    X = np.hstack([num.values, cat_encoded])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)   
    max_row_norm = np.linalg.norm(X_scaled, axis=1).max()
    if max_row_norm > 0:
        X_scaled = X_scaled / max_row_norm
    data_norm = 1.0
    return X_scaled, y, data_norm


def run_single_trial(X_train, y_train, X_test, y_test, epsilon, data_norm,
                      use_smote=False, random_state=0):
    """Train one model (private or non-private) and return utility metrics."""
    if use_smote:
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    if epsilon == float("inf"):
        # Non-private baseline
        clf = SKLogReg(max_iter=1000)
        clf.fit(X_train, y_train)
    else:
        clf = dp_models.LogisticRegression(
            epsilon=epsilon,
            data_norm=data_norm,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        "recall_minority": recall_score(y_test, preds, pos_label=1, zero_division=0),
        "recall_majority": recall_score(y_test, preds, pos_label=0, zero_division=0),       
        "precision_minority": precision_score(y_test, preds, pos_label=1,
                                              zero_division=0),
        "f1_minority": f1_score(y_test, preds, pos_label=1, zero_division=0),
        "pred_pos_rate": float(preds.mean()),
        "base_rate": float(y_test.mean()),
    }


def run_experiment(cfg):
    X, y, data_norm = load_and_preprocess(cfg)
    records = []
    for eps in cfg["epsilons"]:
        for use_smote in [False, True]:
            for rep in range(cfg["n_repeats"]):
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=cfg["test_size"],
                    random_state=rep, stratify=y,
                )                
                trial_seed = rep + 1000 * (cfg["epsilons"].index(eps) + 1)
                metrics = run_single_trial(
                    X_train, y_train, X_test, y_test, eps, data_norm,
                    use_smote=use_smote, random_state=trial_seed,
                )
                metrics.update({
                    "epsilon": eps,
                    "smote": use_smote,
                    "repeat": rep,
                })
                records.append(metrics)
        print(f"epsilon={eps} done")
    results = pd.DataFrame(records)
    out_csv = f"results_{cfg['dataset_name']}.csv"
    results.to_csv(out_csv, index=False)
    print(f"Saved raw results to {out_csv}")
    return results


def plot_results(results, cfg):   
    baseline_no_smote = results[(results["epsilon"] == float("inf")) &
                                (results["smote"] == False)]
    baseline_smote = results[(results["epsilon"] == float("inf")) &
                             (results["smote"] == True)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(
        axes, ["f1_macro", "recall_minority"],
        ["F1 (macro)", "Recall on minority class"]
    ):
        for smote_flag, label in [(False, "DP, no SMOTE"), (True, "DP, with SMOTE")]:
            subset = results[(results["smote"] == smote_flag) & (results["epsilon"] != float("inf"))]
            grouped = subset.groupby("epsilon")[metric].agg(["mean", "std"])
            ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                        marker="o", label=label, capsize=3)
        if not baseline_no_smote.empty:
            ax.axhline(baseline_no_smote[metric].mean(), color="gray",
                       linestyle="--", linewidth=1.5,
                       label="non-private, no SMOTE")
        if not baseline_smote.empty:
            ax.axhline(baseline_smote[metric].mean(), color="darkgreen",
                       linestyle="-.", linewidth=1.5,
                       label="non-private, with SMOTE")
        ax.set_xscale("log")
        ax.set_xlabel("epsilon (privacy budget)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs epsilon -- {cfg['dataset_name']}")
        ax.legend()
    plt.tight_layout()
    out_png = f"results_{cfg['dataset_name']}.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")


CONFIG_BANK_MARKETING = {
    "dataset_name": "bank_marketing",
    "csv_path": "bank_marketing.csv",
    "target_col": "target",
    "positive_label": "yes",   # minority class -- only ~12% of rows
    "numeric_cols": ["age", "balance", "day", "duration", "campaign", "pdays", "previous"],
    "categorical_cols": ["job", "marital", "education", "housing", "loan", "contact", "month"],
    "test_size": 0.2,
    "random_state": 42,
    "n_repeats": 50,   
    "epsilons": [0.1, 0.5, 1, 2, 5, 10, float("inf")],
}


ALL_CONFIGS = [
    CONFIG,                        # german_credit
    CONFIG_MAMMOGRAPHIC_MASS,
    CONFIG_STUDENT_PERFORMANCE,
    CONFIG_BANK_MARKETING,
]


def print_diagnostic_summary(results, cfg):    
    print(f"\n--- Diagnostic: {cfg['dataset_name']} ---")
    d = results[results["epsilon"] != float("inf")]
    summary = d.groupby(["epsilon", "smote"])[
        ["recall_minority", "precision_minority", "f1_minority",
         "pred_pos_rate", "base_rate"]
    ].mean().round(3)
    print(summary.to_string())


if __name__ == "__main__":
    import sys    
    if len(sys.argv) > 1:
        wanted = sys.argv[1]
        configs = [c for c in ALL_CONFIGS if c["dataset_name"] == wanted]
        if not configs:
            names = ", ".join(c["dataset_name"] for c in ALL_CONFIGS)
            raise SystemExit(f"Unknown dataset '{wanted}'. Options: {names}")
    else:
        configs = ALL_CONFIGS
    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"  {cfg['dataset_name']}")
        print(f"{'=' * 60}")
        results = run_experiment(cfg)
        plot_results(results, cfg)
        print_diagnostic_summary(results, cfg)