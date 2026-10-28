import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression as SKLogReg
from sklearn.metrics import f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
import diffprivlib.models as dp_models


CONFIG = {
    "dataset_name": "bank_marketing_full",
    "csv_path": "bank_marketing_full.csv",
    "target_col": "target",
    "positive_label": "yes",
    "numeric_cols": ["age", "balance", "day", "duration", "campaign", "pdays", "previous"],
    "categorical_cols": ["job", "marital", "education", "housing", "loan", "contact", "month"],    
    "sample_sizes": [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 45211],
    "test_size": 0.2,
    "epsilons": [0.1, 0.5, 1, 2, 5, 10, float("inf")],    
    "repeats_by_size": {
        0: 30,        # n < 1000
        1000: 20,     # 1000 <= n < 10000
        10000: 10,    # n >= 10000
    },
    # Epsilon held fixed for the size-sweep summary plots.
    "focus_epsilon": 1.0,
    "query_epsilons": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
    "query_repeats": 1000,
}


def repeats_for_size(n, cfg):
    """Look up how many repeats to run at a given subsample size."""
    applicable = [k for k in cfg["repeats_by_size"] if n >= k]
    return cfg["repeats_by_size"][max(applicable)]


def stratified_subsample(df, target_col, n_target, random_state):    
    if n_target >= len(df):
        return df.copy()
    # train_test_split with stratify gives us a stratified draw
    subsample, _ = train_test_split(
        df,
        train_size=n_target,
        stratify=df[target_col],
        random_state=random_state,
    )
    return subsample.reset_index(drop=True)


def encode(df, cfg):
    """Encode a dataframe into X, y, and the norm bound for diffprivlib."""
    y = (df[cfg["target_col"]] == cfg["positive_label"]).astype(int).values
    num = df[cfg["numeric_cols"]].values
    cat = OrdinalEncoder().fit_transform(df[cfg["categorical_cols"]])
    X = np.hstack([num, cat])
    X = StandardScaler().fit_transform(X)
    data_norm = np.linalg.norm(X.max(axis=0) - X.min(axis=0))
    return X, y, data_norm



def run_count_query_sweep(df_full, cfg):    
    records = []
    for n_target in cfg["sample_sizes"]:
        sub = stratified_subsample(df_full, cfg["target_col"], n_target,
                                   random_state=0)
        true_count = int((sub[cfg["target_col"]] == cfg["positive_label"]).sum())
        for eps in cfg["query_epsilons"]:
            noise = np.random.laplace(0.0, 1.0 / eps, cfg["query_repeats"])
            noisy = true_count + noise
            rel_err = np.abs(noisy - true_count) / true_count * 100
            records.append({
                "n_total": len(sub),
                "true_count": true_count,
                "epsilon": eps,
                "rel_err_mean": rel_err.mean(),
                "rel_err_std": rel_err.std(),
            })
        print(f"  count query: n={len(sub)}, minority={true_count} done")
    return pd.DataFrame(records)



def run_classification_sweep(df_full, cfg):
    """For each subsample size and epsilon, train DP logistic regression
    (with and without SMOTE) and record utility."""
    records = []
    for n_target in cfg["sample_sizes"]:
        n_reps = repeats_for_size(n_target, cfg)
        t0 = time.time()
        for rep in range(n_reps):
            # Resample the subset per repeat so results reflect variation
            # in WHICH records were sampled, not just in the DP noise.
            sub = stratified_subsample(df_full, cfg["target_col"],
                                       n_target, random_state=rep)
            X, y, data_norm = encode(sub, cfg)
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=cfg["test_size"],
                    random_state=rep, stratify=y)
            except ValueError:
                # too few minority samples to stratify at this size
                continue
            for eps in cfg["epsilons"]:
                for use_smote in [False, True]:
                    Xt, yt = X_tr, y_tr
                    if use_smote:
                        minority_n = int(yt.sum())
                        if minority_n < 6:
                            # SMOTE needs enough neighbours; skip and
                            # record as NaN rather than crashing
                            continue
                        k = min(5, minority_n - 1)
                        try:
                            Xt, yt = SMOTE(random_state=rep,
                                           k_neighbors=k).fit_resample(X_tr, y_tr)
                        except ValueError:
                            continue
                    if eps == float("inf"):
                        clf = SKLogReg(max_iter=1000)
                    else:
                        clf = dp_models.LogisticRegression(
                            epsilon=eps, data_norm=data_norm,
                            random_state=rep)
                    clf.fit(Xt, yt)
                    pred = clf.predict(X_te)
                    records.append({
                        "n_total": len(sub),
                        "epsilon": eps,
                        "smote": use_smote,
                        "repeat": rep,
                        "f1_macro": f1_score(y_te, pred, average="macro",
                                             zero_division=0),
                        "recall_minority": recall_score(y_te, pred, pos_label=1,
                                                        zero_division=0),                       
                        "recall_majority": recall_score(y_te, pred, pos_label=0,
                                                        zero_division=0),
                        "precision_minority": precision_score(
                            y_te, pred, pos_label=1, zero_division=0),
                        "f1_minority": f1_score(y_te, pred, pos_label=1,
                                                zero_division=0),                      
                        "pred_pos_rate": float(pred.mean()),
                        "base_rate": float(y_te.mean()),
                    })
        print(f"  classification: n={n_target} done "
              f"({n_reps} repeats, {time.time() - t0:.0f}s)")
    return pd.DataFrame(records)



def plot_count_sweep(results, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Left: relative error vs epsilon, one line per dataset size
    for n_total in sorted(results["n_total"].unique()):
        sub = results[results["n_total"] == n_total]
        minority = sub["true_count"].iloc[0]
        axes[0].plot(sub["epsilon"], sub["rel_err_mean"], marker="o",
                     label=f"n={n_total} (minority={minority})")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].axhline(10, color="gray", linestyle=":", linewidth=1)
    axes[0].set_xlabel("epsilon (privacy budget)")
    axes[0].set_ylabel("relative error (%)")
    axes[0].set_title("Count query error vs epsilon, by dataset size")
    axes[0].legend(fontsize=8)
    # Right: relative error vs dataset size, one line per epsilon
    for eps in [0.1, 0.5, 1, 5]:
        sub = results[results["epsilon"] == eps].sort_values("n_total")
        axes[1].plot(sub["n_total"], sub["rel_err_mean"], marker="s",
                     label=f"epsilon={eps}")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].axhline(10, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xlabel("dataset size (n, log scale)")
    axes[1].set_ylabel("relative error (%)")
    axes[1].set_title("Count query error vs dataset size, by epsilon")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"results_subsampling_count_{cfg['dataset_name']}.png", dpi=150)
    print("Saved count-query subsampling plot")


def plot_classification_sweep(results, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    eps_focus = cfg["focus_epsilon"]
    # Left: utility vs dataset size at a FIXED epsilon
    for smote_flag, label in [(False, f"DP (eps={eps_focus}), no SMOTE"),
                              (True, f"DP (eps={eps_focus}), with SMOTE")]:
        sub = results[(results["epsilon"] == eps_focus) &
                      (results["smote"] == smote_flag)]
        g = sub.groupby("n_total")["recall_minority"].agg(["mean", "std"])
        axes[0].errorbar(g.index, g["mean"], yerr=g["std"], marker="o",
                         capsize=3, label=label)

   
    baseline_styles = [
        (False, "--", "dimgray", "non-private, no SMOTE"),
        (True, "-.", "darkgreen", "non-private, with SMOTE"),
    ]
    for smote_flag, ls, colour, label in baseline_styles:
        base = results[(results["epsilon"] == float("inf")) &
                       (results["smote"] == smote_flag)]
        if not base.empty:
            gb = base.groupby("n_total")["recall_minority"].mean()
            axes[0].plot(gb.index, gb.values, color=colour, linestyle=ls,
                         linewidth=1.5, label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("dataset size (n, log scale)")
    axes[0].set_ylabel("recall on minority class")
    axes[0].set_title(f"Minority recall vs dataset size (epsilon={eps_focus})")
    axes[0].legend(fontsize=8)

    # Right: run-to-run variability vs dataset size
    for smote_flag, label in [(False, "no SMOTE"), (True, "with SMOTE")]:
        sub = results[(results["epsilon"] == eps_focus) &
                      (results["smote"] == smote_flag)]
        g = sub.groupby("n_total")["recall_minority"].std()
        axes[1].plot(g.index, g.values, marker="s", label=label)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("dataset size (n, log scale)")
    axes[1].set_ylabel("std. dev. of minority recall across runs")
    axes[1].set_title(f"Run-to-run instability vs dataset size (epsilon={eps_focus})")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"results_subsampling_clf_{cfg['dataset_name']}.png", dpi=150)
    print("Saved classification subsampling plot")


def plot_diagnostics(results, cfg):
   
    eps_focus = cfg["focus_epsilon"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    base_rate = results["base_rate"].mean()
    panels = [
        ("precision_minority", "precision on minority class",
         "Precision vs base rate"),
        ("pred_pos_rate", "proportion predicted as minority",
         "Prediction rate vs true prevalence"),
        ("f1_minority", "F1 on minority class",
         "Minority F1 (precision and recall combined)"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, panels):
        for smote_flag, label in [(False, "DP, no SMOTE"), (True, "DP, with SMOTE")]:
            sub = results[(results["epsilon"] == eps_focus) &
                          (results["smote"] == smote_flag)]
            g = sub.groupby("n_total")[metric].mean()
            ax.plot(g.index, g.values, marker="o", label=label)
        if metric in ("precision_minority", "pred_pos_rate"):
            ax.axhline(base_rate, color="crimson", linestyle=":", linewidth=1.5,
                       label=f"true base rate ({base_rate:.3f})")
        ax.set_xscale("log")
        ax.set_xlabel("dataset size (n, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (epsilon={eps_focus})")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"results_subsampling_diagnostics_{cfg['dataset_name']}.png", dpi=150)
    print("Saved diagnostics plot")


if __name__ == "__main__":
    cfg = CONFIG
    df_full = pd.read_csv(cfg["csv_path"])
    print(f"Loaded {cfg['csv_path']}: {len(df_full)} rows")
    ratio = (df_full[cfg["target_col"]] == cfg["positive_label"]).mean()
    print(f"Minority proportion held fixed at {ratio:.1%} across all subsamples\n")
    print("Running count-query size sweep...")
    count_results = run_count_query_sweep(df_full, cfg)
    count_results.to_csv(f"results_subsampling_count_{cfg['dataset_name']}.csv",
                         index=False)
    print("\nRunning classification size sweep (this takes a few minutes)...")
    clf_results = run_classification_sweep(df_full, cfg)
    clf_results.to_csv(f"results_subsampling_clf_{cfg['dataset_name']}.csv",
                       index=False)
    plot_count_sweep(count_results, cfg)
    plot_classification_sweep(clf_results, cfg)
    plot_diagnostics(clf_results, cfg)

    # Console summary for the thesis text
    print("\n--- Summary at epsilon =", cfg["focus_epsilon"], "---")
    focus = clf_results[clf_results["epsilon"] == cfg["focus_epsilon"]]
    summary = focus.groupby(["n_total", "smote"])["recall_minority"].agg(
        ["mean", "std"]).round(3)
    print(summary)

   
    print("\n--- Decomposition of effects (minority recall) ---")
    print("privacy_cost   = non-private minus DP, at matching SMOTE setting")
    print("smote_gain_dp  = with-SMOTE minus no-SMOTE, both under DP\n")

    rows = []
    for n_total in sorted(clf_results["n_total"].unique()):
        d = clf_results[clf_results["n_total"] == n_total]
        def mean_of(eps, smote):
            s = d[(d["epsilon"] == eps) & (d["smote"] == smote)]["recall_minority"]
            return s.mean() if len(s) else float("nan")
        eps = cfg["focus_epsilon"]
        inf = float("inf")
        rows.append({
            "n": n_total,
            "np_noSMOTE": mean_of(inf, False),
            "dp_noSMOTE": mean_of(eps, False),
            "np_SMOTE": mean_of(inf, True),
            "dp_SMOTE": mean_of(eps, True),
            "privacy_cost_noSMOTE": mean_of(inf, False) - mean_of(eps, False),
            "privacy_cost_SMOTE": mean_of(inf, True) - mean_of(eps, True),
            "smote_gain_dp": mean_of(eps, True) - mean_of(eps, False),
        })

    print(pd.DataFrame(rows).round(3).to_string(index=False))    
    print("\n--- Diagnostic: is minority recall genuine skill? ---")
    print("A model with NO discriminative power has precision equal to the")
    print("base rate. Precision at or below base_rate means the recall")
    print("figure does not indicate real detection ability.")
    print("pred_pos_rate far above base_rate means the model is shifting")
    print("its threshold (predicting minority more often), which raises")
    print("recall mechanically rather than through better discrimination.\n")
    diag = clf_results[clf_results["epsilon"] == cfg["focus_epsilon"]]
    diag_summary = diag.groupby(["n_total", "smote"])[
        ["recall_minority", "precision_minority", "f1_minority",
         "pred_pos_rate", "base_rate"]
    ].mean().round(3)
    print(diag_summary.to_string())