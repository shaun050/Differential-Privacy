import pandas as pd


def prepare_german_credit(raw_path="german.data", out_path="german_credit.csv"):
    cols = [
        "checking_status", "duration", "credit_history", "purpose",
        "credit_amount", "savings", "employment_since", "installment_rate",
        "personal_status_sex", "other_debtors", "residence_since",
        "property", "age", "other_installment_plans", "housing",
        "existing_credits", "job", "num_liable", "telephone",
        "foreign_worker", "class",
    ]
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, names=cols)
    
    df["target"] = df["class"].map({1: "good", 2: "bad"})
    df = df.drop(columns=["class"])
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {df.shape}, target counts:\n{df['target'].value_counts()}")
    return df


def prepare_mammographic_mass(raw_path="mammographic_masses.data", out_path="mammographic_mass.csv"):
    cols = ["birads", "age", "shape", "margin", "density", "class"]
    df = pd.read_csv(raw_path, header=None, names=cols, na_values="?")
    df = df.dropna()  # drop rows with missing values (documented ~5-15% of rows)
    df["target"] = df["class"].map({0: "benign", 1: "malignant"})
    df = df.drop(columns=["class"])
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {df.shape}, target counts:\n{df['target'].value_counts()}")
    return df


def prepare_student_performance(raw_path="student-mat.csv", out_path="student_performance.csv",
                                 pass_threshold=10):
    df = pd.read_csv(raw_path, sep=";")
    df["target"] = df["G3"].apply(lambda g: "pass" if g >= pass_threshold else "fail")
   
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {df.shape}, target counts:\n{df['target'].value_counts()}")
    return df





def prepare_bank_marketing(raw_path="bank.csv", out_path="bank_marketing.csv"):
    df = pd.read_csv(raw_path, sep=";")
    df["target"] = df["y"]  
    df = df.drop(columns=["y"])
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {df.shape}, target counts:\n{df['target'].value_counts()}")
    return df


def prepare_bank_marketing_full(raw_path="bank-full.csv", out_path="bank_marketing_full.csv"):
    
    df = pd.read_csv(raw_path, sep=";")
    df["target"] = df["y"]
    df = df.drop(columns=["y"])
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {df.shape}, target counts:\n{df['target'].value_counts()}")
    return df


if __name__ == "__main__":
    print("Run the individual prepare_* functions with the path to your")
    print("downloaded raw file, e.g.:")
    print('  python -c "from prepare_data import prepare_german_credit; prepare_german_credit(\'german.data\')"')
