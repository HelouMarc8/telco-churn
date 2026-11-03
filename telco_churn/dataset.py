import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DATA_PATH = Path("data/raw/telco.csv")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_total_charges(series):
    text = series.astype("string").str.strip()
    text = text.mask(text.eq(""))
    return pd.to_numeric(text, errors="raise").astype("float64")


def load_data(path=DATA_PATH):
    """Read the original CSV (never modified on disk) and parse TotalCharges in memory."""
    df = pd.read_csv(path)
    df["TotalCharges"] = parse_total_charges(df["TotalCharges"])
    return df


def make_target(df):
    y = df["Churn"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError(f"Unexpected Churn labels: {df.loc[y.isna(), 'Churn'].unique()}")
    return y


def split_data(df, seed=SEED):
    train_df, remaining_df = train_test_split(
        df, test_size=0.4, stratify=df["Churn"], random_state=seed
    )
    valid_df, test_df = train_test_split(
        remaining_df, test_size=0.5, stratify=remaining_df["Churn"], random_state=seed
    )
    return train_df, valid_df, test_df


def check_disjoint_splits(train_df, valid_df, test_df, n_customers):
    """Fail loudly if a customer appears in two splits or one is missing."""
    ids = [set(d["customerID"]) for d in (train_df, valid_df, test_df)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("customerID overlap between splits")
    if len(ids[0] | ids[1] | ids[2]) != n_customers:
        raise ValueError("splits do not cover every customer exactly once")


def split_allocation(train_df, valid_df, test_df):
    """customerID -> split name, to record exactly which customers were used where."""
    return pd.concat([
        pd.DataFrame({"customerID": d["customerID"], "split": name})
        for name, d in (("train", train_df), ("valid", valid_df), ("test", test_df))
    ]).sort_values("customerID").reset_index(drop=True)
