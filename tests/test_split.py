import pandas as pd
import pytest

from telco_churn.dataset import check_disjoint_splits, split_data


@pytest.fixture
def customers():
    churn = ["Yes"] * 25 + ["No"] * 75
    return pd.DataFrame({"customerID": [f"K{i:03d}" for i in range(100)], "Churn": churn})


def test_split_is_disjoint_complete_and_stratified(customers):
    train, valid, test = split_data(customers, seed=42)
    assert (len(train), len(valid), len(test)) == (60, 20, 20)
    check_disjoint_splits(train, valid, test, n_customers=100)
    for part in (train, valid, test):
        assert (part["Churn"] == "Yes").mean() == pytest.approx(0.25, abs=0.05)


def test_same_seed_gives_same_allocation(customers):
    first = split_data(customers, seed=42)
    second = split_data(customers, seed=42)
    for a, b in zip(first, second):
        assert list(a["customerID"]) == list(b["customerID"])


def test_overlapping_splits_are_rejected(customers):
    train, valid, test = split_data(customers, seed=42)
    leaked = pd.concat([valid, train.head(1)])
    with pytest.raises(ValueError, match="overlap"):
        check_disjoint_splits(train, leaked, test, n_customers=100)
