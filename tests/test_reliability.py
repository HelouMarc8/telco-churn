"""TP02 section 5 - focused reliability checks on the saved, selected pipeline.

Prerequisite (creates the artifact):  uv run python -m telco_churn.training
Run from the project root:             uv run pytest tests/test_reliability.py -v
"""
import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from telco_churn.dataset import DATA_PATH, load_data, split_data
from telco_churn.ml_preprocessing import categorical_features, features, numeric_features
from telco_churn.prediction import ARTIFACT, METADATA, load_model, predict, prepare_input

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def run_from_project_root(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)  # all project paths are relative to the root


@pytest.fixture(scope="module")
def splits():
    df = load_data(PROJECT_ROOT / DATA_PATH)
    return (df, *split_data(df))


@pytest.fixture(scope="module")
def saved():
    if not (PROJECT_ROOT / ARTIFACT).exists():
        pytest.skip("No artifact: run `uv run python -m telco_churn.training` first")
    return load_model(PROJECT_ROOT / ARTIFACT, PROJECT_ROOT / METADATA)  # (pipeline, metadata)


def numeric_imputer(pipeline):
    return pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]


def one_valid_customer(splits):
    _, _, valid_df, _ = splits
    return valid_df[features].head(1).reset_index(drop=True)


# 1. Customer IDs do not overlap across train, validation and test ---------------------------
def test_customer_ids_disjoint_and_complete(splits):
    df, train_df, valid_df, test_df = splits
    tr, va, te = (set(d["customerID"]) for d in (train_df, valid_df, test_df))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == df["customerID"].nunique() == len(df)


def test_saved_allocation_matches_current_split(splits):
    _, train_df, valid_df, test_df = splits
    saved_alloc = pd.read_csv("models/split_allocation.csv").set_index("customerID")["split"]
    for name, d in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        assert (saved_alloc.loc[d["customerID"]] == name).all()


# 2. Learned statistics come from TRAINING data, not the full dataset -------------------------
def test_imputer_medians_are_training_medians(splits, saved):
    df, train_df, _, _ = splits
    pipeline, _ = saved
    learned = numeric_imputer(pipeline).statistics_
    train_medians = train_df[numeric_features].median().to_numpy()
    full_medians = df[numeric_features].median().to_numpy()

    np.testing.assert_allclose(learned, train_medians)
    print("\nlearned :", dict(zip(numeric_features, learned)))
    print("train   :", dict(zip(numeric_features, train_medians)))
    print("full    :", dict(zip(numeric_features, full_medians)))
    # Evidence that the comparison is meaningful: train and full-data medians actually differ.
    assert not np.allclose(train_medians, full_medians)


def test_scaler_mean_is_training_mean(splits, saved):
    _, train_df, _, _ = splits
    pipeline, _ = saved
    num = pipeline.named_steps["preprocessor"].named_transformers_["num"]
    imputed_train = train_df[numeric_features].fillna(train_df[numeric_features].median())
    np.testing.assert_allclose(num.named_steps["scaler"].mean_, imputed_train.mean().to_numpy())


# 3. Predicting does not refit or change learned statistics -----------------------------------
def test_predict_does_not_change_learned_state(splits, saved):
    _, _, valid_df, _ = splits
    pipeline, metadata = saved
    num = pipeline.named_steps["preprocessor"].named_transformers_["num"]
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
    before = (num.named_steps["imputer"].statistics_.copy(),
              num.named_steps["scaler"].mean_.copy(),
              copy.deepcopy(encoder.categories_))

    predict(pipeline, valid_df, metadata["decision_threshold"])

    np.testing.assert_array_equal(before[0], num.named_steps["imputer"].statistics_)
    np.testing.assert_array_equal(before[1], num.named_steps["scaler"].mean_)
    for a, b in zip(before[2], encoder.categories_):
        np.testing.assert_array_equal(a, b)


# 4a. Permitted missing numeric input -> training median (documented policy) ------------------
def test_blank_total_charges_is_imputed_with_training_median(splits, saved):
    _, train_df, _, _ = splits
    pipeline, metadata = saved
    customer = one_valid_customer(splits)

    blank = customer.assign(TotalCharges=" ")  # as it appears in the raw CSV
    as_median = customer.assign(TotalCharges=train_df["TotalCharges"].median())

    p_blank = predict(pipeline, blank, metadata["decision_threshold"])["churn_probability"]
    p_median = predict(pipeline, as_median, metadata["decision_threshold"])["churn_probability"]
    np.testing.assert_allclose(p_blank, p_median)


def test_malformed_total_charges_fails_visibly(splits, saved):
    pipeline, metadata = saved
    bad = one_valid_customer(splits).assign(TotalCharges="abc")
    with pytest.raises(ValueError):
        predict(pipeline, bad, metadata["decision_threshold"])


# 4b. Unseen category -> encoded as all zeros, no crash (documented policy) -------------------
def test_unseen_category_is_ignored_not_crashing(splits, saved):
    pipeline, metadata = saved
    unseen = one_valid_customer(splits).assign(InternetService="Satellite")

    preprocessor = pipeline.named_steps["preprocessor"]
    encoded = preprocessor.transform(prepare_input(unseen, features))
    encoded = encoded.toarray() if hasattr(encoded, "toarray") else encoded
    names = preprocessor.get_feature_names_out()
    internet_cols = [i for i, n in enumerate(names) if n.startswith("cat__InternetService_")]
    assert np.all(encoded[0, internet_cols] == 0)

    out = predict(pipeline, unseen, metadata["decision_threshold"])
    assert 0.0 <= out["churn_probability"].iloc[0] <= 1.0
    # Note: handling an unseen category is not evidence the prediction is reliable for it.


# 5. Missing required feature -> visible failure ----------------------------------------------
@pytest.mark.parametrize("dropped", numeric_features + categorical_features)
def test_missing_required_feature_raises(splits, saved, dropped):
    pipeline, metadata = saved
    incomplete = one_valid_customer(splits).drop(columns=dropped)
    with pytest.raises(ValueError, match=dropped):
        predict(pipeline, incomplete, metadata["decision_threshold"])


def test_pipeline_itself_also_rejects_missing_column(splits, saved):
    """Even bypassing our boundary check, sklearn refuses rather than predicting silently."""
    pipeline, _ = saved
    incomplete = one_valid_customer(splits).drop(columns="Contract")
    with pytest.raises(ValueError):
        pipeline.predict_proba(incomplete)


# 6. Save/reload preserves predictions; inference needs no labels or training -----------------
def test_reference_batch_has_no_labels():
    batch = pd.read_csv("models/reference_batch.csv")
    assert "Churn" not in batch.columns


def test_fresh_process_reload_matches_training_predictions():
    result = subprocess.run(
        [sys.executable, "-m", "telco_churn.prediction",
         "--input", "models/reference_batch.csv",
         "--check", "models/reference_predictions.csv"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_inference_code_never_fits():
    source = (PROJECT_ROOT / "telco_churn" / "prediction.py").read_text()
    assert ".fit(" not in source and "fit_transform" not in source
