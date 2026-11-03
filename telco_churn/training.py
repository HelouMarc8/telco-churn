"""TP02 section 4 - training entry point: compare, select, save the fitted pipeline.

Run from the project root:
    uv run python -m telco_churn.training

Writes:
    reports/model_comparison.csv, reports/model_selection.json   (comparison, section 3)
    models/churn_pipeline.joblib            fitted preprocessing + classifier, together
    models/churn_pipeline.metadata.json     reproduction record for that artifact
    models/split_allocation.csv             customerID -> train/valid/test
    models/reference_batch.csv              a few validation customers, features only (no Churn)
    models/reference_predictions.csv        in-memory predictions on that batch, for reload check

The test set is not evaluated here.
"""
from datetime import datetime, timezone
import json
from importlib.metadata import version
import platform
import subprocess
import sys
from pathlib import Path

import joblib

from telco_churn.dataset import (
    DATA_PATH,
    SEED,
    check_disjoint_splits,
    file_sha256,
    load_data,
    make_target,
    split_allocation,
    split_data,
)
from telco_churn.ml_preprocessing import categorical_features, features, numeric_features
from telco_churn.model_comparison import (
    AP_TOLERANCE,
    THRESHOLD,
    compare_models,
    positive_scores,
    save_comparison,
    select_model,
)

MODEL_DIR = Path("models")
ARTIFACT = MODEL_DIR / "churn_pipeline.joblib"
METADATA = MODEL_DIR / "churn_pipeline.metadata.json"
ALLOCATION = MODEL_DIR / "split_allocation.csv"
REFERENCE_BATCH = MODEL_DIR / "reference_batch.csv"
REFERENCE_PREDICTIONS = MODEL_DIR / "reference_predictions.csv"
REFERENCE_SIZE = 10


def git_identity():
    """Commit hash and whether uncommitted changes exist (None if git is unavailable)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{pkg: version(pkg) for pkg in ("scikit-learn", "pandas", "numpy", "joblib")},
    }


def preprocessing_settings(pipeline):
    """Read the settings from the pipeline object itself, so the record cannot drift from it."""
    p = pipeline.get_params()
    return {
        "numeric_imputer": p["preprocessor__num__imputer__strategy"],
        "numeric_scaler": type(p["preprocessor__num__scaler"]).__name__,
        "categorical_imputer": p["preprocessor__cat__imputer__strategy"],
        "categorical_encoder": type(p["preprocessor__cat__encoder"]).__name__,
        "unknown_category_policy": p["preprocessor__cat__encoder__handle_unknown"],
        "total_charges_parsing": "whitespace-only -> NaN (then imputed); other non-numeric "
                                 "text raises ValueError. Done before the pipeline.",
    }


def save_reference(pipeline, valid_df):
    """Save a small feature-only batch and its in-memory predictions for the reload check."""
    batch = valid_df[["customerID", *features]].head(REFERENCE_SIZE)
    batch.to_csv(REFERENCE_BATCH, index=False)
    scores = positive_scores(pipeline, batch[features])
    reference = batch[["customerID"]].assign(
        churn_probability=scores, churn_prediction=(scores >= THRESHOLD).astype(int)
    )
    reference.to_csv(REFERENCE_PREDICTIONS, index=False, float_format="%.17g")


def main():
    df = load_data(DATA_PATH)
    train_df, valid_df, test_df = split_data(df, seed=SEED)
    check_disjoint_splits(train_df, valid_df, test_df, df["customerID"].nunique())

    X_train, y_train = train_df[features], make_target(train_df)
    X_valid, y_valid = valid_df[features], make_target(valid_df)

    results, fitted = compare_models(X_train, y_train, X_valid, y_valid)
    selected = select_model(results)
    save_comparison(results, selected, len(train_df), len(valid_df))

    # The selected candidate is already fitted on train only; save it as-is (no refit).
    pipeline = fitted[selected]

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, ARTIFACT)
    split_allocation(train_df, valid_df, test_df).to_csv(ALLOCATION, index=False)
    save_reference(pipeline, valid_df)

    selected_row = results.set_index("model").loc[selected]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": {"path": str(ARTIFACT), "sha256": file_sha256(ARTIFACT)},
        "selected_model": selected,
        "selection_rule": f"highest validation average precision; within {AP_TOLERANCE} "
                          "-> simplest model",
        "model_class": type(pipeline.named_steps["classifier"]).__name__,
        "model_params": json.loads(selected_row["params"]),
        "preprocessing": preprocessing_settings(pipeline),
        "feature_roles": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "identifier_not_a_feature": "customerID",
            "target_not_a_feature": "Churn",
        },
        "input_schema": {**{c: "float (blank allowed)" if c == "TotalCharges" else "number"
                            for c in numeric_features},
                         **{c: "string" for c in categorical_features}},
        "positive_class": "Churn = Yes -> 1",
        "decision_threshold": THRESHOLD,
        "validation_metrics": {
            k: float(selected_row[k])
            for k in ("average_precision", "precision", "recall", "f1", "accuracy")
        } | {k: int(selected_row[k]) for k in ("tn", "fp", "fn", "tp")},
        "fit_seconds": float(selected_row["fit_seconds"]),
        "comparison_table": "reports/model_comparison.csv",
        "data": {
            "source_path": str(DATA_PATH),
            "source_sha256": file_sha256(DATA_PATH),
            "rows": len(df),
        },
        "split": {
            "seed": SEED,
            "method": "stratified on Churn, 60/20/20 train/valid/test",
            "sizes": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
            "allocation_file": str(ALLOCATION),
            "allocation_sha256": file_sha256(ALLOCATION),
            "test_set_evaluated": False,
        },
        "code": git_identity(),
        "command": " ".join([Path(sys.executable).name, "-m", "telco_churn.training"]),
        "environment": environment(),
    }
    METADATA.write_text(json.dumps(metadata, indent=2))

    print(results.drop(columns="params").round(3).to_string(index=False))
    print(f"\nSelected: {selected}")
    print(f"Saved {ARTIFACT} and {METADATA}")


if __name__ == "__main__":
    main()
