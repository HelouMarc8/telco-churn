"""TP02 section 4 - inference only. Loads the saved pipeline; never calls fit.

Run from the project root:
    uv run python -m telco_churn.prediction --input models/reference_batch.csv
    uv run python -m telco_churn.prediction --input models/reference_batch.csv \
        --check models/reference_predictions.csv

The input CSV needs the six feature columns (customerID optional, Churn not needed).
Only load artifacts you trust: joblib files can execute code when loaded.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from telco_churn.dataset import parse_total_charges

ARTIFACT = Path("models/churn_pipeline.joblib")
METADATA = Path("models/churn_pipeline.metadata.json")
TOLERANCE = 1e-9


def load_model(artifact=ARTIFACT, metadata=METADATA):
    return joblib.load(artifact), json.loads(Path(metadata).read_text())


def prepare_input(df, required):
    """Boundary validation: missing columns fail visibly; TotalCharges parsed like in training."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    X = df[required].copy()
    X["TotalCharges"] = parse_total_charges(X["TotalCharges"])
    return X


def predict(pipeline, df, threshold):
    X = prepare_input(df, list(pipeline.feature_names_in_))
    positive_column = list(pipeline.classes_).index(1)
    scores = pipeline.predict_proba(X)[:, positive_column]
    out = pd.DataFrame({
        "churn_probability": scores,
        "churn_prediction": (scores >= threshold).astype(int),
    }, index=df.index)
    if "customerID" in df.columns:
        out.insert(0, "customerID", df["customerID"])
    return out


def check_against_reference(predictions, reference_path):
    reference = pd.read_csv(reference_path)
    same_ids = predictions["customerID"].tolist() == reference["customerID"].tolist()
    max_diff = float(np.max(np.abs(predictions["churn_probability"] - reference["churn_probability"])))
    same_classes = bool((predictions["churn_prediction"] == reference["churn_prediction"]).all())
    ok = same_ids and same_classes and max_diff <= TOLERANCE
    print(f"Reload check: same customers={same_ids}, same classes={same_classes}, "
          f"max |prob diff|={max_diff:.2e} (tolerance {TOLERANCE}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Predict churn with the saved pipeline.")
    parser.add_argument("--input", required=True, help="CSV with the six feature columns")
    parser.add_argument("--output", help="optional CSV path for predictions")
    parser.add_argument("--check", help="reference predictions CSV to compare against")
    args = parser.parse_args()

    pipeline, metadata = load_model()
    new_customers = pd.read_csv(args.input)
    predictions = predict(pipeline, new_customers, metadata["decision_threshold"])

    print(f"Model: {metadata['selected_model']} (artifact sha256 "
          f"{metadata['artifact']['sha256'][:12]}..., threshold {metadata['decision_threshold']})")
    print(predictions.to_string(index=False))
    if args.output:
        predictions.to_csv(args.output, index=False)
    if args.check and not check_against_reference(predictions, args.check):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
