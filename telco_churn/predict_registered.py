"""TP03 section 4 - inference only, through the registry alias. Never calls fit.

Run in a NEW terminal from the project root (server running):
    uv run python -m telco_churn.predict_registered --input models/reference_batch.csv --check

--input   CSV with the six feature columns (customerID optional, Churn not needed)
--check   compare with the predictions the source run made in memory before logging

Only load models from your own trusted tracking server: loading unpickles code.
"""
import argparse
from pathlib import Path
import tempfile

import mlflow
from mlflow import MlflowClient
import mlflow.artifacts
import mlflow.sklearn
import pandas as pd

from telco_churn.prediction import check_against_reference, predict
from telco_churn.tracking import ALIAS, REGISTERED_MODEL, TRACKING_URI

MODEL_URI = f"models:/{REGISTERED_MODEL}@{ALIAS}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="models/reference_batch.csv")
    parser.add_argument("--check", action="store_true",
                        help="compare with the source run's reference predictions")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    version = client.get_model_version_by_alias(REGISTERED_MODEL, ALIAS)
    run = client.get_run(version.run_id)
    threshold = float(run.data.params["threshold"])

    pipeline = mlflow.sklearn.load_model(MODEL_URI)  # complete fitted pipeline; no fit here
    new_customers = pd.read_csv(args.input)
    predictions = predict(pipeline, new_customers, threshold)  # same input parsing as TP02

    print(f"Loaded {MODEL_URI} -> version {version.version} "
          f"(source run {version.run_id}, {run.info.run_name}, threshold {threshold})")
    print(predictions.to_string(index=False))

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            reference = mlflow.artifacts.download_artifacts(
                run_id=version.run_id,
                artifact_path="reference/reference_predictions.csv",
                dst_path=tmp,
            )
            if not check_against_reference(predictions, Path(reference)):
                raise SystemExit(1)


if __name__ == "__main__":
    main()
