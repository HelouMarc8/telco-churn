import json
from pathlib import Path
import tempfile
import time

from matplotlib.figure import Figure
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

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
from telco_churn.ml_preprocessing import (
    build_pipeline,
    categorical_features,
    features,
    numeric_features,
)
from telco_churn.model_comparison import (
    THRESHOLD,
    evaluate_model,
    make_candidates,
    positive_scores,
)
from telco_churn.tracking import connect
from telco_churn.training import environment, git_identity

RUN_NAMES = {
    "baseline_most_frequent": "baseline-most-frequent",
    "logistic_regression": "logistic-C1",
    "decision_tree": "tree-depth5",
    "random_forest": "forest-200trees-depth8",
}

KEY_PARAMS = {
    "baseline_most_frequent": ["strategy"],
    "logistic_regression": ["C", "penalty", "solver", "max_iter"],
    "decision_tree": ["max_depth", "criterion", "min_samples_leaf"],
    "random_forest": ["n_estimators", "max_depth", "min_samples_leaf", "max_features"],
}
REFERENCE_SIZE = 10
CM_LABELS = [0, 1]  

def confusion_matrix_figure(metrics, title):
    matrix = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    fig = Figure(figsize=(4.5, 4))
    ax = fig.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=np.array(matrix),
        display_labels=["0 = No", "1 = Yes (churn)"],
    ).plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(f"{title}\nvalidation, threshold {THRESHOLD}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    return fig


def build_provenance(name, classifier, context):
    return {
        "candidate": name,
        "estimator": type(classifier).__name__,
        "estimator_params": classifier.get_params(),
        "features": {"numeric": numeric_features, "categorical": categorical_features},
        "excluded_from_features": ["customerID", "Churn"],
        "class_mapping": {"No": 0, "Yes": 1, "positive_class": 1},
        "preprocessing": {
            "numeric": "median imputation -> StandardScaler (fit on train only)",
            "categorical": "most_frequent imputation -> OneHotEncoder(handle_unknown='ignore')",
            "TotalCharges": "whitespace-only -> NaN before pipeline; other bad text raises",
        },
        "evaluation": {
            "dataset": "validation",
            "primary_metric": "average precision from predict_proba of class 1 (via classes_)",
            "threshold_metrics": ["precision", "recall", "f1"],
            "threshold": THRESHOLD,
            "confusion_matrix_order": "rows = true [0, 1], columns = predicted [0, 1]",
            "test_set_evaluated": False,
        },
        **context,
    }


def run_candidate(name, classifier, data, context, allocation_path):
    X_train, y_train, X_valid, y_valid, train_ds, valid_ds, reference_batch = data
    with mlflow.start_run(run_name=RUN_NAMES[name]) as run:
        params = classifier.get_params()
        mlflow.log_params({
            "model_type": type(classifier).__name__,
            "candidate": name,
            "split_seed": SEED,
            "threshold": THRESHOLD,
            "random_state": params.get("random_state"),
            "n_features": len(features),
            **{k: params[k] for k in KEY_PARAMS[name]},
        })
        mlflow.set_tags({
            "git_commit": context["code"]["commit"],
            "git_uncommitted_changes": str(context["code"]["uncommitted_changes"]),
            "data_sha256": context["data"]["source_sha256"],
            "split_allocation_sha256": context["split"]["allocation_sha256"],
        })
        mlflow.log_input(train_ds, context="training")
        mlflow.log_input(valid_ds, context="validation")

        pipeline = build_pipeline(classifier)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        metrics = evaluate_model(pipeline, X_valid, y_valid, threshold=THRESHOLD)
        mlflow.log_metrics({
            "val_average_precision": metrics["average_precision"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
            "val_accuracy": metrics["accuracy"],
            **{f"val_{k}": metrics[k] for k in ("tn", "fp", "fn", "tp")},
            "fit_seconds": fit_seconds,
        })

        mlflow.log_figure(confusion_matrix_figure(metrics, RUN_NAMES[name]),
                          "validation_confusion_matrix.png")
        mlflow.log_dict(
            {"labels_order": CM_LABELS, "rows": "true", "columns": "predicted",
             "matrix": [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]},
            "validation_confusion_matrix.json",
        )
        mlflow.log_dict(build_provenance(name, classifier, context), "provenance.json")
        mlflow.log_artifact(str(allocation_path))

        example = X_train.head(3).astype({"tenure": "float64"})
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="churn_pipeline",
            input_example=example,
            serialization_format="cloudpickle",
        )
        mlflow.set_tag("logged_model_uri", model_info.model_uri)

        scores = positive_scores(pipeline, reference_batch[features])
        reference_predictions = reference_batch[["customerID"]].assign(
            churn_probability=scores, churn_prediction=(scores >= THRESHOLD).astype(int)
        )
        mlflow.log_text(reference_batch.to_csv(index=False), "reference/reference_batch.csv")
        mlflow.log_text(
            reference_predictions.to_csv(index=False, float_format="%.17g"),
            "reference/reference_predictions.csv",
        )

        print(f"{RUN_NAMES[name]:<24} run {run.info.run_id}  "
              f"AP={metrics['average_precision']:.3f}  recall={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}  model={model_info.model_uri}")
        return run.info.run_id, pipeline


def main():
    connect()

    df = load_data(DATA_PATH)
    train_df, valid_df, test_df = split_data(df, seed=SEED)
    check_disjoint_splits(train_df, valid_df, test_df, df["customerID"].nunique())

    X_train, y_train = train_df[features], make_target(train_df)
    X_valid, y_valid = valid_df[features], make_target(valid_df)

    described = ["customerID", *features, "Churn"]
    train_ds = mlflow.data.from_pandas(train_df[described], name="telco_train", targets="Churn")
    valid_ds = mlflow.data.from_pandas(valid_df[described], name="telco_valid", targets="Churn")

    with tempfile.TemporaryDirectory() as tmp:
        allocation_path = Path(tmp) / "split_allocation.csv"
        split_allocation(train_df, valid_df, test_df).to_csv(allocation_path, index=False)
        context = {
            "data": {"source_path": str(DATA_PATH), "source_sha256": file_sha256(DATA_PATH),
                     "rows": len(df)},
            "split": {
                "seed": SEED,
                "method": "stratified on Churn, 60/20/20 train/valid/test",
                "sizes": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
                "allocation_artifact": "split_allocation.csv",
                "allocation_sha256": file_sha256(allocation_path),
            },
            "code": git_identity(),
            "environment": environment() | {"mlflow": mlflow.__version__},
        }

        reference_batch = valid_df[["customerID", *features]].head(REFERENCE_SIZE)
        data = (X_train, y_train, X_valid, y_valid, train_ds, valid_ds, reference_batch)
        results = {}
        for name, classifier in make_candidates().items():
            run_id, _pipeline = run_candidate(name, classifier, data, context, allocation_path)
            results[name] = run_id

    print("\nRuns:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
