import json
import time
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

from telco_churn.dataset import SEED, load_data, make_target, split_data
from telco_churn.ml_preprocessing import build_pipeline, features

THRESHOLD = 0.5
AP_TOLERANCE = 0.01
COMPLEXITY_ORDER = ["logistic_regression", "decision_tree", "random_forest"]
REPORT_DIR = Path("reports")


def make_candidates():
    """Fresh, unfitted classifiers with modest configurations."""
    return {
        "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=-1
        ),
    }


def positive_scores(model, X):
    """Probability of class 1 (Churn=Yes), found via classes_ rather than assuming column 1."""
    positive_column = list(model.classes_).index(1)
    return model.predict_proba(X)[:, positive_column]


def evaluate_model(model, X, y, threshold=THRESHOLD):
    """Metrics on an already-fitted model. AP uses continuous scores; the rest use the threshold."""
    scores = positive_scores(model, X)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": average_precision_score(y, scores),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "accuracy": accuracy_score(y, predictions),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_model(results, tolerance=AP_TOLERANCE):
    """Apply the pre-declared rule: best AP, ties within tolerance go to the simplest model."""
    learned = results[results["model"].isin(COMPLEXITY_ORDER)]
    best_ap = learned["average_precision"].max()
    close = learned[learned["average_precision"] >= best_ap - tolerance]
    return min(close["model"], key=COMPLEXITY_ORDER.index)


def compare_models(X_train, y_train, X_valid, y_valid):
    """Fit every candidate on train, evaluate on validation.

    Returns (results table, {name: fitted pipeline}).
    """
    rows, fitted = [], {}
    for name, classifier in make_candidates().items():
        candidate = build_pipeline(classifier)  # fresh preprocessing every time

        start = time.perf_counter()
        candidate.fit(X_train, y_train)  # training data only
        fit_seconds = time.perf_counter() - start

        metrics = evaluate_model(candidate, X_valid, y_valid)
        rows.append({
            "model": name,
            **metrics,
            "fit_seconds": round(fit_seconds, 4),
            "threshold": THRESHOLD,
            "params": json.dumps(classifier.get_params(), default=str),
        })
        fitted[name] = candidate

    results = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    return results, fitted


def save_comparison(results, selected, n_train, n_valid):
    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "model_selection.json").write_text(json.dumps({
        "selected_model": selected,
        "rule": f"highest validation average precision; within {AP_TOLERANCE} -> simplest "
                f"by {COMPLEXITY_ORDER}",
        "positive_class": "Churn = Yes (1)",
        "threshold": THRESHOLD,
        "split_seed": SEED,
        "features": features,
        "evaluated_on": "validation only (test reserved)",
        "n_train": n_train,
        "n_valid": n_valid,
    }, indent=2))


def main():
    df = load_data()
    train_df, valid_df, _test_df = split_data(df)  # test is reserved, deliberately unused

    X_train, y_train = train_df[features], make_target(train_df)
    X_valid, y_valid = valid_df[features], make_target(valid_df)

    results, _fitted = compare_models(X_train, y_train, X_valid, y_valid)
    selected = select_model(results)
    save_comparison(results, selected, len(train_df), len(valid_df))

    print(results.drop(columns="params").round(3).to_string(index=False))
    print(f"\nSelected: {selected}")


if __name__ == "__main__":
    main()
