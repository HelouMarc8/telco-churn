import argparse
from pathlib import Path
import subprocess

import mlflow

from telco_churn.model_comparison import AP_TOLERANCE, select_model
from telco_churn.tracking import EXPERIMENT_NAME, connect

OUTPUT = Path("reports/tp03_comparison.csv")
COLUMNS = {
    "run_id": "run_id",
    "tags.mlflow.runName": "run_name",
    "params.candidate": "model",
    "metrics.val_average_precision": "average_precision",
    "metrics.val_precision": "precision",
    "metrics.val_recall": "recall",
    "metrics.val_f1": "f1",
    "metrics.val_accuracy": "accuracy",
    "metrics.fit_seconds": "fit_seconds",
    "params.threshold": "threshold",
    "params.split_seed": "split_seed",
    "tags.git_commit": "git_commit",
    "tags.data_sha256": "data_sha256",
}


def resolve_commit(ref="HEAD"):
    return subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def fetch_runs(commit):
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=(
            f"tags.git_commit = '{commit}' "
            "and tags.git_uncommitted_changes = 'False' "
            "and attributes.status = 'FINISHED'"
        ),
    )
    if runs.empty:
        raise SystemExit(f"No clean finished runs for commit {commit}")
    table = runs[list(COLUMNS)].rename(columns=COLUMNS)
    table = table.assign(start=runs["start_time"]).sort_values("start")
    table = table.drop_duplicates("model", keep="last").drop(columns="start")
    return table.sort_values("average_precision", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Compare tracked runs and select a model.")
    parser.add_argument("--commit", help="git commit to compare (default: HEAD)")
    parser.add_argument(
        "--override",
        help="candidate chosen instead of the rule's pick (documented team decision)",
    )
    args = parser.parse_args()

    connect()
    commit = resolve_commit(args.commit or "HEAD")
    table = fetch_runs(commit)
    if table[["split_seed", "threshold", "data_sha256"]].nunique().max() > 1:
        raise SystemExit("Runs differ in split seed, threshold or data: not comparable")

    rule_pick = select_model(table)
    selected = args.override or rule_pick
    if selected not in set(table["model"]) or selected.startswith("baseline"):
        raise SystemExit(f"Cannot select {selected!r}: not a learned candidate in these runs")

    def decision(model):
        if model == selected:
            return "selected" if selected == rule_pick else "selected (override)"
        if model == rule_pick:
            return "rule pick, overridden"
        return "reference only" if model.startswith("baseline") else "not selected"

    table["decision"] = table["model"].map(decision)
    OUTPUT.parent.mkdir(exist_ok=True)
    table.to_csv(OUTPUT, index=False)

    shown = table[["run_id", "run_name", "average_precision", "recall", "f1", "decision"]]
    print(shown.round(3).to_string(index=False))
    print(
        f"\nRule: highest val AP; within {AP_TOLERANCE} -> simplest model. "
        f"Rule pick: {rule_pick}. Selected: {selected}  (commit {commit[:7]})"
    )
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
