"""TP03 section 4 - register the selected logged pipeline and point the alias at it.

Run from the project root (server running), after run_comparison:
    uv run python -m telco_churn.register_model                 # run marked selected in the CSV
    uv run python -m telco_churn.register_model --run-id <id>   # explicit run

Registers the run's logged model as `telco-churn` and sets alias `candidate` on the
returned version (the version number is read back, never assumed to be 1).
"""
import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import pandas as pd

from telco_churn.tracking import ALIAS, REGISTERED_MODEL, connect

COMPARISON = Path("reports/tp03_comparison.csv")
HANDOVER = Path("reports/tp03_handover.json")


def selected_run_id(comparison=COMPARISON):
    table = pd.read_csv(comparison)
    chosen = table[table["decision"].str.startswith("selected")]
    if len(chosen) != 1:
        raise SystemExit(f"Expected exactly one selected run in {comparison}, found {len(chosen)}")
    return chosen["run_id"].iloc[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", help="source run (default: selected run in the CSV)")
    args = parser.parse_args()

    connect()
    client = MlflowClient()
    run_id = args.run_id or selected_run_id()
    run = client.get_run(run_id)
    model_uri = run.data.tags.get("logged_model_uri")
    if not model_uri:
        raise SystemExit(f"Run {run_id} has no logged model (tag logged_model_uri missing)")

    version = mlflow.register_model(model_uri, REGISTERED_MODEL)
    client.set_registered_model_alias(REGISTERED_MODEL, ALIAS, version.version)
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "source_run_id", run_id)

    handover = {
        "source_run_id": run_id,
        "run_name": run.info.run_name,
        "logged_model_uri": model_uri,
        "registered_model": REGISTERED_MODEL,
        "registered_version": int(version.version),
        "alias": ALIAS,
        "alias_uri": f"models:/{REGISTERED_MODEL}@{ALIAS}",
        "version_uri": f"models:/{REGISTERED_MODEL}/{version.version}",
        "git_commit": run.data.tags.get("git_commit"),
        "git_uncommitted_changes": run.data.tags.get("git_uncommitted_changes"),
        "data_sha256": run.data.tags.get("data_sha256"),
        "split_allocation_sha256": run.data.tags.get("split_allocation_sha256"),
        "threshold": float(run.data.params["threshold"]),
        "val_average_precision": run.data.metrics["val_average_precision"],
    }
    HANDOVER.write_text(json.dumps(handover, indent=2))
    print(json.dumps(handover, indent=2))
    print(f"Saved {HANDOVER}")


if __name__ == "__main__":
    main()
