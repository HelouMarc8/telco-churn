# TP03 Summary — MLflow tracking, comparison and model handover

Tracking server: local, `http://127.0.0.1:5001` (port 5000 is used by macOS Control Center).
Experiment: `telco-churn-tp03` (ID 1). Code commit for the compared runs: `0aec61f`
(no uncommitted changes, recorded as the `git_uncommitted_changes = False` tag on every run).

## 1. Candidate comparison

Screenshot of the MLflow run comparison (4 runs, commit `0aec61f`):

![Run comparison](figures/tp03_run_comparison.png)

All four runs use the same six features (`tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`,
`InternetService`, `PaymentMethod`), the same seeded split (seed 42, stratified 60/20/20, same
customer allocation: `split_allocation_sha256` tag identical across runs), the same metric
definitions and the same threshold (0.5). Metrics are on the validation set (1409 customers,
374 churners); the test set was not evaluated. Each run logs its complete fitted pipeline.
Exported from MLflow with
`uv run python -m telco_churn.run_comparison --override logistic_regression` →
`reports/tp03_comparison.csv`.

| Run ID | Candidate / changed setting | Validation AP | Validation recall | Validation F1 | Decision |
|---|---|---:|---:|---:|---|
| `ed57351d676e44fd899791e4f3f1d99a` | forest-200trees-depth8 (RandomForest, 200 trees, max_depth 8, min_samples_leaf 5) | 0.667 | 0.484 | 0.555 | Rule's pick, overridden |
| `973f0bca88fc43bc904e8842c698e118` | logistic-C1 (LogisticRegression, C=1.0) | 0.647 | 0.529 | 0.589 | **Selected** |
| `55e0beda4c6747f1b58a9c029a70bc15` | tree-depth5 (DecisionTree, max_depth 5) | 0.603 | 0.602 | 0.616 | Not selected |
| `06ceac3a5f694f51b67bc370e25f5fdc` | baseline-most-frequent (DummyClassifier) | 0.265 | 0.000 | 0.000 | Reference only |

**Selection.** The primary metric, fixed in TP02 before any comparison, is validation average
precision from positive-class (`Churn = Yes`) probabilities; all three learned models clearly beat
the majority baseline (AP 0.265, equal to the churn rate). The pre-declared rule (highest AP, simpler
model within 0.01) picks the random forest, which leads logistic regression by 0.020 AP. We
nevertheless select **logistic regression** as a documented override: the gap is small and comes
from a single validation split with 374 churners, so it is not evidence of a real improvement,
while logistic regression is simpler, about 10× faster to fit, interpretable through its
coefficients, and at the 0.5 threshold finds more churners (recall 0.529 vs 0.484) with a higher F1
(0.589 vs 0.555). The primary metric itself is unchanged; accuracy (0.804 vs 0.794) was not used to
decide, since the baseline already reaches 0.735 accuracy while finding no churners.

## 2. Selected run

![Selected run](figures/tp03_selected_run.png)

Run `logistic-C1` (`973f0bca88fc43bc904e8842c698e118`) records:

- **Parameters:** `model_type=LogisticRegression`, `C=1.0`, `penalty`, `solver`, `max_iter=1000`,
  `split_seed=42`, `threshold=0.5`, `n_features=6`.
- **Metrics:** `val_average_precision=0.647`, `val_precision=0.664`, `val_recall=0.529`,
  `val_f1=0.589` (plus accuracy, confusion counts and fit time).
- **Tags:** `git_commit`, `git_uncommitted_changes`, `data_sha256`, `split_allocation_sha256`,
  `logged_model_uri`.
- **Dataset inputs:** `telco_train` (context `training`) and `telco_valid` (context `validation`).
  These describe the data (name, digest, schema); they do not store it. The MLflow digest is not the
  CSV's SHA-256; the original file is identified by the `data_sha256` tag and the exact allocation
  by `split_allocation.csv` and its hash.
- **Artifacts:** `validation_confusion_matrix.png/.json` (rows = true, columns = predicted, order
  [0 = No, 1 = Yes]), `provenance.json`, `split_allocation.csv`, `reference/` (10 validation
  customers and the in-memory predictions made before logging).

## 3. Registry

![Registry](figures/tp03_registry.png)

Registered model `telco-churn`, version **1**, alias **`candidate`** → version 1. The version is
tagged with its `source_run_id`. Registered with `uv run python -m telco_churn.register_model`.

## 4. Identifiers

| Identifier | Value | Answers |
|---|---|---|
| Source run ID | `973f0bca88fc43bc904e8842c698e118` | Which execution produced the model |
| Logged model URI | `models:/m-9fd84ad6a2ef4206aaf9cfd976653d8e` | Which recorded model artifact to load |
| Registered name + version | `telco-churn` version `1` (`models:/telco-churn/1`) | Which managed version |
| Alias | `candidate` → version 1 (`models:/telco-churn@candidate`) | Which version currently holds this role |
| Git commit | `0aec61fc8286c4b1963721a736be4522f44eff00` (no uncommitted changes) | Which code |
| Original data SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` | Which CSV bytes |
| Split allocation SHA-256 | `a981175e5fa0ef378bc7cea8b18f5db669157ad01a9f532239e30514bc791f16` | Which customers in train/valid/test |

Machine-readable copy: `reports/tp03_handover.json`.

**Logged input schema** (inferred from a 3-row training example, `tenure` cast to float so that a
missing value stays valid): `tenure`, `MonthlyCharges`, `TotalCharges`: double; `Contract`,
`InternetService`, `PaymentMethod`: string; all required. Output: integer class. It describes the
column names and types the model expects; it does not validate value ranges or category sets.

## 5. Fresh-process inference

Command, run in a new terminal from the project root with the server running:

```bash
uv run python -m telco_churn.predict_registered --input models/reference_batch.csv --check
```

The process loads `models:/telco-churn@candidate` with `mlflow.sklearn.load_model`, applies the
same TotalCharges parsing and column checks as TP02, and never calls `fit`. Result:

```
Loaded models:/telco-churn@candidate -> version 1 (source run 973f0bca88fc43bc904e8842c698e118, logistic-C1, threshold 0.5)
customerID  churn_probability  churn_prediction
1777-JYQPJ           0.391515                 0
5804-HYIEZ           0.353611                 0
2969-WGHQO           0.261210                 0
7354-OIJLX           0.677833                 1
5201-USSQZ           0.425583                 0
0017-IUDMW           0.062984                 0
0599-XNYDO           0.470135                 0
2146-EGVDT           0.003715                 0
3400-ESFUW           0.016812                 0
1699-UOTXU           0.025663                 0
Reload check: same customers=True, same classes=True, max |prob diff|=0.00e+00 (tolerance 1e-09) -> PASS
```

The reference predictions are the ones the training process computed in memory, before logging,
for the same 10 validation customers; they are stored in the source run. Loading in a separate
process is stronger evidence than predicting again in the training process: nothing from training
(variables, fitted objects, imports) is available, so the model must be fully recoverable from the
tracking server and artifact store alone.

If the alias later points to a new version, an already-loaded Python object does not change: it is
a copy in memory. The new version is only used after loading `models:/telco-churn@candidate` again.

## 6. Restart and storage

**Restart observation.** The server was stopped and restarted from the project root with the same
command (`--backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts`, port 5001).

| | Before stop | Server stopped | After restart |
|---|---|---|---|
| Runs in `telco-churn-tp03` | 13 | — | 13 |
| `telco-churn` versions | [1] | — | [1] |
| Alias `candidate` | → v1 (run `973f0bca…`) | — | → v1 (run `973f0bca…`) |
| Inference via alias | PASS, max diff 0.0 | client cannot connect and keeps retrying | PASS, max diff 0.0 |

Nothing was lost: stopping the server only stops the process serving requests. The records stay in
`mlflow.db` and the model files in `mlartifacts/`, and the restarted server reads them again.

**What do the database and the artifact store each preserve?** `mlflow.db` preserves the
metadata: experiments, runs, parameters, metrics, tags, dataset records, and the registry (model
names, versions, aliases, and where each version's files are). `mlartifacts/` preserves the files:
the pickled pipelines with their `MLmodel`, environment and input-example files, the confusion
matrices, provenance JSON, split allocations and reference predictions.

- If the database survived but the artifact files were lost, all runs, metrics and registry
  entries would still be visible, but no model could be loaded and no logged file opened: the
  records would point to files that no longer exist.
- If only the artifact files survived, the pickled models could still be found and loaded
  directly from their folders, but without their context: no parameters, metrics or tags, no run
  or dataset records, no `telco-churn` version or `candidate` alias to say which model was selected
  and why.

**What must a colleague on another machine obtain separately?** Run IDs and `models:/` URIs only
resolve against this local server. A colleague would need a copy of both `mlflow.db` and
`mlartifacts/` (or access to a shared tracking server), this repository at commit `0aec61f`, an
environment recreated from `uv.lock` (compatible scikit-learn), and, to retrain or verify the data,
the original CSV with SHA-256 `16320c9c…3055e91`; the hash identifies the file but cannot recreate
it.
