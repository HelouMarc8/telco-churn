# TP03 Summary — MLflow tracking, comparison and model handover

Tracking server: local, `http://127.0.0.1:5001` (port 5000 is used by macOS Control Center).
Experiment: `telco-churn-tp03` (ID 1). Code commit for the compared runs: `fe641b9`.

## 1. Candidate comparison

Screenshot of the MLflow run comparison (4 runs, commit `fe641b9`):

![Run comparison](figures/tp03_run_comparison.png)

All four runs use the same six features (`tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`,
`InternetService`, `PaymentMethod`), the same seeded split (seed 42, stratified 60/20/20, same
customer allocation: `split_allocation_sha256` tag identical across runs), the same metric
definitions and the same threshold (0.5). Metrics are on the validation set (1409 customers,
374 churners); the test set was not evaluated. Exported from MLflow with
`uv run python -m telco_churn.run_comparison --override logistic_regression` →
`reports/tp03_comparison.csv`.

| Run ID | Candidate / changed setting | Validation AP | Validation recall | Validation F1 | Decision |
|---|---|---:|---:|---:|---|
| `45e37e388f404efdb87f12b7ae2a8910` | forest-200trees-depth8 (RandomForest, 200 trees, max_depth 8, min_samples_leaf 5) | 0.667 | 0.484 | 0.555 | Rule's pick, overridden |
| `1b6f58e016e84e8897dc3799441dab94` | logistic-C1 (LogisticRegression, C=1.0) | 0.647 | 0.529 | 0.589 | **Selected** |
| `584def22b16a4ec3a429f747ceafcd8b` | tree-depth5 (DecisionTree, max_depth 5) | 0.603 | 0.602 | 0.616 | Not selected |
| `183517f224e646ddbc0414e2fe068ee2` | baseline-most-frequent (DummyClassifier) | 0.265 | 0.000 | 0.000 | Reference only |

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

_To do (step 4): selected-run screenshot showing parameters, validation metrics and dataset records._

## 3. Registry

_To do (step 4)._

## 4. Identifiers

_To do (step 4): source run ID, logged model URI, registered version, alias, Git commit, data hash._

## 5. Fresh-process inference

_To do (step 4)._

## 6. Restart and storage

_To do (step 5)._
