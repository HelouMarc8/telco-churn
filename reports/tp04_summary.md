# TP04 Summary — Testing and code quality

Branch `TD4`. Package: `telco_churn` (root-package layout, CCDS).

## 1. Setup and input/output contract

### Tools

Development tools are in the uv `dev` dependency group, not in the runtime dependencies
(`pytest` and `ruff` were moved out of `[project.dependencies]`):

```bash
uv add --dev pytest pytest-cov ruff pre-commit
```

| Tool | Version |
|---|---|
| pytest | 9.2.0 |
| pytest-cov | 7.1.0 |
| ruff | 0.17.5 |
| pre-commit | 4.7.0 |

Resolved versions are recorded in `uv.lock`; `uv sync --locked` succeeds.

### Input/output contract

The contract covers the prediction boundary `telco_churn.prediction.predict(pipeline, df,
threshold)` and the pipeline built by `telco_churn.ml_preprocessing.build_pipeline(classifier)`.

**Inputs**

1. **Required features.** Exactly these six columns are model inputs: numeric `tenure`,
   `MonthlyCharges`, `TotalCharges`; categorical `Contract`, `InternetService`, `PaymentMethod`.
   If any is missing, `predict` raises `ValueError` with a message naming the missing column(s)
   (`"Missing required feature columns: [...]"`), before the model is called.
2. **Not inputs.** `Churn` (the target) and `customerID` (an identifier) are never passed to the
   model. Extra columns in the batch are ignored; `customerID`, if present, is copied to the output
   to identify rows.
3. **Numeric parsing.** `TotalCharges` is parsed before the pipeline: a whitespace-only value
   becomes missing (NaN); any other non-numeric text (e.g. `"abc"`) raises `ValueError` instead of
   being silently treated as missing. `tenure` and `MonthlyCharges` must be numeric.
4. **Permitted missing values.** A missing numeric value is replaced by the training median; a
   missing categorical value by the training most frequent category. Both are learned during `fit`
   only.
5. **Unknown categories.** A category not seen during training is accepted and encoded as all
   zeros for that feature (`OneHotEncoder(handle_unknown="ignore")`); prediction still succeeds.
   This makes the model robust, not accurate, for such customers.

**Outputs**

6. **One row per customer.** A nonempty batch of *n* customers gives exactly *n* output rows, in
   the same order, with `churn_probability` and `churn_prediction`.
7. **Class encoding.** `Churn = Yes` is class `1` (positive), `No` is class `0`; the classifier's
   `classes_` is `[0, 1]`. `predict_proba` returns shape *(n, 2)* in that column order.
8. **Probabilities.** Every probability is finite and in [0, 1], and each row of `predict_proba`
   sums to 1. `churn_probability` is the column of class `1`, found through `classes_`, not by a
   hard-coded position.
9. **Decision.** `churn_prediction` is `1` when `churn_probability >= threshold` (0.5 by default),
   otherwise `0`.

**Persistence**

10. **Save/reload.** A fitted pipeline saved with joblib and loaded again, without refitting,
    returns the same class predictions and the same probabilities (within 1e-12) on the same inputs.

### Four realistic defects the tests must detect

| # | Realistic defect (e.g. a colleague's change) | What goes wrong | Contract | Planned test |
|---|---|---|---|---|
| 1 | The required-column check in `prepare_input` is removed or weakened, or a column is renamed upstream | A batch without `Contract` is rejected late with an unclear sklearn error, or a renamed column silently stops being used | 1 | A — missing feature rejected with `ValueError` naming the column |
| 2 | The encoder is changed to `handle_unknown="error"`, or the numeric imputer is removed from the pipeline | A new payment method or a blank `TotalCharges` in production makes the whole batch crash | 3, 4, 5 | B — missing numeric value and unseen category still give valid predictions |
| 3 | The positive-class column is hard-coded as `[:, 0]`, or the label mapping is flipped (`Yes → 0`) | The "churn probability" is actually the probability of staying: the riskiest customers are ranked last, with no error raised | 7, 8 | C — shape, bounds, row sums, and positive column taken from `classes_` |
| 4 | The save helper stores only the classifier instead of the complete pipeline, or a step keeps unfitted state | The reloaded model cannot accept raw columns, or gives different predictions from the trained one | 10 | D — save to `tmp_path`, reload, predictions identical |

"The program runs" would detect none of these: defects 1, 3 and 4 can all still produce output.

Tests will use a small synthetic fixture (about 16 invented customers, both classes) and the real
`build_pipeline`, without reading the Telco CSV or contacting MLflow. Planned files:

| Path | Responsibility |
|---|---|
| `tests/conftest.py` | Synthetic features/target and fitted-pipeline fixtures |
| `tests/test_input_contract.py` | Tests A and B |
| `tests/test_prediction.py` | Test C |
| `tests/test_persistence.py` | Test D |

## 2. Focused tests

`uv run pytest -q` → **19 passed, 18 deselected in 0.35 s**. The default suite uses only a synthetic
fixture: no Telco CSV is read, MLflow is not imported and nothing touches the network (checked
by running the suite with a file-open spy: `mlflow` absent from `sys.modules`, `telco.csv` never
opened).

**Fixture** (`tests/conftest.py`): 16 invented customers with the six features, `customerID` and a
separate binary target (6 churners, 10 non-churners), covering every category used in training
except the one tested as unseen. `fitted_pipeline` fits the real `build_pipeline` with
`LogisticRegression(max_iter=1000, random_state=0)` on this fixture only. Each test gets fresh
copies, so mutating a DataFrame cannot leak between tests.

**Test-to-risk mapping**

| Test | File | What it asserts | Real defect it catches |
|---|---|---|---|
| A `test_missing_required_feature_is_rejected` (+ parametrized for all six columns) | `test_input_contract.py` | Dropping a feature raises `ValueError` whose message names that column | Column check removed/weakened; column renamed upstream |
| `test_target_and_identifier_are_not_model_inputs` | `test_input_contract.py` | Pipeline inputs are exactly the six features; adding `Churn` changes nothing; `customerID` is passed through | Target leakage or identifier used as a feature |
| B `test_missing_numeric_and_unseen_category_still_predict` | `test_input_contract.py` | Blank `TotalCharges`, missing `tenure` and unseen `InternetService="Satellite"` (verified absent from the fixture and from the fitted encoder's categories) still give one finite probability in [0, 1] and a 0/1 prediction per row | `handle_unknown="error"`; imputer removed; blank parsing broken |
| `test_malformed_numeric_text_is_rejected_not_imputed` | `test_input_contract.py` | `TotalCharges="abc"` raises `ValueError` | Malformed text silently coerced to missing (`errors="coerce"`) |
| `test_target_mapping_makes_yes_the_positive_class`, `test_unexpected_target_label_is_rejected` | `test_input_contract.py` | `Yes → 1`, `No → 0`; unknown labels raise | Flipped class mapping; silent NaN targets |
| C `test_predict_proba_contract` | `test_prediction.py` | Shape (n, 2), `classes_ == [0, 1]`, finite, within [0, 1], rows sum to 1 | Wrong output shape or invalid probabilities |
| C `test_churn_probability_is_the_positive_class_column` | `test_prediction.py` | `churn_probability` equals the `classes_`-selected column of `predict_proba`; `churn_prediction = probability >= 0.5` | Hard-coded `[:, 0]`; wrong threshold rule |
| `test_churn_probability_points_towards_churners` | `test_prediction.py` | On the training fixture, mean score of churners > mean score of non-churners | Reported probability is "stay" instead of "churn" (direction check only, not a performance claim) |
| `test_one_output_row_per_customer` | `test_prediction.py` | Batches of 1, 5 and 16 give exactly that many rows and the documented columns | Rows dropped, duplicated or reordered |
| D `test_reload_preserves_predictions` | `test_persistence.py` | joblib save to `tmp_path` and reload give identical classes and probabilities (atol 1e-12) | Serialization changing behaviour |
| D `test_reloaded_artifact_is_the_complete_pipeline` | `test_persistence.py` | Reloaded object has `preprocessor` + `classifier` and gives the same output through `predict` on raw columns | Saving only the classifier |
| D `test_reload_does_not_refit` | `test_persistence.py` | Learned medians are identical before/after reload and unchanged by predicting | Hidden refitting at load or predict time |

**Suite organisation.** The CCDS placeholder `tests/test_data.py` (`assert False`) was removed. The
TP02 checks in `tests/test_reliability.py` read the real CSV and the saved artifact, so they are
marked `integration` and deselected by default (`[tool.pytest.ini_options]` in `pyproject.toml`);
run them with `uv run pytest -m integration` (18 passed).

**Does successful prediction on an unknown category prove it is accurate?** No. The encoder turns
`Satellite` into all zeros, so the model predicts as if the customer had none of the known internet
services. The test proves the pipeline does not crash and returns a valid probability; it says
nothing about whether that probability is right for satellite customers, a group the model never
saw.

## 3. Regression demonstration

**Defect introduced.** In `telco_churn/ml_preprocessing.py`, `build_pipeline` changed the
categorical encoder from the documented permissive policy to rejection, a realistic "make it
strict" edit:

```diff
-        ("encoder", OneHotEncoder(handle_unknown="ignore")),
+        ("encoder", OneHotEncoder(handle_unknown="error")),
```

Training still runs with this change, so nothing outside the tests would reveal it until a
production batch contained a new category.

**Result.** `uv run pytest -q` → **1 failed, 18 passed, 18 deselected**:

```
FAILED tests/test_input_contract.py::test_missing_numeric_and_unseen_category_still_predict

tests/test_input_contract.py:55
>       out = predict(fitted_pipeline, varied, THRESHOLD)
telco_churn/prediction.py:32: in predict
E       ValueError: Found unknown categories ['Satellite'] in column 1 during transform
```

**Why it failed.** Contract rule 5 says an unseen category is accepted and encoded as all zeros.
With `handle_unknown="error"`, the fitted encoder raises as soon as `predict` transforms the
customer with `InternetService="Satellite"`, so the whole batch of 16 customers gets no
prediction. Only the test covering that rule failed; the other 18 still passed, which points
directly at the cause.

**Fix.** The encoder was restored to `handle_unknown="ignore"` (`git diff` on
`ml_preprocessing.py` is empty again). Rerun: `uv run pytest -q` → **19 passed, 18 deselected**;
`uv run pytest -q -m integration` → **18 passed**. The submitted code is the working version.

## 4. Ruff and coverage

### Ruff

Configuration in `pyproject.toml` (existing `line-length = 99` and isort settings preserved):

```toml
[tool.ruff]
include = ["pyproject.toml", "telco_churn/**/*.py", "tests/**/*.py"]

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]
```

`E4`/`E7`/`E9`: pycodestyle errors for imports, statements and syntax/IO errors; `F`: Pyflakes
(undefined names, unused imports/variables, redefinitions); `I`: isort import ordering. The rule
set is selected explicitly: Ruff 0.17's default set also enabled pyupgrade, flynt and pylint rules,
which is wider than the agreed scope.

| | Before | After |
|---|---|---|
| `ruff check telco_churn tests` | 6 × I001 (import order) | All checks passed |
| `ruff format --check telco_churn tests` | 14 files would be reformatted | 19 files already formatted |

Fixes were reviewed: an AST comparison shows that every reformatted module has identical code
apart from import order. After fixing imports, Ruff reported one real finding,
`F841 Local variable 'y_valid' is assigned to but never used` in `ml_training_script.py`. Ruff's
suggested fix (delete the assignment) would have hidden the problem; the actual issue was that the
TP02 milestone-1 script never evaluated its predictions. It now reports validation accuracy and a
confusion matrix for the two-feature model and the majority baseline (0.783 vs 0.735).

Three TP01/TP02 scripts executed work at import time, which TP04 forbids for importable code:
`hash_test.py` hashed a hard-coded absolute path on import, `data_inspection.py` read the CSV and
wrote the report on import, and `ml_training_script.py` trained a model on import. Their work now
runs only under `if __name__ == "__main__":`, `hash_test.py` uses a path relative to the project,
and the regenerated inspection report is byte-identical to the previous one.

### Coverage

```bash
uv run pytest --cov=telco_churn --cov-report=term-missing --cov-report=xml:coverage.xml
```

First result (19 unit tests): **TOTAL 13 %** (483 statements, 418 missed). Per module:
`dataset.py` 61 %, `ml_preprocessing.py` 62 %, `prediction.py` 56 %; all command-line and MLflow
modules (`training.py`, `tracked_training.py`, `register_model.py`, `run_comparison.py`,
`predict_registered.py`, `tracking.py`, `model_comparison.py`, `data_inspection.py`) 0 %.

The low total is expected: most uncovered statements are entry points that train on the real CSV
or contact the MLflow server, which the unit suite must not do. Third-party code (scikit-learn,
pandas, MLflow) is outside the measured scope (`--cov=telco_churn`).

**Uncovered relevant behaviour:** `split_data` and `check_disjoint_splits` in `dataset.py`
(lines 35–49). They protect against leakage between train, validation and test, yet no unit test
executed them; only the slow integration test on the real CSV did.

**Proposed and implemented test** (`tests/test_split.py`, 3 tests on 100 synthetic customers):
the split gives 60/20/20 rows with the churn rate preserved in each part, the same seed gives the
same customer allocation, and a customer copied into two splits makes `check_disjoint_splits`
raise. After adding it: `dataset.py` **61 % → 82 %**, total 15 %, 22 tests passed.

**What coverage does and does not tell us.** An executed line only means the tests ran it, not that
they checked its result: deleting every `assert` from the suite would keep coverage identical. For
example, `test_one_output_row_per_customer` executes all of `predict`, but would not notice a
wrong threshold rule; that is caught by a separate assertion in
`test_churn_probability_is_the_positive_class_column`. Coverage finds untested code; it does not
prove tested code is correct.

**Still uncovered and worth a test next:** `prediction.check_against_reference` (lines 46–57), the
reload agreement check. A useful test would write a reference CSV to `tmp_path` and assert that a
probability changed by more than the tolerance makes it return `False`.

## 5. pre-commit

`.pre-commit-config.yaml` (pinned, Ruff hook version = installed Ruff 0.17.5, lint fixes before
formatting, Python hooks scoped to `telco_churn/` and `tests/`):

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.17.5
    hooks:
      - id: ruff-check
        args: [--fix]
        types_or: [python, pyi]
        files: ^(telco_churn|tests)/
      - id: ruff-format
        types_or: [python, pyi]
        files: ^(telco_churn|tests)/
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
        files: ^(telco_churn|tests)/|\.(toml|yaml|yml)$
      - id: end-of-file-fixer
        files: ^(telco_churn|tests)/|\.(toml|yaml|yml)$
      - id: check-yaml
      - id: check-toml
```

Hooks only see Git-tracked files, so `.venv/`, `mlflow.db` and `mlartifacts/` are never scanned.

```bash
uv run pre-commit validate-config   # valid
uv run pre-commit install           # pre-commit installed at .git/hooks/pre-commit
uv run pre-commit run --all-files   # first run: trailing-whitespace fixed pyproject.toml, then all Passed
```

**Blocked commit.** `tests/test_split.py` (the new coverage test) was written with correct
behaviour but bad style: unsorted imports, single quotes, missing spaces
(`def test_x( customers ):`, `train,valid,test=split_data(...)`). It was staged and committed:

```
$ git commit -m "TD4: add split leakage tests"
ruff check...............................................................Failed
- hook id: ruff-check
- files were modified by this hook
Found 1 error (1 fixed, 0 remaining).
ruff format..............................................................Failed
- hook id: ruff-format
- files were modified by this hook
1 file reformatted
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
```

No commit was created (`HEAD` stayed at `22d5a43`). `git diff tests/test_split.py` showed the
hooks' changes: imports sorted (`check_disjoint_splits, split_data`), double quotes, spaces around
operators and after commas. The tests still passed (3 passed).

**Successful retry.** After reviewing the diff: `git add tests/test_split.py` and the same commit
→ all hooks Passed, commit `ee4c4ba`. `--no-verify` was not used.

pytest and full training are deliberately not in the hooks: hooks must stay fast, and the tests
will run in CI (TP05). Hooks are local and can be skipped (`--no-verify`, or a clone where
`pre-commit install` was never run), so they cannot replace server-side checks.

## 6. Handover and Quality Gate reflection

The README section "Tests and code quality (TP04)" lists setup (`uv sync --locked`,
`uv run pre-commit install`) and each command with the kind of issue it detects.
`.venv/`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`, `coverage.xml` and `htmlcov/` are
ignored by Git.

| Check | Question it answers | Example from this project |
|---|---|---|
| Tests (pytest) | Does the code *behave* as the contract says? | `handle_unknown="error"` made test B fail |
| Static analysis (Ruff lint) | Is the code suspicious without running it? | Unused `y_valid` (F841) revealed a script that never evaluated its model |
| Formatting (Ruff format) | Is the code written in one consistent style? | 14 files reformatted, no behaviour change |
| Coverage | Which code did the tests execute? | Split functions were never executed by unit tests |
| Quality Gate (policy) | Do the analysis results meet the team's thresholds? | e.g. "no new bugs, coverage on new code ≥ 80 %" |

A Quality Gate is a policy applied to analysis results; analysis can complete successfully while
the gate fails, and the result depends on the thresholds and the definition of "new code" chosen.
SonarQube was not run (optional); no SonarQube results are claimed.

**If tests pass, what can still be wrong?** Anything the tests do not assert: a model that is
valid but useless, a threshold that misses most churners, data drift in production, a bug in an
untested script, or tests that execute code without checking it. **If a Quality Gate passes, is
the model useful, fair or secure?** No. The gate checks code against chosen thresholds; it says
nothing about predictive quality on new customers, fairness across customer groups, data
privacy, or vulnerabilities in dependencies and pickled models.
