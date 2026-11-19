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
