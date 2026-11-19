# TP03 and TP04, explained simply

This file explains, in plain words, what we did in TP03 (MLflow) and TP04 (tests and code
quality), why we did it, and the answer to every question the two briefs ask.

Imagine our churn model is a **cake recipe**. In TP02 we learned to bake the cake (train the
model) and put it in a box (save it with joblib). TP03 is about **keeping a notebook of every cake
we ever baked** so we can find the best one again. TP04 is about **checking the kitchen** so that if
someone changes the recipe by mistake, an alarm goes off.

---

# Part 1 — TP03: Tracking experiments with MLflow

## The big idea

When you try four different models, after a week you forget which one was best, with which
settings, on which data, from which version of the code. MLflow is a **lab notebook that writes
itself**: every time we train, it writes down what we used, what score we got, and keeps a copy of
the trained model.

## The words, with a kid's picture

| Word | What it is | Kid's picture |
|---|---|---|
| **Experiment** | A named folder of attempts (`telco-churn-tp03`) | A notebook titled "Cake contest" |
| **Run** | One single training attempt, with its own ID | One page: "Tuesday, chocolate cake" |
| **Parameter** | A setting chosen *before* training (`max_depth=5`, `threshold=0.5`, `split_seed=42`) | The ingredients and oven temperature |
| **Metric** | A number measured *after* training (`val_average_precision=0.647`) | The score the judges gave |
| **Tag** | A label stuck on the run (`git_commit`, `data_sha256`) | A sticker: "made with flour bag #16320c…" |
| **Artifact** | A *file* saved with the run (confusion-matrix picture, `provenance.json`) | A photo of the cake glued on the page |
| **Dataset input** | A description of the data used (name, digest, schema) | "Used the train flour and the valid flour" — a label, not the flour itself |
| **Model** | The whole trained pipeline saved by MLflow, with its instructions | The actual cake, in a box, with a "how to open" sheet |
| **Model Registry** | A shop shelf of official models with version numbers | The shelf "Official cakes": version 1, version 2… |
| **Alias** | A movable sticker pointing at one version (`candidate`) | A sticker "Today's special" you can move to another cake |

## Step 1 — Start the tracking server

We ran:

```bash
uv run mlflow server --host 127.0.0.1 --port 5001 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

- The **server** is like a librarian: our training program hands it records, and the website
  (<http://127.0.0.1:5001>) shows them.
- It stores things in **two places**:
  - `mlflow.db` (a small database): the *words and numbers* (runs, parameters, metrics, tags,
    the registry).
  - `mlartifacts/` (a folder): the *files* (model files, pictures, JSON).
- We used port **5001** because on a Mac, port 5000 is already used by AirPlay.
- Our first tiny run was `smoke-test`, just to check the connection works.

**Questions from the brief:**

- *Experiment vs run?* The experiment is the notebook; a run is one page in it. One experiment
  holds many runs.
- *Is the browser page running our training code?* No. Our Python program trains on our computer
  and *sends* results to the server. The web page only *shows* what was stored, like a TV showing a
  recorded match.
- *If a colleague gets only `mlflow.db`, what else do they need?* The `mlartifacts/` folder (that is
  where the actual model file lives), plus our code package and a compatible Python environment
  (`uv.lock`). The database only says "the model is over there"; it does not contain it.

## Step 2 — Record what produced each result

`telco_churn/tracked_training.py` creates **one run per candidate** (baseline, logistic regression,
decision tree, random forest). In each run it records:

- **Parameters:** model type, important settings, seed 42, threshold 0.5.
- **Metrics:** `val_average_precision`, `val_precision`, `val_recall`, `val_f1` (plus accuracy and
  the confusion counts).
- **Tags:** the Git commit, whether there were uncommitted edits, the SHA-256 fingerprint of the
  CSV, and the fingerprint of the customer allocation.
- **Artifacts:** the confusion matrix (picture + JSON, with the order "rows = true, columns =
  predicted, 0 = No, 1 = Yes"), `provenance.json` (features, class mapping, rules, versions…), and
  `split_allocation.csv` (which customer went to train, valid or test).
- **Dataset inputs:** `telco_train` (context "training") and `telco_valid` (context "validation").

**Questions from the brief:**

- *Metric vs artifact?* A metric is **one number** in the database, easy to sort and compare across
  runs. An artifact is **a file** (picture, JSON, model) kept in the artifact folder.
- *What evidence distinguishes two different train/validation allocations from the same CSV?*
  The CSV fingerprint would be the same, so it is not enough. What differs is the
  **allocation**: the seed, the `split_allocation.csv` file and its SHA-256 tag, and the MLflow
  dataset digests (`telco_train` digest `9f14f013`, `telco_valid` digest `01b38f56`). If someone
  splits differently, those change.
- *Does recording a seed guarantee identical results on every machine?* No. The seed makes the
  *random choices* the same, but different library versions, operating systems or hardware can
  still change results slightly. That is why we also record library versions.
- *Can someone recover the training data from its hash?* No. A hash is like a fingerprint: it
  tells you "this is exactly that file" but you cannot rebuild a person from a fingerprint. You
  still need the CSV itself.
- *What if you log a commit while you have uncommitted edits?* The commit points to code that is
  *not* what actually ran, so the record lies a little. That is why we also record
  `git_uncommitted_changes` and why we **committed before the final runs**: our final runs say
  `False`.

## Step 3 — Compare at least three runs

All runs used the same 6 features, the same split (seed 42), the same metrics and the same
threshold. Validation results:

| Run | Average precision | Recall | F1 | Decision |
|---|---:|---:|---:|---|
| forest-200trees-depth8 | **0.667** | 0.484 | 0.555 | The rule's pick, overridden |
| logistic-C1 | 0.647 | 0.529 | 0.589 | **Selected** |
| tree-depth5 | 0.603 | 0.602 | 0.616 | Not selected |
| baseline-most-frequent | 0.265 | 0.000 | 0.000 | Reference only |

- **Average precision (AP)** answers: "if we sort customers from most to least likely to leave, are
  the real leavers near the top?" It was chosen **before** looking at results (in TP02).
- The baseline always says "nobody leaves". It is 73.5 % accurate and finds **zero** churners.
  That is why accuracy alone is a trap.
- Our rule said "highest AP wins, unless they are within 0.01". The forest wins by 0.020, so the
  rule picks the forest. **We chose logistic regression anyway**, and wrote down why: the gap is tiny
  and comes from one split; logistic regression is simpler, 10× faster, easy to explain, and finds
  more churners at 0.5. We did **not** choose it because of accuracy, and we did not change the
  metric after seeing the results.

**Questions from the brief:**

- *Why are results from different validation populations hard to compare?* If model A is tested on
  easy customers and model B on hard ones, A looks better even if it is not. It is like comparing
  two students who took different exams. Same exam for everyone = fair.
- *Why not select repeatedly on the test set?* If you keep peeking at the test set to choose, the
  test set becomes just another training aid, and its score stops being an honest "final exam". We
  never evaluated the test set.

## Step 4 — Log, register and reload the model

1. **Log:** every run saved its **whole fitted pipeline** (preprocessing + classifier) with
   `mlflow.sklearn.log_model`. MLflow saves the model *plus* a "how to open" sheet (`MLmodel`), the
   list of libraries (`requirements.txt`, `python_env.yaml`) and an input example.
2. **Register:** the selected logistic-regression model was put on the official shelf as
   `telco-churn`, **version 1**.
3. **Alias:** we stuck the `candidate` sticker on version 1.
4. **Reload in a new process:** `telco_churn/predict_registered.py` loads
   `models:/telco-churn@candidate` in a brand-new terminal, never trains, and compares its
   predictions with the ones the training run made: **max difference 0.0 → PASS**.

The four identifiers, each answering a different question:

| Identifier | Our value | Question it answers |
|---|---|---|
| Run ID | `973f0bca88fc43bc904e8842c698e118` | Which training attempt made it? |
| Logged model URI | `models:/m-9fd84ad6a2ef4206aaf9cfd976653d8e` | Which exact saved model file? |
| Registered name + version | `telco-churn` v1 | Which official version? |
| Alias | `candidate` → v1 | Which version is "today's special"? |

**Questions from the brief:**

- *What does the stored input schema describe?* The **shape of the door** the model accepts: six
  columns with names and types (`tenure`, `MonthlyCharges`, `TotalCharges` = numbers; `Contract`,
  `InternetService`, `PaymentMethod` = text), all required. It does not check that values make
  sense (a tenure of −5 still has the right type). We saved `tenure` as a decimal number so that a
  missing value is allowed.
- *Why is loading in a fresh process stronger evidence than predicting again in the training
  notebook?* In the notebook, the model is still sitting in memory, so of course it works. A new
  process starts with **nothing**: if it can still load and predict, the model truly survives on its
  own, like a cake that still tastes right after travelling to another house.
- *If the alias later points to a new version, does an already-loaded model change?* No. Once
  loaded, the model is a copy in memory. Moving the sticker on the shelf does not change the cake
  already on your plate. You must load `models:/telco-churn@candidate` again to get the new one.
  That is also why we keep the **version number** next to the alias for reproducible handovers.

## Step 5 — Restart and check nothing is lost

We stopped the server and started it again with the same command, from the same folder.

| | Before | Server stopped | After restart |
|---|---|---|---|
| Runs | 13 | — | 13 |
| `telco-churn` versions | [1] | — | [1] |
| `candidate` | → v1 | — | → v1 |
| Inference | PASS | cannot connect, keeps retrying | PASS |

Stopping the server is like closing the library for the night: the books are still on the
shelves (`mlflow.db` and `mlartifacts/`).

**Questions from the brief:**

- *What does each store keep?* `mlflow.db` keeps the **records** (runs, parameters, metrics, tags,
  dataset records, registry versions and aliases). `mlartifacts/` keeps the **files** (models,
  pictures, JSON).
- *Database survives, files lost?* You can still read every score and see the registry, but you
  **cannot load any model**: the records point to files that are gone. Like a library catalogue
  with empty shelves.
- *Only the files survive?* You could still open a model file, but you would not know which one is
  the selected one, how it was trained or how well it scored. Like a pile of books with no
  catalogue and no labels.
- *What must a colleague on another machine get separately?* Run IDs and `models:/` links only work
  on **our** server. They need a copy of `mlflow.db` + `mlartifacts/` (or a shared team server),
  our code at the recorded commit, an environment from `uv.lock`, and the original CSV if they want
  to retrain.

---

# Part 2 — TP04: Testing and code quality

## The big idea

A colleague changes one line of the preprocessing. Training still runs. Nothing crashes. But maybe
the predictions are now wrong. **"It runs" is not proof it works.** TP04 adds alarms:

- **Tests** check that the code *behaves* correctly.
- **Ruff** checks the code *looks* correct and has no obvious mistakes.
- **Coverage** shows which parts of the code the tests actually visited.
- **pre-commit** runs the quick checks automatically every time you try to save a commit.

## Step 1 — Tools and the contract

We installed `pytest`, `pytest-cov`, `ruff` and `pre-commit` as **development tools** (people
who only want to *use* the model do not need them).

Then we wrote the **contract**: the promises our prediction code makes. In short:

1. The six features are required; a missing one gives a clear error naming it.
2. `Churn` and `customerID` are never model inputs.
3. A blank `TotalCharges` becomes "missing"; text like `"abc"` is an error.
4. Missing values are filled with what was learned in training.
5. An unknown category (e.g. `"Satellite"`) does not crash.
6. One output row per customer.
7. `Yes` = class 1, probabilities between 0 and 1, rows sum to 1, the churn column is chosen
   through `classes_`.
8. Saving and reloading gives the same predictions.

**Checkpoint question — name four realistic defects the tests should catch:**

1. Someone removes the "missing column" check → confusing crash, or a column silently ignored.
2. Someone sets the encoder to reject unknown categories → one new payment method crashes the whole
   batch in production.
3. Someone takes the wrong probability column (`[:, 0]`) → we report the chance of **staying**
   instead of leaving, and nothing crashes. The worst kind of bug: silent.
4. Someone saves only the classifier, not the whole pipeline → the reloaded model cannot read raw
   data or predicts differently.

## Step 2 — The tests

A **fixture** is a small fake toy dataset made just for tests: 16 invented customers (6 leave, 10
stay), so tests run in **less than a second** without the real CSV and without MLflow. It is
different from the real dataset or a saved model: it exists only to try the code.

| Test | What it checks | Kid's picture |
|---|---|---|
| **A** missing feature | Removing `Contract` raises `ValueError: Missing required feature columns: ['Contract']` (checked for all 6 columns) | A form refuses to be sent if a box is empty, and says which one |
| **B** strange but allowed inputs | A blank `TotalCharges`, a missing `tenure` and `InternetService="Satellite"` still give one valid answer per customer. We first check that "Satellite" really was never seen in training | A new kid with an unusual name still gets a seat in class |
| **C** probability rules | Shape (n, 2), values between 0 and 1, each row sums to 1, the churn column is found through `classes_`, and churners get higher scores than non-churners on average | A weather app must say "70 % rain, 30 % no rain", never "130 %" |
| **D** save and reload | Save to a temporary folder (`tmp_path`), reload, same predictions; the whole pipeline is inside; nothing is re-trained | Freeze the cake, defrost it: it should taste exactly the same |

Extra checks: `Yes → 1`, `"abc"` in `TotalCharges` is rejected, the target never enters the model.

We also removed the template test that always failed (`assert False`), and marked the TP02 checks
that need the real CSV as `integration` (run them with `uv run pytest -m integration`).

Result: `uv run pytest -q` → **19 passed in 0.35 s**.

**Question from the brief:** *Does a successful prediction on an unknown category prove it is
accurate?* No. The model turns "Satellite" into "none of the internet types I know" and guesses.
The test proves it **does not crash**, not that the guess is right. It never saw a satellite
customer, so it cannot know.

## Step 3 — Prove a test catches a real bug

We broke the code on purpose, in one word:

```diff
-        ("encoder", OneHotEncoder(handle_unknown="ignore")),
+        ("encoder", OneHotEncoder(handle_unknown="error")),
```

Then: **1 failed, 18 passed**. Exactly test B failed with
`ValueError: Found unknown categories ['Satellite']`. We put the word back: **19 passed**.

Why this matters: a test that never fails is like a smoke alarm without a battery. We showed our
alarm really rings when there is smoke, and only that alarm rang, so it points to the problem.

## Step 4 — Ruff and coverage

**Ruff** reads the code without running it, like a spell-checker.

- `ruff check` (lint) looks for real mistakes: undefined names, unused variables, unused imports,
  wrongly ordered imports. We chose a small rule set: `E4`, `E7`, `E9` (basic errors), `F` (Pyflakes
  mistakes) and `I` (import order).
- `ruff format` makes every file look the same (spaces, quotes, line breaks). It changes the
  *look*, never the behaviour. We checked that: the code of every reformatted file is identical.

Before: 6 import-order problems and 14 files needing formatting. After: all clean.

Ruff also found a **real** problem: in the TP02 first-model script, `y_valid` was computed but
never used. Ruff's quick fix would have deleted it, which hides the problem. The real fix was to use
it: the script now prints the accuracy and the confusion matrix it was always supposed to produce
(two-feature model 0.783 vs baseline 0.735).

We also fixed three old scripts that did work as soon as they were imported (one even contained
the full path of this laptop). Now they only run when you call them.

**Coverage** colours the code the tests walked through.

- First result: **13 %** of the package. That sounds bad, but most of the code is training
  scripts and MLflow scripts that unit tests are *supposed* to avoid.
- Useful gap found: `split_data` and `check_disjoint_splits` (the code that prevents the same
  customer from being in both train and test) were never tested.
- We added `tests/test_split.py` (3 tests): sizes 60/20/20, same churn rate everywhere, same seed
  gives the same customers, and a customer copied into two splits is rejected. `dataset.py` went
  from **61 % to 82 %**.

**Questions from the brief:**

- *What does an executed line tell you?* Only that the tests **walked past** it, not that they
  checked it did the right thing.
- *Can a test execute a line without asserting the right behaviour?* Yes. If you deleted every
  `assert` from our tests, coverage would stay exactly the same, and nothing would be checked.
  Visiting a room is not the same as checking the windows are closed.
- *Which code is outside the coverage scope?* Other people's libraries (scikit-learn, pandas,
  MLflow). We measure only `telco_churn`, our own code.

## Step 5 — pre-commit hooks

**pre-commit** is a guard at the door of every commit. Our guard runs:

- Ruff check (with automatic fixes), then Ruff format,
- remove trailing spaces, make sure files end with a newline,
- check YAML and TOML files are valid.

Every tool is **pinned** to a real version (`v0.17.5` for Ruff, the same as the installed Ruff;
`v6.0.0` for the whitespace hooks) so everybody runs exactly the same checks.

**The demonstration:**

1. We wrote the new split test with messy style (unsorted imports, single quotes, no spaces).
2. `git commit` → **blocked**. Ruff fixed the imports and the formatter cleaned the style. No
   commit was created.
3. We looked at `git diff` to check the changes, ran the tests (still passing), did `git add`
   again, and committed again → **all hooks passed**, commit `ee4c4ba`.

We never used `--no-verify` (that would be sneaking past the guard).

Why no pytest or training in the hooks? The guard must be **fast**, or people get annoyed and skip
it. Tests will run in CI (TP05). Also, hooks live on each person's computer: someone can skip them
or never install them. So they help, but they cannot replace checks on the server.

## Step 6 — Handover and Quality Gate

The README now lists every command and what it detects. Caches and coverage files are ignored by
Git.

| Check | Question it answers |
|---|---|
| Tests | Does it **behave** correctly? |
| Lint (Ruff check) | Does it contain obvious **mistakes**? |
| Format (Ruff format) | Is it written in **one style**? |
| Coverage | Which code did the tests **visit**? |
| Quality Gate | Do the results meet the **team's rules** (for example "no new bugs, 80 % coverage on new code")? |

A **Quality Gate** (in a tool like SonarQube) is a rule of the team, like "you can only go to recess
if your homework has no mistakes and is at least 80 % done". The analysis can finish fine while the
gate still says "fail". We did not run SonarQube (it is optional), so we claim no SonarQube results.

**Questions from the brief:**

- *If tests pass, what can still be wrong?* Everything the tests do not check: a model that is
  valid but bad at finding churners, a poor threshold, customers changing over time, a bug in an
  untested script, or tests that run code without really checking it.
- *If a Quality Gate passes, is the model useful, fair or secure?* No. The gate checks the **code**
  against chosen rules. It does not know if the model predicts well on new customers, if it treats
  groups of customers fairly, if data is private, or if a library or a pickled model is unsafe.

---

# Commands cheat sheet

```bash
# TP03 — MLflow (terminal 1, keep running)
uv run mlflow server --host 127.0.0.1 --port 5001 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts

# TP03 — terminal 2
uv run python -m telco_churn.tracked_training
uv run python -m telco_churn.run_comparison --override logistic_regression
uv run python -m telco_churn.register_model
uv run python -m telco_churn.predict_registered --input models/reference_batch.csv --check

# TP04
uv sync --locked
uv run pre-commit install
uv run pytest -q
uv run pytest -q -m integration
uv run ruff check telco_churn tests
uv run ruff format --check telco_churn tests
uv run pytest --cov=telco_churn --cov-report=term-missing --cov-report=xml:coverage.xml
uv run pre-commit run --all-files
```

# Still to do by hand

- TP03 screenshots in `reports/figures/`: `tp03_run_comparison.png`, `tp03_selected_run.png`,
  `tp03_registry.png` (start the server first).
- Optional: SonarQube (TP04 extension).
