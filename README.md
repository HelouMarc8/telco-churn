# Telco Churn

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Creating  a churn prediction model.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         telco_churn and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── telco_churn   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes telco_churn a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


## Experiment tracking with MLflow (TP03)

All commands run from the project root (`telco-churn/`). The raw CSV must be at
`data/raw/telco.csv` (SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`).

### 1. Start the tracking server (terminal 1, keep it running)

```bash
uv sync --locked
uv run mlflow server --host 127.0.0.1 --port 5001 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

UI: <http://127.0.0.1:5001>. Port 5001 is used because port 5000 is taken by macOS Control
Center (AirPlay Receiver). Always restart the server from this directory with the same options,
otherwise it opens a different, empty store.

### 2. Tracked training, comparison and registration (terminal 2)

Commit your code first: every run records the Git commit and whether there were uncommitted
changes.

```bash
uv run python -m telco_churn.tracked_training
uv run python -m telco_churn.run_comparison --override logistic_regression
uv run python -m telco_churn.register_model
```

- `tracked_training`: one run per candidate in experiment `telco-churn-tp03`, with parameters,
  validation metrics, Git/data tags, confusion matrix, provenance JSON, split allocation,
  train/validation dataset records and the complete fitted pipeline (`mlflow.sklearn.log_model`).
- `run_comparison`: exports the clean runs of the current commit to
  `reports/tp03_comparison.csv` and applies the selection rule (`--override` records a documented
  choice different from the rule's pick).
- `register_model`: registers the selected run's model as `telco-churn`, points alias
  `candidate` at the new version and writes `reports/tp03_handover.json`.

### 3. Inference through the registry alias (new terminal, server running)

```bash
uv run python -m telco_churn.predict_registered --input models/reference_batch.csv --check
```

Loads `models:/telco-churn@candidate` without training, applies the same input parsing as TP02,
and with `--check` compares against the predictions the source run made before logging. The input
CSV needs `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`,
`PaymentMethod` (`customerID` optional, `Churn` not needed). Only load models from your own
trusted server: loading unpickles code.

### Where tracking data are stored and how to recover the selected model

| Location | Contents |
|---|---|
| `mlflow.db` (SQLite backend store) | Experiments, runs, parameters, metrics, tags, dataset records, registry (`telco-churn` versions and aliases) |
| `mlartifacts/` (artifact store) | Logged files: fitted pipelines (`model.pkl`, `MLmodel`, requirements), confusion matrices, provenance, allocations |

Both are local to this machine and excluded from Git (`.gitignore`). To recover the selected
model: start the server as above, then load `models:/telco-churn@candidate`, or the fixed version
`models:/telco-churn/1` for a reproducible handover (the alias can be moved later).

**Limitation:** run IDs and `models:/` URIs only resolve against this server and these two
storage locations. A colleague on another machine needs a copy of both `mlflow.db` and
`mlartifacts/` (or a shared team server), the raw CSV, this repository at the recorded commit and
an environment from `uv.lock`.

## Tests and code quality (TP04)

All commands run from the project root.

### Setup (once, after cloning)

```bash
uv sync --locked
uv run pre-commit install
```

`uv sync` installs the runtime and the `dev` group (pytest, pytest-cov, ruff, pre-commit).
`pre-commit install` activates the Git hooks on your machine; the configuration
(`.pre-commit-config.yaml`) is versioned, but each clone must install the hooks itself.

### Commands

| Command | Detects |
|---|---|
| `uv run pytest -q` | Broken behaviour: input contract, unknown categories/missing values, probability contract, save/reload, split leakage. Fast, synthetic data only, no CSV, no MLflow |
| `uv run pytest -q -m integration` | TP02 reliability checks on the real CSV and saved artifacts (needs `data/raw/telco.csv` and `models/`) |
| `uv run ruff check telco_churn tests` | Static issues: syntax errors, undefined or unused names, bad imports, import order (rules E4, E7, E9, F, I) |
| `uv run ruff format --check telco_churn tests` | Formatting differences (does not change behaviour) |
| `uv run pytest --cov=telco_churn --cov-report=term-missing --cov-report=xml:coverage.xml` | Which lines the tests execute (not whether they assert the right thing) |
| `uv run pre-commit run --all-files` | Runs every hook (Ruff check with fixes, Ruff format, whitespace/end-of-file, YAML/TOML syntax) |

Apply fixes with `uv run ruff check --fix telco_churn tests` and `uv run ruff format telco_churn tests`,
then review `git diff`. When a commit is blocked because a hook modified files, inspect the changes,
`git add` them and commit again; do not use `--no-verify`. Hooks run locally and can be bypassed, so
they do not replace CI checks.
