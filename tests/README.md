# BSIX Tests

This folder contains the automated tests for BSIX (*Benchmarking Survival analysis: Inference and eXplainability*). The suite checks data processing, metrics, orchestration, result loading, and complete survival workflows.

## Requirements

Run the commands from the repository root (`bsix/`), not from `tests/`. First install the project and test dependencies:

```bash
python -m pip install -e .
python -m pip install -r tests/requirements-test.txt
```

`tests/conftest.py` imports PyTorch when loading the suite, so `torch` is required even to collect the tests. Tests using deep learning models, SHAP, or `SurvivalTabPFN` also require their dependencies and may need more time or resources.

## Structure

```text
tests/
├── conftest.py
├── requirements-test.txt
├── test_data/
│   ├── test_load_data.py
│   └── test_survival_utils.py
├── test_metrics/
│   ├── test_calculate_metrics.py
│   ├── test_classification_metrics.py
│   └── test_survival_metrics.py
├── test_execution/
│   └── test_orchestrator.py
├── test_results/
│   └── test_load_results.py
└── test_integration/
    ├── test_edge_cases.py
    └── test_end_to_end.py
```

The `__init__.py` files make the folders behave as packages.

### Covered areas

- **`test_data/`**: `_prepare_data`, `get_data`, and survival utilities. The suite checks train/validation/test splits, reproducibility, categorical variable encoding, `NaN` removal, invalid times, time-varying and multi-event formats, and feature-name preservation.
- **`test_metrics/`**: calculation and aggregation of survival and classification metrics. It includes Harrell's C-Index, IPCW C-Index, dynamic AUC, Brier Score, CCR, AMAE, GM, MAE, MS, recall, MZE, MMAE, and weighted kappa, as well as multi-event metrics with `P0`, `P1`, etc. prefixes.
- **`test_execution/`**: configuration generation, executor parameters, workers, configuration loading, and the existence/signature of the survival workflow. Invalid configurations are also tested.
- **`test_results/`**: loading results from `_results_`, filtering by estimator, dataset, and seed, accessing configuration/data, and temporary Excel report generation.
- **`test_integration/`**: complete data → model → predictions → metrics pipelines, cross-validation, `NaN` handling, determinism, SHAP explainability, and import checks. `AcceleratedFailureTime`, `CoxRegression`, `CoxRegressionWithTimeVarying`, `DeepHit`, `DeepMultiTask`, `DeepSurv`, `DeepTimeVarying`, `RandomSurvForest`, `SurvivalTabPFN`, and `SurvTree` are covered.
- **Edge cases**: empty or single-sample sets, invalid, infinite, and `NaN` times, invalid seeds, invalid predictions, fully censored data, and configuration errors.

## Shared fixtures

`conftest.py` contains function-scoped fixtures and a session-scoped fixture for PyTorch seeds (`setup_pytorch_seed`). Data is generated deterministically with seed `42`, unless a test specifies otherwise.

### Data

- `simple_survival_data` and `survival_data`: structured data with `event` and `time` fields (100 and 5000 samples, respectively).
- `*_timevarying` variants with `time_start` and `time_stop`.
- `*_multi_event` variants for two competing events.
- `*_all_censored`, `single_*`, and `wrong_dtype_*` variants for robustness.

### DataFrames

- `dataframe_mixed_types`, `dataframe_time_varying`, and `dataframe_multi_event`.
- `dataframe_with_nans` and its time-varying/multi-event variants.
- `dataframe_with_invalid_times`, `dataframe_empty`, and `dataframe_single_row` (including format variants).

### Predictions and configuration

- Risk, binary, and survival-function predictions, including perfect, inverse, random, and single-sample variants.
- `basic_execution_config`, `basic_experiment_config`, `invalid_estimator_config` e `invalid_dataset_config`.

## Execution

The [pytest.ini](../pytest.ini) configuration sets `tests/` as the discovery path, recognizes `test_*.py` and `test_*`, adds `src` to `PYTHONPATH`, uses verbose output, short tracebacks, a result summary, and a 300-second timeout.

### Full suite

```bash
pytest
```

Useful equivalent commands:

```bash
# Discover tests without running them
pytest --collect-only -q

# Compact summary or maximum information
pytest -q
pytest -vv -s

# Stop at the first failure or after N failures
pytest -x
pytest --maxfail=3

# Show the slowest tests
pytest --durations=10
```

### Run by area, file, class, or function

```bash
pytest tests/test_data/ -v
pytest tests/test_metrics/test_survival_metrics.py -v
pytest tests/test_metrics/test_survival_metrics.py::TestConcordanceIndexHarrel -v
pytest tests/test_metrics/test_survival_metrics.py::TestConcordanceIndexHarrel::test_concordanceIndexHarrel_random -v
```

### Markers

The configuration registers `slow`, `integration`, and `edge_case` in `pytest.ini`

```bash
pytest -m slow
pytest -m integration
pytest -m edge_case
pytest -m "not slow"
```

### Coverage

```bash
# Coverage in the terminal and HTML
pytest --cov=bsix --cov-report=term-missing --cov-report=html
```
