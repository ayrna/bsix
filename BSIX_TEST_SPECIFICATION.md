# BSIX Test Suite Specification

## 1. Purpose and scope

This document describes **the tests that currently exist** in `tests/`. The specification is derived from `conftest.py`, each `test_*.py` file, and `pytest.ini`.

The suite checks five areas:

1. data preparation and survival utilities;
2. survival and classification metrics;
3. metric calculation and aggregation;
4. experiment orchestration and result loading;
5. end-to-end pipelines, reproducibility, explainability, and estimator edge cases.

The suite is designed to run from the repository root. The configuration adds `.` and `src` to `pythonpath`, so the tests import the `bsix` package from `src/` and the `execution` package from the root.

## 2. File inventory

```text
tests/
├── __init__.py
├── conftest.py
├── requirements-test.txt
├── test_data/
│   ├── __init__.py
│   ├── test_load_data.py
│   └── test_survival_utils.py
├── test_metrics/
│   ├── __init__.py
│   ├── test_calculate_metrics.py
│   ├── test_classification_metrics.py
│   └── test_survival_metrics.py
├── test_execution/
│   ├── __init__.py
│   └── test_orchestrator.py
├── test_results/
│   ├── __init__.py
│   └── test_load_results.py
├── test_integration/
│   ├── __init__.py
│   ├── test_edge_cases.py
│   └── test_end_to_end.py
└── _results_/
    ├── ffb44bbd-6909-4d70-a321-41910bbd8d54.json
    └── ffb44bbd-6909-4d70-a321-41910bbd8d54.pkl
```

## 3. Pytest configuration

The configuration is in [`pytest.ini`](pytest.ini):

| Option | Value or effect |
|---|---|
| `pythonpath` | `.` and `src` |
| `minversion` | `6.0` |
| `testpaths` | `tests` |
| `python_files` | `test_*.py` |
| `python_functions` | `test_*` |
| `addopts` | `-v --tb=short --strict-markers --disable-warnings -ra` |
| `timeout` | 300 seconds |
| `log_cli` | disabled |
| `log_cli_level` | `INFO` |

The warnings filter ignores `DeprecationWarning` and `PendingDeprecationWarning`.

The markers declared in `pytest.ini` are:

- `slow`: slow tests;
- `integration`: integration tests;
- `edge_case`: edge cases.

## 4. Shared fixtures

All `conftest.py` fixtures are function-scoped, except `setup_pytorch_seed`, which is session-scoped. Generators use seed `42`, unless otherwise specified.

### 4.1 Initialization and cleanup

`setup_pytorch_seed` sets:

- `numpy.random.seed(42)`;
- `torch.manual_seed(42)`;
- the CUDA seed when CUDA is available;
- `torch.backends.cudnn.deterministic = True`;
- `torch.backends.cudnn.benchmark = False`.

`cleanup_after_test` is an autouse fixture. After each test, it clears CUDA memory when a GPU is available.

### 4.2 Structured survival data

The fixture's private helpers generate `X` as a `float32` `numpy.ndarray` and `y` as a structured array:

- basic format: `event` (`bool`) and `time` fields (`float64` in the helper);
- time-varying format: `event`, `time_start`, and `time_stop` fields;
- multi-event format: an array of shape `(n_samples, 2)` with `event` and `time` fields.

In the basic datasets, times are generated as an exponential with scale 100 plus 1. In data that is not fully censored, `event` is generated with a Bernoulli probability of `0.5`. Fully censored variants have all events set to `False`.

Available fixtures:

| Fixture | Size and contents |
|---|---|
| `simple_survival_data` | `(100, 10)` and basic survival |
| `simple_survival_data_timevarying` | `(100, 10)` and `time_start/time_stop` |
| `simple_survival_data_multi_event` | `(100, 10)` and two events |
| `survival_data` | `(5000, 10)` and basic survival |
| `survival_data_timevarying` | `(5000, 10)` and time-varying format |
| `survival_data_multi_event` | `(5000, 10)` and two events |
| `survival_data_all_censored` | `(5000, 10)`, all censored |
| `survival_data_timevarying_all_censored` | time-varying, all censored |
| `survival_data_multi_event_all_censored` | two events, all censored |
| `single_survival_data` | one observation, one event |
| `single_survival_data_time_varying` | one time-varying observation |
| `single_survival_data_multi_event` | one observation and two events |
| `wrong_dtype_survival_data` | `y` as an unstructured numeric array |
| `wrong_dtype_timevarying_survival_data` | unstructured time-varying |
| `wrong_dtype_multi_event_survival_data` | unstructured multi-event |

The three `single_*` fixtures force `event = 1` after creating the data.

### 4.3 DataFrames

Normal DataFrames have 100 rows and these columns:

```text
event, time, age, gender, score, bmi
```

`gender` is categorical (`M`/`F`); the remaining columns are numeric. The fixtures are:

- `dataframe_mixed_types`: numeric and categorical types;
- `dataframe_time_varying`: `event`, `time_start`, `time_stop`, and features;
- `dataframe_multi_event`: `event1`, `time1`, `event2`, `time2`, and features;
- `dataframe_with_nans`: approximately 10% `NaN` values;
- `dataframe_with_nans_time_varying`;
- `dataframe_with_nans_multi_event`;
- `dataframe_with_invalid_times`: times between -100 and 100;
- `dataframe_empty`: `event`, `time`, and `feature1` columns, with no rows;
- `dataframe_single_row`;
- `dataframe_single_row_time_varying`;
- `dataframe_single_row_multi_event`.

### 4.4 Predictions

Prediction fixtures use 5000 observations except for the `single_*` variants:

- `risk_scores`: random normal `float32` values;
- `binary_predictions`: uniform probabilities in `[0, 1]`;
- `survival_functions`: one `StepFunction` array per observation;
- `random_predictions`: uniform values in `[0, 1]`;
- `perfect_classification_predictions`: `y["event"]`;
- `inverse_classification_predictions`: negation of `y["event"]`;
- `perfect_survival_predictions`: `-y["time"]`;
- `inverse_survival_predictions`: `y["time"]`;
- `perfect_survival_functions`: 1 before the observed time and 0 afterward;
- `inverse_survival_functions`: 0 before the observed time and 1 afterward;
- `random_survival_functions`: alternating perfect and inverse functions.

For classification metrics, random predictions are thresholded at `0.5` before calling the metric functions.

### 4.5 Configurations

`basic_execution_config` temporarily replaces `execution.orchestrator.CONFIG` with a `SimpleNamespace` containing, among other values, `pipeline="internal_cv"`, the `CoxRegression` estimator, the `metabric.h5` dataset, 30 seeds, `test_size=0.25`, `validation_size=0.2`, `n_jobs=5`, and SLURM resource configuration.

`basic_experiment_config` contains:

```python
{
    "data_dir": "bsix.datasets",
    "results_dir": "./results",
    "dataset": "metabric.h5",
    "test_size": 0.25,
    "validation_size": 0.2,
    "seed": 42,
    "estimator_name": "CoxRegression",
    "n_iter": 5,
    "n_jobs": -1,
    "interactive": True,
}
```

The invalid variants change `estimator_name` to `"InvalidEstimator"` or `dataset` to `"InvalidDataset"`, respectively.

## 5. Data tests (`test_data/`)

### 5.1 Data preparation: `test_load_data.py`

The module tests `bsix.utils.load_data._prepare_data` and `_transformTrainValidationTest`.

#### Splitting, cleaning, and types

- **DATA-001**: the default split with `test_size=0.2` and `validation_size=0.2` produces approximately 60% train, 20% validation, and 20% test. The implemented tolerances are 55–65%, 15–25%, and 15–25%. Repeating with `seed=42` produces the same `X_train`, `X_val`, and `X_test` arrays.
- **DATA-002**: categorical columns are converted to `float32` numeric arrays; the input arrays contain no `NaN`.
- **DATA-003**: rows with `NaN` are removed; the clean total is smaller than the original size and the remaining values are finite.
- **DATA-004**: non-positive times are filtered. After `_transformTrainValidationTest`, all checked times are greater than zero.
- **DATA-005**: the presence of `time_start` and `time_stop` is processed without error and produces non-empty training data.
- **DATA-008**: the transformation preserves `X` as `float32` and produces `y` with a structured dtype.
- **DATA-009**: feature names are returned as a list or tuple, do not contain `event` or `time`, and their number is compatible with the dimension of `X`.

#### Grouping, multi-event, and output shape

- **DATA-006**: a DataFrame with `event1/time1` and `event2/time2` is detected as multi-event and transformed without error.
- **DATA-007**: if `identifier` exists, preparation uses the grouped-split path; the test checks that all three splits are non-empty and preserve the total number of rows.
- The output of `_prepare_data` is a ten-element tuple, in the order observed by the tests:

  ```text
  X_train, y_train, X_validation, y_validation, X_test, y_test, ..., feature_names
  ```

- `X` and `y` have the same number of observations in each split, and the three `X` matrices share the same feature dimension.
- The following `(test, validation)` proportions are tested: `(0.1, 0.1)`, `(0.2, 0.3)`, `(0.3, 0.2)`, and `(0.25, 0.25)`. Each ratio is compared with an absolute tolerance of 0.05.

#### Edge cases

`dataframe_empty` and `dataframe_single_row` must raise `ValueError`.

### 5.2 Survival utilities: `test_survival_utils.py`

`bsix.utils.survival_utils.getTau` and `getTimes` are tested:

- `getTau` returns a `float` or `numpy.floating`, greater than zero and no larger than the maximum train/test time used by the test.
- `getTau` does not raise an error with `survival_data_all_censored` and returns a positive value.
- `getTimes(y, num_division=5)` returns a `numpy.ndarray` with five points within the time range.
- The result of `getTimes` is sorted in non-decreasing order.
- `num_division` values of 2, 5, 10, 20, and 50 are tested.
- With a single observation, `getTimes` returns at least one point.

## 6. Metric tests (`test_metrics/`)

### 6.1 Survival metrics: `test_survival_metrics.py`

The following functions from `bsix.utils.survival_metrics` are tested:

- `concordanceIndexHarrel`;
- `concordanceIndexIPCW`;
- `cumulativeDinamicAUC`;
- `integratedBrierScore`.

For Harrell's C-Index, IPCW C-Index, and dynamic AUC, the test requires:

- scalar `float`/`numpy.floating` result;
- random prediction approximately 0.5 (`abs=0.05`);
- perfect prediction approximately 1 (`abs=0.001`);
- inverse prediction approximately 0 (`abs=0.001`);
- result in `[0, 1]`;
- consistency when repeating the same call;
- approximate invariance when scores are multiplied by 5, as expected from the implemented rank checks.

For `integratedBrierScore`, the test requires:

- random result approximately 0.5;
- perfect result approximately 0;
- inverse result approximately 1;
- result in `[0, 1]`;
- consistency when repeating the call.

Edge cases:

- empty arrays: all four functions must raise `ValueError`;
- fully censored data: all four functions must raise `ValueError`;
- a copy with rounded times is created to simulate ties, but the current code calculates metrics with `y_true`, not that copy (`y_tied`). Therefore, this test does not actually verify behavior on tied times; it only verifies that the base case produces finite, in-range values.

### 6.2 Classification metrics: `test_classification_metrics.py`

The functions are tested:

```text
ccr, amae, gm, mae, mmae, recall, ms, mze, tkendall, wkappa, spearman
```

Each metric is tested with random, perfect, and inverse predictions. The reference expectations are:

| Metric | Random | Perfect | Inverse | Checked range |
|---|---:|---:|---:|---|
| `ccr` | ≈ 0.5 | ≈ 1 | ≈ 0 | [0, 1] |
| `amae` | ≈ 0.5 | ≈ 0 | ≈ 1 | [0, infinity) |
| `gm` | ≈ 0.5 | ≈ 1 | ≈ 0 | [0, 1] |
| `mae` | ≈ 0.5 | ≈ 0 | ≈ 1 | [0, infinity) |
| `mmae` | ≈ 0.5 | ≈ 0 | ≈ 1 | [0, infinity) |
| `recall` | ≈ 0.5 | ≈ 1 | ≈ 0 | [0, 1] |
| `ms` | ≈ 0.5 | ≈ 1 | ≈ 0 | [0, 1] |
| `mze` | ≈ 0.5 | ≈ 0 | ≈ 1 | [0, 1] |
| `tkendall` | ≈ 0 | ≈ 1 | ≈ -1 | [-1, 1] |
| `wkappa` | ≈ 0 | ≈ 1 | ≈ -1 | [-1, 1] |
| `spearman` | ≈ 0 | ≈ 1 | ≈ -1 | [-1, 1] |

Random comparisons use `abs=0.05`, and perfect/inverse comparisons use `abs=0.001`.

Empty arrays must raise `ValueError` in all eleven functions. Repeating a metric with the same inputs is also checked to produce the same result.

### 6.3 Aggregation: `test_calculate_metrics.py`

`format_predictions` and `compute_metrics` from `bsix.utils` are tested.

`format_predictions`:

- for a NumPy matrix/vector, returns a dictionary with the `survival` key;
- for a `(pred_survival, pred_binary)` tuple, returns the `survival` and `binary` keys.

`compute_metrics`:

- with risk scores and no survival function, returns a non-empty dictionary;
- with `survival_function`, includes `Brier Score`;
- with `(survival, binary)` predictions and a survival function, includes at least:

  ```text
  C-Index Harrel, C-Index IPCW, Cumulative Dinamic AUC, Brier Score, MAE, AMAE, MS, CCR, RECALL0, RECALL1
  ```

- for two-event data, metrics are prefixed with `P0  ` and `P1  `; the ten equivalent keys for each event are checked;
- in the normal case, all returned metrics are finite;
- empty arrays, a single observation, or a fully censored evaluation set must raise `ValueError`;
- when survival scores are multiplied by 5, the same keys are retained and the selected metrics are approximately equal (`abs=0.001`).

## 7. Orchestration (`test_execution/test_orchestrator.py`)

The `execution` package is tested:

- `generate_internal_cv_experiments_configs()` returns a dictionary; when present, its keys are two-element tuples `(estimator, dataset)` and its values are lists;
- with the fixture configuration, at least one configuration exists in the first entry;
- `get_executor_params(estimator="CoxRegression", dataset="metabric")` returns a dictionary;
- `execution.workers` contains `run_experiment`, `run_experiment_with_best_config`, `run_aggregator`, and `run_results_collector`;
- `execution.flows.survival_flow.run_survival_flow` is callable and has at least one parameter in its signature;
- `execution.configs` imports successfully;
- `execution.config.default` imports successfully;
- the basic flow with `basic_experiment_config` is marked `integration` and runs without exception;
- a nonexistent estimator must raise `KeyError`;
- a nonexistent dataset must raise `ValueError`.

## 8. Results (`test_results/test_load_results.py`)

The tests use `./tests/_results_` and check `get_results` and `save_results` from `bsix.utils.load_results`.

`get_results`:

- without filters, returns at least one result;
- each loaded result exposes `config` and `data_`;
- filters by `estimator_name="DeepSurv"`;
- filters by `dataset="breastcancer.arff"`;
- filters by `seed=3`;
- applies all three filters simultaneously;
- returns an empty list for non-matching values;
- returns an empty list with `result_folder="InvalidFolderPath"`.

The configuration of a result must contain the keys:

```text
estimator_name, dataset, random_state
```

`result.data_` must expose:

```text
targets, predictions, train_targets, train_predictions, best_model, best_params
```

`save_results` must create an Excel file at `./tests/_results_/report.xlsx`. The test checks that it exists and deletes it at the end.

## 9. End-to-end integration

### 9.1 Main pipelines: `test_end_to_end.py`

For each of the following estimators, a data-loading, training, validation/test prediction, and metric-calculation pipeline is tested:

| Estimator | Fixture dataset | Variants |
|---|---|---|
| `AcceleratedFailureTime` | `dataframe_mixed_types` | pipeline and cross-validation |
| `CoxRegression` | `dataframe_mixed_types` | pipeline and cross-validation |
| `CoxRegressionWithTimeVarying` | `dataframe_time_varying` | pipeline and cross-validation |
| `DeepHit` | `dataframe_mixed_types` | pipeline and cross-validation |
| `DeepMultiTask` | `dataframe_multi_event` | pipeline and cross-validation |
| `DeepSurv` | `dataframe_mixed_types` | pipeline and cross-validation |
| `DeepTimeVarying` | `dataframe_time_varying` | pipeline and cross-validation |
| `RandomSurvForest` | `dataframe_mixed_types` | pipeline and cross-validation |
| `SurvivalTabPFN` | `dataframe_mixed_types` | pipeline and cross-validation |
| `SurvTree` | `dataframe_mixed_types` | pipeline and cross-validation |

All these pipelines check that train and test are non-empty, predictions have shape `(len(X_split),)`, and validation and test metrics are non-empty dictionaries.

The cross-validation variants combine train and validation using `sklearn.model_selection.PredefinedSplit`, construct the estimator with `get_estimator`, use `seed=42`, `n_jobs=5`, and `n_iter=30`, and reevaluate validation and test.

### 9.2 Preprocessing variations

Each of the ten estimator pipelines is repeated with fixtures containing `NaN`:

- `dataframe_with_nans` for normal formats;
- `dataframe_with_nans_time_varying` for `CoxRegressionWithTimeVarying` and `DeepTimeVarying`;
- `dataframe_with_nans_multi_event` for `DeepMultiTask`.

The checked objective is that each pipeline can run with this preprocessed data.

### 9.3 Reproducibility

Each of the ten estimators has a determinism test with the corresponding 100-observation fixture. Seeds are fixed, and predictions or results from repeated executions are compared according to the specific test. `DeepMultiTask` uses `simple_survival_data_multi_event`; time-varying models use `simple_survival_data_timevarying`.

### 9.4 Explainability

The SHAP explainability path is invoked for the ten estimators, and the result of `calculate_xai` is checked with:

```text
AcceleratedFailureTime, CoxRegression, CoxRegressionWithTimeVarying, DeepHit, DeepMultiTask, DeepSurv, DeepTimeVarying, RandomSurvForest, SurvivalTabPFN, SurvTree
```

The data used are the corresponding normal, time-varying, or multi-event DataFrames.

## 10. Integration edge cases (`test_edge_cases.py`)

### 10.1 Empty and single-observation data

All ten estimators are trained with empty arrays and are expected to raise `ValueError`. The ten estimators are also tested with a single observation and are expected to raise `ValueError`.

### 10.2 Special values

Data is tested with:

- `NaN` in features;
- `NaN` in targets;
- positive or negative infinity in features;
- infinity in targets.

The cases are applied to the relevant estimators and formats and verify `ValueError` when the model cannot accept the input.

### 10.3 Target dtypes and shape

The unstructured arrays from the `wrong_dtype_*` fixtures are tested. Structured targets with basic, time-varying, and multi-event formats are also tested. Incompatible inputs must be rejected with `ValueError` where specified by each test.

### 10.4 Seeds

Valid, `None`, negative, string, and very large seeds are tested on the estimators covered by `TestSeedHandling`. Invalid values must raise `ValueError`; accepted cases must create a non-null model.

### 10.5 Boundary times

The `TestBoundaryTimes` class covers zero, negative, very small, and large times. Invalid times must raise `ValueError`; accepted large times must allow the model to be created or trained, depending on the tested case.

### 10.6 Complete censoring

The `TestEventCensoring` class runs pipelines with `survival_data_all_censored`, `survival_data_timevarying_all_censored`, or `survival_data_multi_event_all_censored` for the ten estimators. Each case is marked `integration` and checks the behavior defined in the test, usually `ValueError` when the algorithm requires observed events.

### 10.7 Imports

`test_all_core_imports` checks that the core model and utility components required by the suite can be imported.

## 11. Reproducible execution

Commands must be run from the repository root:

```bash
python -m pip install -e .
python -m pip install -r tests/requirements-test.txt
pytest
```

Diagnostic commands:

```bash
pytest --collect-only -q
pytest -q
pytest -vv -s
pytest -x
pytest --maxfail=3
pytest --durations=10
```

Selection by area:

```bash
pytest tests/test_data/ -v
pytest tests/test_metrics/ -v
pytest tests/test_execution/ -v
pytest tests/test_results/ -v
pytest tests/test_integration/ -v
```

Selection by file, class, or method:

```bash
pytest tests/test_metrics/test_survival_metrics.py -v
pytest tests/test_metrics/test_survival_metrics.py::TestConcordanceIndexHarrel -v
pytest tests/test_metrics/test_survival_metrics.py::TestConcordanceIndexHarrel::test_concordanceIndexHarrel_random -v
```

Selection by marker or name:

```bash
pytest -m integration
pytest -m edge_case
pytest -m "not slow"
pytest -k "cox"
pytest -k "not deep"
```

Coverage and parallel execution require the plugins to be installed:

```bash
pytest --cov=bsix --cov-report=term-missing --cov-report=html
pytest -n 4
```

To debug a failure:

```bash
pytest --pdb tests/test_data/test_load_data.py::TestPrepareDataEdgeCases::test_empty_dataframe
pytest -l -vv -s tests/test_integration/
```

## 12. Test dependencies

[`tests/requirements-test.txt`](tests/requirements-test.txt) declares:

- `pytest`, `pytest-cov`, `pytest-timeout`, `pytest-xdist`;
- `numpy`, `pandas`, `scipy`;
- `scikit-learn`, `scikit-survival`, `lifelines`;
- `torch`, `torchtuples`;
- `shap`;
- `h5py`, `matplotlib`, `seaborn`;
- optional quality tools such as `black`, `flake8`, and `pytest-pep8`.

## 13. Interpretation criteria

The suite distinguishes between:

- **output contracts**: types, shapes, keys, ranges, and attribute presence;
- **expected errors**: primarily `ValueError` for empty, incompatible, or event-free inputs, and `KeyError` for invalid estimators in the workflow;
- **approximate statistical properties**: perfect, inverse, and random predictions for the metrics;
- **integration**: the ability to complete the data → model → prediction → metrics/explainability path.
