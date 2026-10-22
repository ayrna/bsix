"""
Pytest configuration and shared fixtures for BSIX test suite.
All fixtures follow the specifications in BSIX_TEST_SPECIFICATION.md
"""

import pytest
import numpy as np
import pandas as pd
import torch

from typing import Tuple, Dict, Any
from types import SimpleNamespace
from unittest.mock import patch


# ==================== NUMPY/PANDAS FIXTURES ====================

@pytest.fixture(scope="session")
def setup_pytorch_seed():
    """Fijar seed para reproducibilidad en PyTorch"""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _create_survival_data(n_samples, n_features, seed, all_censored=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to create a basic survival dataset
    """
    np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    times = np.random.exponential(100, n_samples).astype(np.float32) + 1

    if all_censored == False:
        events = np.random.binomial(1, 0.5, n_samples).astype(bool)
    else:
        events = np.zeros(n_samples, dtype=bool)
    
    y = np.zeros(n_samples, dtype=[('event', '?'), ('time', 'f8')])
    y['event'] = events
    y['time'] = times
    
    return X, y

def _create_survival_data_time_varying(n_samples, n_features, seed, all_censored=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to create a time_varying survival dataset
    """
    np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    time_start = np.random.uniform(0, 50, n_samples).astype(np.float32)
    time_stop = time_start + np.random.exponential(50, n_samples).astype(np.float32)

    if all_censored == False:
        events = np.random.binomial(1, 0.5, n_samples).astype(bool)
    else:
        events = np.zeros(n_samples, dtype=bool)
    
    y = np.zeros(n_samples, dtype=[
        ('event', '?'),
        ('time_start', 'f8'), 
        ('time_stop', 'f8'), 
    ])
    y['time_start'] = time_start
    y['time_stop'] = time_stop
    y['event'] = events
    
    return X, y

def _create_survival_data_multi_event(n_samples, n_features, seed, all_censored=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to create a multi_event survival dataset
    """
    np.random.seed(seed)
    
    n_events = 2
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Two competing events
    times = np.random.exponential(100, n_samples).astype(np.float32) + 1

    if all_censored == False:
        events = np.random.binomial(1, 0.5, n_samples).astype(bool)
    else:
        events = np.zeros(n_samples, dtype=bool)

    y = np.zeros([n_samples, n_events], dtype=[('event', '?'), ('time', 'f8')])
    for i in range(n_events):
        y[:, i]['event'] = events
        y[:, i]['time'] = times
    
    return X, y

@pytest.fixture
def simple_survival_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple dataset (100 samples, 10 features)
    
    Returns:
        Tuple[X, y] where:
        - X: shape (100, 10), dtype float32
        - y: structured array with ('event', bool), ('time', float32)
    """

    n_samples, n_features = 100, 10
    seed = 42
    
    return _create_survival_data(n_samples, n_features, seed)

@pytest.fixture
def simple_survival_data_timevarying() -> Tuple[np.ndarray, np.ndarray]:
    """
    Time-varying survival data with time_start and time_stop
    """

    n_samples, n_features = 100, 10
    seed = 42
    
    return _create_survival_data_time_varying(n_samples, n_features, seed)


@pytest.fixture
def simple_survival_data_multi_event() -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-event survival data with competing events
    """
    n_samples, n_features = 100, 10
    seed = 42
    
    return _create_survival_data_multi_event(n_samples, n_features, seed)

@pytest.fixture
def survival_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset (5000 samples, 10 features)
    
    Returns:
        Tuple[X, y] where:
        - X: shape (5000, 10), dtype float32
        - y: structured array with ('event', bool), ('time', float32)
    """

    n_samples, n_features = 5000, 10
    seed = 42
    
    return _create_survival_data(n_samples, n_features, seed)


@pytest.fixture
def survival_data_timevarying() -> Tuple[np.ndarray, np.ndarray]:
    """
    Time-varying survival data with time_start and time_stop
    """

    n_samples, n_features = 5000, 10
    seed = 42
    
    return _create_survival_data_time_varying(n_samples, n_features, seed)


@pytest.fixture
def survival_data_multi_event() -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-event survival data with competing events
    """
    n_samples, n_features = 5000, 10
    seed = 42
    
    return _create_survival_data_multi_event(n_samples, n_features, seed)


@pytest.fixture
def survival_data_all_censored() -> Tuple[np.ndarray, np.ndarray]:
    """
    Edge case: all events are censored (event=0)
    """
    n_samples, n_features = 5000, 10
    seed = 42
    
    return _create_survival_data(n_samples, n_features, seed, all_censored=True)

@pytest.fixture
def survival_data_timevarying_all_censored() -> Tuple[np.ndarray, np.ndarray]:
    """
    Edge case: all events are censored (event=0)
    """
    n_samples, n_features = 5000, 10
    seed = 42
    
    return _create_survival_data_time_varying(n_samples, n_features, seed, all_censored=True)

@pytest.fixture
def survival_data_multi_event_all_censored() -> Tuple[np.ndarray, np.ndarray]:
    """
    Edge case: all events are censored (event=0)
    """
    n_samples, n_features = 5000, 10
    seed = 42
    
    return _create_survival_data_multi_event(n_samples, n_features, seed, all_censored=True)

@pytest.fixture
def single_survival_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset simple para testing (1 samples, 10 features)
    
    Returns:
        Tuple[X, y] where:
        - X: shape (1, 10), dtype float32
        - y: structured array with ('event', bool), ('time', float32)
    """

    n_samples, n_features = 1, 10
    seed = 42

    X, y = _create_survival_data(n_samples, n_features, seed)
    y["event"] = 1

    return X, y

@pytest.fixture
def single_survival_data_time_varying() -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset simple para testing (1 samples, 10 features)
    
    Returns:
        Tuple[X, y] where:
        - X: shape (1, 10), dtype float32
        - y: structured array with ('event', bool), ('time', float32)
    """

    n_samples, n_features = 1, 10
    seed = 42

    X, y = _create_survival_data_time_varying(n_samples, n_features, seed)
    y["event"] = 1
    
    return X, y

@pytest.fixture
def single_survival_data_multi_event() -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset simple para testing (1 samples, 10 features)
    
    Returns:
        Tuple[X, y] where:
        - X: shape (1, 10), dtype float32
        - y: structured array with ('event', bool), ('time', float32)
    """

    n_samples, n_features = 1, 10
    seed = 42

    X, y = _create_survival_data_multi_event(n_samples, n_features, seed)
    y["event"] = 1
    
    return X, y

@pytest.fixture
def wrong_dtype_survival_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset with wrong dtype para testing (100 samples, 10 features)
    
    Returns:
        Tuple[X, y] where:
        - X: shape (100, 10), dtype float32
        - y: shape (100, 2), dtype float32 (should be structured array)
    """
    np.random.seed(42)
    n_samples, n_features = 100, 10
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    events = np.random.binomial(1, 0.6, n_samples).astype(bool)
    times = np.random.exponential(100, n_samples).astype(np.float32) + 1
    
    y = np.array([events, times]).T.astype(np.float32)  # Wrong dtype: regular array instead of structured
    
    return X, y


@pytest.fixture
def wrong_dtype_timevarying_survival_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset with wrong dtype para testing (100 samples, 10 features)
    
    Returns:
        Tuple[X, y] where:
        - X: shape (100, 10), dtype float32
        - y: shape (100, 3), dtype float32 (should be structured array)
    """
    np.random.seed(42)
    n_samples, n_features = 100, 10
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    time_start = np.random.uniform(0, 50, n_samples).astype(np.float32)
    time_stop = time_start + np.random.exponential(50, n_samples).astype(np.float32)
    events = np.random.binomial(1, 0.5, n_samples).astype(bool)
    
    y = np.array([events, time_start, time_stop]).T.astype(np.float32)  # Wrong dtype: regular array instead of structured
    
    return X, y


@pytest.fixture
def wrong_dtype_multi_event_survival_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset with wrong dtype para testing (100 samples, 10 features)
    
    Returns:
        Tuple[X, y] where:
        - X: shape (100, 10), dtype float32
        - y: shape (100, 2), dtype float32 (should be structured array)
    """
    np.random.seed(42)
    n_samples, n_features = 100, 10
    n_events = 2
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Two competing events
    events = np.random.binomial(1, 0.4, n_samples).astype(bool)
    times = np.random.exponential(100, n_samples).astype(np.float32) + 1

    y = np.zeros([n_samples, n_events, 2], dtype=np.float32)
    for i in range(n_events):
        y[:, i, 0] = events # Wrong dtype: regular array instead of structured
        y[:, i, 1] = times  # Wrong dtype: regular array instead of structured
    
    return X, y


# ==================== DATAFRAME FIXTURES ====================

def _create_dataframe(n, seed) -> pd.DataFrame:
    """
    Helper function to create a basic DataFrame for testing
    """
    np.random.seed(seed)

    df = pd.DataFrame({
        'event': np.random.randint(0, 2, n),
        'time': np.round(np.random.uniform(10.0, 100.0, n), 1),
        'age': np.random.randint(30, 75, n),
        'gender': np.random.choice(['M', 'F'], n),
        'score': np.round(np.random.uniform(0.05, 0.95, n), 2),
        'bmi': np.round(np.random.uniform(18.0, 35.0, n), 1)
    })
    return df

def _create_dataframe_time_varying(n, seed) -> pd.DataFrame:
    """
    Helper function to create a time_varying DataFrame for testing
    """
    np.random.seed(seed)

    t1 = np.round(np.random.uniform(10.0, 100.0, n), 1)
    t2 = np.round(np.random.uniform(10.0, 100.0, n), 1)

    time_start = np.minimum(t1, t2)
    time_stop = np.maximum(t1, t2)

    drafts = time_start == time_stop
    time_stop[drafts] += 1

    df = pd.DataFrame({
        'event': np.random.randint(0, 2, n),
        'time_start': time_start,
        'time_stop': time_stop,
        'age': np.random.randint(30, 75, n),
        'gender': np.random.choice(['M', 'F'], n),
        'score': np.round(np.random.uniform(0.05, 0.95, n), 2),
        'bmi': np.round(np.random.uniform(18.0, 35.0, n), 1)
    })
    return df

def _create_dataframe_multi_event(n, seed) -> pd.DataFrame:
    """
    Helper function to create a multi_event DataFrame for testing
    """
    np.random.seed(seed)

    df = pd.DataFrame({
        'event1': np.random.randint(0, 2, n),
        'time1': np.round(np.random.uniform(10.0, 100.0, n), 1),
        'event2': np.random.randint(0, 2, n),
        'time2': np.round(np.random.uniform(10.0, 100.0, n), 1),
        'age': np.random.randint(30, 75, n),
        'gender': np.random.choice(['M', 'F'], n),
        'score': np.round(np.random.uniform(0.05, 0.95, n), 2),
        'bmi': np.round(np.random.uniform(18.0, 35.0, n), 1)
    })
    return df

def _insert_nans(df, seed) -> pd.DataFrame:
    """
    Helper function to insert NaN values into a DataFrame
    """
    np.random.seed(seed)

    n = len(df)
    num_nan = int(n * 0.10)
    index_nan = np.random.choice(df.index, size=num_nan, replace=False)
    col_idx = np.random.randint(0, len(df.columns), size=3)
    col_nan = df.columns[col_idx]

    for i, idx in enumerate(index_nan):
        col = col_nan[i % len(col_nan)]
        df.loc[idx, col] = np.nan

    return df

@pytest.fixture
def dataframe_mixed_types() -> pd.DataFrame:
    """
    DataFrame with mixed numeric and categorical features
    """

    n = 100
    seed = 42

    return _create_dataframe(n, seed)


@pytest.fixture
def dataframe_time_varying() -> pd.DataFrame:
    """
    DataFrame with time-varying format
    """
    n = 100
    seed = 42

    return _create_dataframe_time_varying(n, seed)


@pytest.fixture
def dataframe_multi_event() -> pd.DataFrame:
    """
    DataFrame with multi-event format
    """
    n = 100
    seed = 42

    return _create_dataframe_multi_event(n, seed)


@pytest.fixture
def dataframe_with_nans() -> pd.DataFrame:
    """
    DataFrame containing NaN values (~10%)
    """
    n = 100
    seed = 42
    
    df = _create_dataframe(n, seed)
    df = _insert_nans(df, seed)

    return df

@pytest.fixture
def dataframe_with_nans_time_varying() -> pd.DataFrame:
    """
    DataFrame containing NaN values (~10%)
    """
    n = 100
    seed = 42
    
    df = _create_dataframe_time_varying(n, seed)
    df = _insert_nans(df, seed)

    return df

@pytest.fixture
def dataframe_with_nans_multi_event() -> pd.DataFrame:
    """
    DataFrame containing NaN values (~10%)
    """
    n = 100
    seed = 42
    
    df = _create_dataframe_multi_event(n, seed)
    df = _insert_nans(df, seed)

    return df

@pytest.fixture
def dataframe_with_invalid_times() -> pd.DataFrame:
    """
    DataFrame with negative/zero times (should be filtered)
    """
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'event': np.random.randint(0, 2, n),
        'time': np.round(np.random.uniform(-100.0, 100.0, n), 1),
        'age': np.random.randint(30, 75, n),
        'gender': np.random.choice(['M', 'F'], n),
        'score': np.round(np.random.uniform(0.05, 0.95, n), 2),
        'bmi': np.round(np.random.uniform(18.0, 35.0, n), 1)
    })

    return df


@pytest.fixture
def dataframe_empty() -> pd.DataFrame:
    """
    Empty DataFrame edge case
    """
    return pd.DataFrame({
        'event': [],
        'time': [],
        'feature1': []
    })


@pytest.fixture
def dataframe_single_row() -> pd.DataFrame:
    """
    Single row DataFrame edge case
    """
    n = 1
    seed = 42
    
    return _create_dataframe(n, seed)

@pytest.fixture
def dataframe_single_row_time_varying() -> pd.DataFrame:
    """
    Single row DataFrame edge case
    """
    n = 1
    seed = 42
    
    return _create_dataframe_time_varying(n, seed)

@pytest.fixture
def dataframe_single_row_multi_event() -> pd.DataFrame:
    """
    Single row DataFrame edge case
    """
    n = 1
    seed = 42
    
    return _create_dataframe_multi_event(n, seed)

# ==================== MOCK FIXTURES ====================

@pytest.fixture
def basic_execution_config():
    """
    Base execution configuration
    """
    mock_conf = SimpleNamespace(
        experiment_name = "experiments",
        pipeline = "internal_cv",
        flow = "survival_flow",
        estimators=["CoxRegression"],
        data_dir="bsix.datasets",
        datasets=["metabric.h5"],
        seeds = 30,
        validation_size = 0.2,
        test_size = 0.25,
        n_jobs = 5,
        dry_run = False,
        cpus = 2,
        memory = 10,
        gpus = 2,
        nice = 0,
        gpu_type = "high_vram",
        gpu_legacy = True,
        max_time = "05:00:00",
        max_concurrent_jobs = 5,
        memory_override = {},
        gpu_type_override = {},
        batch_size_override = {},
        max_time_override = {},
        results_dir = "./results",
        logs_output_dir = "./logs",
        best_configs_file = "best_configs.pkl",
        prepared_results_dir = "./prepared_results",
        prepared_results_appendix = "survival",
        collect_methods = None,
        collect_datasets = None,
        collect_seeds = None,
        config_columns_to_include = ["dataset","estimator_name","random_state",],
        best_params_columns_to_include = None,
        collect_train = True,
        collect_val = False,
        skip_zip = False,
        collect_n_jobs = 4,
        extra=0
    )

    with patch('execution.orchestrator.CONFIG', mock_conf):
        yield mock_conf


# ==================== PREDICTION FIXTURES ====================

def _create_risk_scores(n, seed) -> np.ndarray:
    """
    Helper function to create risk_scores
    """
    np.random.seed(seed)
    return np.random.randn(n).astype(np.float32)


def _create_binary_predictions(n, seed) -> np.ndarray:
    """
    Helper function to create binary_predictions
    """
    np.random.seed(seed)
    return np.random.uniform(0, 1, n).astype(np.float32)


def _create_survival_functions(n):
    """
    Helper function to create a survival_function
    """
    from bsix.models.metodologies.utils import StepFunction

    times = np.linspace(1, 300, n)
    risks = np.random.uniform(low=0.005, high=0.05, size=n).reshape(n, 1)
    sf = np.exp(-risks * times)
    S = np.array([StepFunction(X=times, y=individual_survival, is_survival=True) for individual_survival in sf])
    return S

@pytest.fixture
def risk_scores() -> np.ndarray:
    """
    Mock risk scores for n_samples=5000
    """
    n = 5000
    seed = 42

    return _create_risk_scores(n, seed)

@pytest.fixture
def binary_predictions() -> np.ndarray:
    """
    Mock binary predictions (probabilities) in [0, 1]
    """
    n = 5000
    seed = 42

    return _create_binary_predictions(n, seed)


@pytest.fixture
def survival_functions():
    """
    Mock survival functions
    """

    n = 5000

    return _create_survival_functions(n)

@pytest.fixture
def single_risk_score() -> np.ndarray:
    """
    Mock risk scores for n_samples=1
    """
    n = 1
    seed = 42

    return _create_risk_scores(n, seed)



@pytest.fixture
def single_binary_prediction() -> np.ndarray:
    """
    Mock binary predictions (probabilities) in [0, 1]
    """
    n = 1
    seed = 42

    return _create_binary_predictions(n, seed)


@pytest.fixture
def single_survival_function():
    """
    Mock survival functions
    """

    n = 1

    return _create_survival_functions(n)


# ==================== METRICS FIXTURES ====================

@pytest.fixture
def perfect_classification_predictions(survival_data) -> np.ndarray:
    """
    Perfect predictions classification
    """
    _, y = survival_data
    return (y['event']).astype(np.float32)

@pytest.fixture
def perfect_survival_predictions(survival_data) -> np.ndarray:
    """
    Perfect predictions survival
    """
    _, y = survival_data
    return -y['time'].astype(np.float32)

@pytest.fixture
def inverse_classification_predictions(survival_data) -> np.ndarray:
    """
    Inverse predictions classification
    """
    _, y = survival_data
    return (~y['event']).astype(np.float32)

@pytest.fixture
def inverse_survival_predictions(survival_data) -> np.ndarray:
    """
    Inverse predictions survival
    """
    _, y = survival_data
    return y['time'].astype(np.float32)

@pytest.fixture
def random_predictions() -> np.ndarray:
    """
    Random predictions
    """
    np.random.seed(42)
    return np.random.rand(5000).astype(np.float32)

@pytest.fixture
def perfect_survival_functions(survival_data):
    """
    Perfect survival functions.
    """
    from bsix.models.metodologies.utils import StepFunction

    _, y = survival_data

    time_grid = np.linspace(1, 300, 5000)
        
    S = []
    for item in y:
        time = item["time"]  
        
        sf = np.where(time_grid < time, 1.0, 0.0)
        
        S.append(StepFunction(X=time_grid, y=sf, is_survival=True))
        
    return np.array(S)


@pytest.fixture
def inverse_survival_functions(survival_data):
    """
    Inverse survival functions.
    """
    from bsix.models.metodologies.utils import StepFunction

    _, y = survival_data

    time_grid = np.linspace(1, 300, 5000)
        
    S = []
    for item in y:
        time = item["time"]  
        
        sf = np.where(time_grid < time, 0.0, 1.0)
        
        S.append(StepFunction(X=time_grid, y=sf, is_survival=True))
        
    return np.array(S)


@pytest.fixture
def random_survival_functions(survival_data):
    """
    Random survival functions.
    """
    from bsix.models.metodologies.utils import StepFunction

    _, y = survival_data
    
    time_grid = np.linspace(1, 300, 5000)
        
    S = []
    for i, item in enumerate(y):
        time = item["time"]  
        
        if i % 2 == 0:
            sf = np.where(time_grid < time, 1.0, 0.0)
        else:
            sf = np.where(time_grid < time, 0.0, 1.0)
            
        S.append(StepFunction(X=time_grid, y=sf, is_survival=True))
        
    return np.array(S)


# ==================== CONFIGURATION FIXTURES ====================

@pytest.fixture
def basic_experiment_config() -> Dict[str, Any]:
    """
    Default experiment configuration
    """
    return {
        "data_dir": "bsix.datasets",
        "results_dir": "./results",
        "dataset": "metabric.h5",
        "test_size": 0.25,
        "validation_size": 0.2,
        "seed": 42,
        "estimator_name": "CoxRegression",
        "n_iter": 5,
        "n_jobs": -1,
        "interactive": True
    }

@pytest.fixture
def invalid_estimator_config() -> Dict[str, Any]:
    """
    Default experiment configuration
    """
    return {
        "data_dir": "bsix.datasets",
        "results_dir": "./results",
        "dataset": "metabric.h5",
        "test_size": 0.25,
        "validation_size": 0.2,
        "seed": 42,
        "estimator_name": "InvalidEstimator",
        "n_iter": 5,
        "n_jobs": -1,
        "interactive": True
    }

@pytest.fixture
def invalid_dataset_config() -> Dict[str, Any]:
    """
    Default experiment configuration
    """
    return {
        "data_dir": "bsix.datasets",
        "results_dir": "./results",
        "dataset": "InvalidDataset",
        "test_size": 0.25,
        "validation_size": 0.2,
        "seed": 42,
        "estimator_name": "CoxRegression",
        "n_iter": 5,
        "n_jobs": -1,
        "interactive": True
    }


# ==================== UTILITY FIXTURES ====================

#@pytest.fixture
#def times_grid() -> np.ndarray:
#    """
#    Grid of time points for evaluation
#    """
#    return np.linspace(1, 300, 50, dtype=np.float32)


#@pytest.fixture
#def feature_names() -> list:
#    """
#    List of feature names
#    """
#    return [f'feature_{i}' for i in range(10)]


# ==================== PYTEST MARKERS ====================

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks tests as edge case tests"
    )
    config.addinivalue_line(
        "markers", "deep_learning: marks tests requiring PyTorch"
    )


# ==================== AUTO-USE CLEANUP ====================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test (especially for GPU memory)"""
    yield
    
    # Cleanup GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
