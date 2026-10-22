import numpy as np
import numpy.lib.recfunctions as rfn

from .survival_utils import getTau, getTimes
from sksurv.metrics import integrated_brier_score, concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc

def _filter_times(survival_train, survival_test, predictions):
    """
    Filter test to remove event times that exceed the maximum time observed in training.
    """
     
    mask = survival_test["time"] < survival_train["time"].max()

    survival_test_filtered = survival_test[mask]
    predictions_filtered = predictions[mask]

    return survival_test_filtered, predictions_filtered

def _validate_input(y_true, y_pred):
    """
    Validates the input data.
    """
    
    if y_true[0] is None or y_true[1] is None or y_pred is None:
        raise ValueError("Input data cannot be None.")
    if len(y_true[0]) == 0 or len(y_true[1]) == 0 or len(y_pred) == 0:
        raise ValueError("Input data X and y cannot be empty.")
    if len(y_true[0]) != len(y_true[1]) != len(y_pred):
        raise ValueError("The number of samples in y_true and y_pred must be equal.")
    
def scorerConcordanceIndex(y_true, y_pred):

    """
    Scorer for Concordance Index (C-index).
    """

    is_list_pred = isinstance(y_pred, tuple)
    risk = y_pred[0].copy() if is_list_pred else y_pred.copy()
    _y_true = y_true.copy()

    if all(name in _y_true.dtype.names for name in ["time_start", "time_stop"]):
        _y_true = rfn.drop_fields(_y_true, ["time_start", "time"])
        _y_true = rfn.rename_fields(_y_true, {"time_stop": "time"})
            
    if _y_true.ndim == 1:
        _y_true = _y_true.reshape(-1, 1)
        risk = risk.reshape(-1, 1)

    c_indices = []
    for p in range(_y_true.shape[1]):
        col_y = _y_true[:, p]
        
        e = np.array([evento for evento, _ in col_y], dtype=np.bool_)
        t = np.array([tiempo for _, tiempo in col_y], dtype=np.float32)
        
        c_index = concordance_index_censored(e, t, risk[:, p])[0]
        c_indices.append(c_index)

    c_index_censored = np.mean(c_indices)
    
    return c_index_censored

def integratedBrierScore(y_true, y_pred, times=None):

    """
    Computes the Brier Score (BS).
    """

    survival_function = y_pred.copy()
    survival_train = y_true[0].copy()
    survival_test = y_true[1].copy()

    survival_function = survival_function.squeeze() if survival_function.ndim > 1 else survival_function
    survival_train = survival_train.squeeze() if survival_train.ndim > 1 else survival_train
    survival_test = survival_test.squeeze() if survival_test.ndim > 1 else survival_test

    _validate_input(y_true, y_pred)

    if all(name in survival_train.dtype.names for name in ["time_start", "time_stop"]):
            survival_train = rfn.drop_fields(survival_train, ["time_start", "time"])
            survival_train = rfn.rename_fields(survival_train, {"time_stop": "time"})

    if all(name in survival_test.dtype.names for name in ["time_start", "time_stop"]):
            survival_test = rfn.drop_fields(survival_test, ["time_start", "time"])
            survival_test = rfn.rename_fields(survival_test, {"time_stop": "time"})

    survival_test, survival_function = _filter_times(survival_train, survival_test, survival_function)

    if times is None:
        times = getTimes(survival_test)

    times = np.atleast_1d(times).tolist()
    survival_function_times = np.array([sf(times) for sf in survival_function])

    b_score = integrated_brier_score(survival_train, survival_test, survival_function_times, times)

    return b_score

def concordanceIndexHarrel(y_true, y_pred):

    """
    Computes the Harrell's Concordance Index (C-index).
    """
    
    risk = y_pred.copy()
    _y_true = y_true[1].copy()

    risk = risk.squeeze() if risk.ndim > 1 else risk
    _y_true = _y_true.squeeze() if _y_true.ndim > 1 else _y_true

    _validate_input(y_true, y_pred)

    if all(name in _y_true.dtype.names for name in ["time_start", "time_stop"]):
        _y_true = rfn.drop_fields(_y_true, ["time_start", "time"])
        _y_true = rfn.rename_fields(_y_true, {"time_stop": "time"})
    
    e = np.array([evento for evento, _ in _y_true], np.bool_)
    t = np.array([tiempo for _, tiempo in _y_true], np.float32)

    return concordance_index_censored(e, t, risk)[0]

def concordanceIndexIPCW(y_true, y_pred):

    """
    Computes the Inverse Probability of Censoring Weighted (IPCW).
    """

    risk = y_pred.copy()
    survival_train = y_true[0].copy()
    survival_test = y_true[1].copy()
    
    risk = risk.squeeze() if risk.ndim > 1 else risk
    survival_train = survival_train.squeeze() if survival_train.ndim > 1 else survival_train
    survival_test = survival_test.squeeze() if survival_test.ndim > 1 else survival_test

    _validate_input(y_true, y_pred)

    if all(name in survival_train.dtype.names for name in ["time_start", "time_stop"]):
            survival_train = rfn.drop_fields(survival_train, ["time_start", "time"])
            survival_train = rfn.rename_fields(survival_train, {"time_stop": "time"})

    if all(name in survival_test.dtype.names for name in ["time_start", "time_stop"]):
            survival_test = rfn.drop_fields(survival_test, ["time_start", "time"])
            survival_test = rfn.rename_fields(survival_test, {"time_stop": "time"})

    survival_test, risk = _filter_times(survival_train, survival_test, risk)
    tau = getTau(survival_train, survival_test)
    
    return concordance_index_ipcw(survival_train, survival_test, risk, tau)[0]

def cumulativeDinamicAUC(y_true, y_pred, times=None):
    
    """
    Computes the Cumulative Dynamic AUC (AUC).
    """

    risk = y_pred.copy()
    survival_train = y_true[0].copy()
    survival_test = y_true[1].copy()

    risk = risk.squeeze() if risk.ndim > 1 else risk
    survival_train = survival_train.squeeze() if survival_train.ndim > 1 else survival_train
    survival_test = survival_test.squeeze() if survival_test.ndim > 1 else survival_test

    _validate_input(y_true, y_pred)

    if all(name in survival_train.dtype.names for name in ["time_start", "time_stop"]):
            survival_train = rfn.drop_fields(survival_train, ["time_start", "time"])
            survival_train = rfn.rename_fields(survival_train, {"time_stop": "time"})

    if all(name in survival_test.dtype.names for name in ["time_start", "time_stop"]):
            survival_test = rfn.drop_fields(survival_test, ["time_start", "time"])
            survival_test = rfn.rename_fields(survival_test, {"time_stop": "time"})

    survival_test, risk = _filter_times(survival_train, survival_test, risk)
    tau = getTau(survival_train, survival_test)
    
    if times is None:
        # times = getTimes(survival_test)
        times = np.array(tau, np.float32)
    
    c_auc = cumulative_dynamic_auc(survival_train, survival_test, risk, times)[0]
    c_auc = c_auc[0] if len(c_auc) == 1 else (c_auc).tolist()

    return c_auc
