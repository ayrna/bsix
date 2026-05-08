import numpy as np
import numpy.lib.recfunctions as rfn

from .survival_utils import getTau, getTimes
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc

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
        t = np.array([tiempo for _, tiempo in col_y], dtype=np.float64)
        
        c_index = concordance_index_censored(e, t, risk[:, p])[0]
        c_indices.append(c_index)

    c_index_censored = np.mean(c_indices)
    
    return c_index_censored

def concordanceIndexHarrel(y_true, y_pred):

    """
    Computes the Harrell's Concordance Index (C-index).
    """
    
    risk = y_pred.copy()
    _y_true = y_true[1].copy()

    risk = risk.squeeze()
    _y_true = _y_true.squeeze()

    if all(name in _y_true.dtype.names for name in ["time_start", "time_stop"]):
        _y_true = rfn.drop_fields(_y_true, ["time_start", "time"])
        _y_true = rfn.rename_fields(_y_true, {"time_stop": "time"})
        
    e = np.array([evento for evento, _ in _y_true], np.bool_)
    t = np.array([tiempo for _, tiempo in _y_true], np.float64)

    return concordance_index_censored(e, t, risk)[0]

def concordanceIndexIPCW(y_true, y_pred):

    """
    Computes the Inverse Probability of Censoring Weighted (IPCW).
    """

    risk = y_pred.copy()
    survival_train = y_true[0].copy()
    survival_test = y_true[1].copy()
    
    risk = risk.squeeze()
    survival_train = survival_train.squeeze()
    survival_test = survival_test.squeeze()

    if all(name in survival_train.dtype.names for name in ["time_start", "time_stop"]):
            survival_train = rfn.drop_fields(survival_train, ["time_start", "time"])
            survival_train = rfn.rename_fields(survival_train, {"time_stop": "time"})

    if all(name in survival_test.dtype.names for name in ["time_start", "time_stop"]):
            survival_test = rfn.drop_fields(survival_test, ["time_start", "time"])
            survival_test = rfn.rename_fields(survival_test, {"time_stop": "time"})

    tau, survival_train, survival_test, risk = getTau(survival_train, survival_test, risk)
    
    return concordance_index_ipcw(survival_train, survival_test, risk)[0]

def cumulativeDinamicAUC(y_true, y_pred):
    
    """
    Computes the Cumulative Dynamic AUC (AUC).
    """

    risk = y_pred.copy()
    survival_train = y_true[0].copy()
    survival_test = y_true[1].copy()

    risk = risk.squeeze()
    survival_train = survival_train.squeeze()
    survival_test = survival_test.squeeze()

    if all(name in survival_train.dtype.names for name in ["time_start", "time_stop"]):
            survival_train = rfn.drop_fields(survival_train, ["time_start", "time"])
            survival_train = rfn.rename_fields(survival_train, {"time_stop": "time"})

    if all(name in survival_test.dtype.names for name in ["time_start", "time_stop"]):
            survival_test = rfn.drop_fields(survival_test, ["time_start", "time"])
            survival_test = rfn.rename_fields(survival_test, {"time_stop": "time"})

    tau, survival_train, survival_test, risk = getTau(survival_train, survival_test, risk)
    times = getTimes(survival_test)
    
    return (cumulative_dynamic_auc(survival_train, survival_test, risk, times)[0]).tolist()