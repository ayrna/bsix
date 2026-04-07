import numpy as np
import numpy.lib.recfunctions as rfn

from .survival_utils import getTau, getTimes
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc

def scorerConcordanceIndex(y_true, y_pred):

    """
    Scorer for Concordance Index (C-index).
    """

    c_index_censored = 0
    
    if y_true.ndim == 1:
        risk = y_pred.copy()
        _y_true = y_true.copy()
        
        if all(name in _y_true.dtype.names for name in ["time_start", "time_stop"]):
            _y_true = rfn.drop_fields(_y_true, ["time_start", "time"])
            _y_true = rfn.rename_fields(_y_true, {"time_stop": "time"})
            
        e = np.array([evento for evento, _ in _y_true], np.bool_)
        t = np.array([tiempo for _, tiempo in _y_true], np.float64)

        c_index_censored = concordance_index_censored(e, t, risk)[0]
    else:
        risk = y_pred[0].copy()
        e = []
        t = []
        for i in range(y_true.shape[1]):
            e.append(np.array([evento for evento, _ in y_true[:, i]], np.bool_))
            t.append(np.array([tiempo for _, tiempo in y_true[:, i]], np.float32))
        e = np.array(e, np.bool_).T
        t = np.array(t, np.float32).T
        
        c_index = []
        for p in range(y_true.shape[1]):
            c_index.append(concordance_index_censored(e[:, p], t[:, p], risk[:, p])[0])
        c_index = np.array(c_index, np.float32)

        c_index_censored = np.mean(c_index)
    
    return c_index_censored

def concordanceIndexHarrel(y_true, y_pred):

    """
    Computes the Harrell's Concordance Index (C-index).
    """

    risk = y_pred.copy()
    _y_true = y_true[1].copy()

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
    
    survival_train = y_true[0].copy()
    survival_test = y_true[1].copy()
    
    if all(name in survival_train.dtype.names for name in ["time_start", "time_stop"]):
            survival_train = rfn.drop_fields(survival_train, ["time_start", "time"])
            survival_train = rfn.rename_fields(survival_train, {"time_stop": "time"})

    if all(name in survival_test.dtype.names for name in ["time_start", "time_stop"]):
            survival_test = rfn.drop_fields(survival_test, ["time_start", "time"])
            survival_test = rfn.rename_fields(survival_test, {"time_stop": "time"})
            
    risk = y_pred

    tau, survival_train, survival_test, risk = getTau(survival_train, survival_test, risk)
    
    return concordance_index_ipcw(survival_train, survival_test, risk)[0]

def cumulativeDinamicAUC(y_true, y_pred):
    
    """
    Computes the Cumulative Dynamic AUC (AUC).
    """
    
    survival_train = y_true[0].copy()
    survival_test = y_true[1].copy()

    if all(name in survival_train.dtype.names for name in ["time_start", "time_stop"]):
            survival_train = rfn.drop_fields(survival_train, ["time_start", "time"])
            survival_train = rfn.rename_fields(survival_train, {"time_stop": "time"})

    if all(name in survival_test.dtype.names for name in ["time_start", "time_stop"]):
            survival_test = rfn.drop_fields(survival_test, ["time_start", "time"])
            survival_test = rfn.rename_fields(survival_test, {"time_stop": "time"})
    risk = y_pred

    tau, survival_train, survival_test, risk = getTau(survival_train, survival_test, risk)
    times = getTimes(survival_test)
    
    return (cumulative_dynamic_auc(survival_train, survival_test, risk, times)[0]).tolist()