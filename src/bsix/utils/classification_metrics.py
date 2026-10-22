import numpy as np
import scipy.stats
import warnings

from sklearn.metrics import confusion_matrix

def _validate_input(y_true, y_pred):
    """
    Validates the input data.
    """

    if y_true is None or y_pred is None:
        raise ValueError("Input data cannot be None.")
    if len(y_true) == 0 or  len(y_pred) == 0:
        raise ValueError("Input data X and y cannot be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("The number of samples in y_true and y_pred must be equal.")
    
def scorerAmae(y, y_pred):

    """
    Scorer for Average Mean Absolute Error (AMAE).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = np.array([evento for evento, _ in y], np.float32)
        ePred = np.where(y_pred >= 0.5, 1.0, 0.0)

        return amae(e, ePred)

def ccr(y_true, y_pred):

    """
    Computes the Correct Classification Rate (CCR).
    """

    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
    y_true = y_true.squeeze() if y_true.ndim > 1 else y_true

    _validate_input(y_true, y_pred)

    return float(np.count_nonzero(y_true == y_pred) / float(len(y_true)))

def amae(y_true, y_pred):

    """
    Computes the Average Mean Absolute Error (AMAE).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
        y_true = y_true.squeeze() if y_true.ndim > 1 else y_true

        _validate_input(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        n_class = cm.shape[0]
        costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
        costs = np.abs(costs - np.transpose(costs))
        errores = costs * cm

        sum_rows = np.sum(cm, axis=1).astype("float")
        with np.errstate(divide='ignore', invalid='ignore'):
            amaes = np.sum(errores, axis=1) / sum_rows

        amaes = amaes[~np.isnan(amaes)]
        if len(amaes) == 0:
            return 0.0
        return float(np.mean(amaes))

def gm(y_true, y_pred):

    """
    Computes the Geometric Mean (GM).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
        y_true = y_true.squeeze() if y_true.ndim > 1 else y_true

        _validate_input(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        sum_byclass = np.sum(cm, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            sensitivities = np.diag(cm) / sum_byclass.astype("float")

        sensitivities[sum_byclass == 0] = 1
        gm_result = pow(np.prod(sensitivities), 1.0 / cm.shape[0])
        return float(gm_result)


def mae(y_true, y_pred):
    
    """
    Computes the Mean Absolute Error (MAE).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
        y_true = y_true.squeeze() if y_true.ndim > 1 else y_true

        _validate_input(y_true, y_pred)

        return float(np.abs(y_true.astype(float) - y_pred.astype(float)).sum() / len(y_true))

def mmae(y_true, y_pred):
    
    """
    Computes the Maximum Mean Absolute Error (MMAE).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
        y_true = y_true.squeeze() if y_true.ndim > 1 else y_true

        _validate_input(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        n_class = cm.shape[0]
        costes = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
        costes = np.abs(costes - np.transpose(costes))
        errores = costes * cm

        sum_rows = np.sum(cm, axis=1).astype("float")
        with np.errstate(divide='ignore', invalid='ignore'):
            amaes = np.sum(errores, axis=1) / sum_rows

        amaes = amaes[~np.isnan(amaes)]
        if len(amaes) == 0:
            return 0.0
        
        return float(amaes.max())


def recall(y_true, y_pred, average='macro'):

    """
    Computes the Recall (Sensitivity).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
        y_true = y_true.squeeze() if y_true.ndim > 1 else y_true

        _validate_input(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        sum_byclass = np.sum(cm, axis=1).astype("float")

        with np.errstate(divide='ignore', invalid='ignore'):
            sensitivities = np.diag(cm) / sum_byclass

        sensitivities = np.nan_to_num(sensitivities)

        if average is None:
            return sensitivities
        else:
            return float(np.mean(sensitivities))

def ms(y_true, y_pred):

    """
    Computes the Minimum Sensitivity (MS).
    """

    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
    y_true = y_true.squeeze() if y_true.ndim > 1 else y_true

    _validate_input(y_true, y_pred)

    sensitivities = recall(y_true, y_pred, average=None)

    return float(np.min(sensitivities))

def mze(y_true, y_pred):

    """
    Computes the Mean Zero Error (MZE).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
        y_true = y_true.squeeze() if y_true.ndim > 1 else y_true

        _validate_input(y_true, y_pred)

        confusion = confusion_matrix(y_true, y_pred)
        return float(1 - np.diagonal(confusion).sum() / confusion.sum())

def tkendall(y_true, y_pred):
    
    """
    Computes the Kendall's Tau (TKendall).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
        y_true = y_true.squeeze() if y_true.ndim > 1 else y_true

        _validate_input(y_true, y_pred)

        corr, pvalue = scipy.stats.kendalltau(y_true, y_pred)
        if np.isnan(corr):
            return 0.0
        return float(corr)

def wkappa(y_true, y_pred):

    """
    Computes the Weighted Kappa (WKappa).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
        y_true = y_true.squeeze() if y_true.ndim > 1 else y_true

        _validate_input(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        n_class = cm.shape[0]
        costes = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
        costes = np.abs(costes - np.transpose(costes))
        f = 1 - costes
        n = cm.sum()
        x = cm / n
        r = x.sum(axis=1)
        s = x.sum(axis=0)
        Ex = r.reshape(-1, 1) * s
        po = (x * f).sum()
        pe = (Ex * f).sum()
        if pe == 1:
            return 1.0
        return float((po - pe) / (1 - pe))

def spearman(y_true, y_pred):

    """
    Computes the Spearman's Rank Correlation Coefficient (Spearman).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
        y_true = y_true.squeeze() if y_true.ndim > 1 else y_true

        _validate_input(y_true, y_pred)

        corr, pvalue = scipy.stats.spearmanr(y_true, y_pred)
        if np.isnan(corr):
            return 0.0
        return float(corr)
