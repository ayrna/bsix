import numpy as np
import scipy.stats
import warnings

from sklearn.metrics import confusion_matrix

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
    Compute the Correct Classification Rate (CCR).
    """

    return np.count_nonzero(y_true == y_pred) / float(len(y_true))

def amae(y_true, y_pred):

    """
    Compute the Average Mean Absolute Error (AMAE).
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y_true, y_pred)
        n_class = cm.shape[0]
        costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
        costs = np.abs(costs - np.transpose(costs))
        errores = costs * cm
        
        sum_rows = np.sum(cm, ).astype("float")
        with np.errstate(divide="ignore", invalid="ignore"):
            amaes = np.sum(errores, ) / sum_rows
        
        amaes = amaes[~np.isnan(amaes)]
        return np.mean(amaes)

def gm(y_true, y_pred):

    """
    Compute the Geometric Mean (GM).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y_true, y_pred)
        sum_byclass = np.sum(cm, )
        
        with np.errstate(divide="ignore", invalid="ignore"):
            sensitivities = np.diag(cm) / sum_byclass.astype("float")
            
        sensitivities[sum_byclass == 0] = 1
        gm_result = pow(np.prod(sensitivities), 1.0 / cm.shape[0])
        return gm_result

def mae(y_true, y_pred):

    """
    Compute the Mean Absolute Error (MAE).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return abs(y_true - y_pred).sum() / len(y_true)

def mmae(y_true, y_pred):

    """
    Compute the Maximum Mean Absolute Error (MMAE).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y_true, y_pred)
        n_class = cm.shape[0]
        costes = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
        costes = np.abs(costes - np.transpose(costes))
        errores = costes * cm
        
        sum_rows = np.sum(cm, ).astype("float")
        with np.errstate(divide="ignore", invalid="ignore"):
            amaes = np.sum(errores, ) / sum_rows
            
        amaes = amaes[~np.isnan(amaes)]
        if len(amaes) == 0: return 0.0
        return amaes.max()

def recall(y_true, y_pred, average="macro"):

    """
    Compute the Recall (Sensitivity).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y_true, y_pred)
        sum_byclass = np.sum(cm, ).astype("float")
        
        with np.errstate(divide="ignore", invalid="ignore"):
            sensitivities = np.diag(cm) / sum_byclass
        
        sensitivities = np.nan_to_num(sensitivities)
        
        if average is None:
            return sensitivities
        else:
            return np.mean(sensitivities)

def ms(y_true, y_pred):

    """
    Compute the Minimum Sensitivity (MS).
    """

    sensitivities = recall(y_true, y_pred, average=None)
    return np.min(sensitivities)

def mze(y_true, y_pred):

    """
    Compute the Mean Zero Error (MZE).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        confusion = confusion_matrix(y_true, y_pred)
        return 1 - np.diagonal(confusion).sum() / confusion.sum()

def tkendall(y_true, y_pred):

    """
    Compute the Kendall"s Tau (TKendall).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, pvalue = scipy.stats.kendalltau(y_true, y_pred)
        return corr

def wkappa(y_true, y_pred):

    """
    Compute the Weighted Kappa (WKappa).
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y_true, y_pred)
        n_class = cm.shape[0]
        costes = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
        costes = np.abs(costes - np.transpose(costes))
        f = 1 - costes
        n = cm.sum()
        x = cm / n
        r = x.sum()
        s = x.sum(axis=0)
        Ex = r.reshape(-1, 1) * s
        po = (x * f).sum()
        pe = (Ex * f).sum()
        if pe == 1: return 1.0
        return (po - pe) / (1 - pe)

def spearman(y_true, y_pred):

    """
    Compute the Spearman"s Rank Correlation Coefficient (Spearman).
    """


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, pvalue = scipy.stats.spearmanr(y_true, y_pred)
        if np.isnan(corr):
            return 0.0
        return corr