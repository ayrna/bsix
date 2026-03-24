import numpy as np
import scipy.stats
import warnings

from sklearn.metrics import confusion_matrix

def ccr(y, y_pred):
    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred

    return np.count_nonzero(y == y_pred) / float(len(y))


def amae(y, y_pred):
    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y, y_pred)
        n_class = cm.shape[0]
        costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
        costs = np.abs(costs - np.transpose(costs))
        errores = costs * cm
        
        sum_rows = np.sum(cm, ).astype("float")
        with np.errstate(divide='ignore', invalid='ignore'):
            amaes = np.sum(errores, ) / sum_rows
        
        amaes = amaes[~np.isnan(amaes)]
        return np.mean(amaes)


def gm(y, y_pred):
    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y, y_pred)
        sum_byclass = np.sum(cm, )
        
        with np.errstate(divide='ignore', invalid='ignore'):
            sensitivities = np.diag(cm) / sum_byclass.astype("float")
            
        sensitivities[sum_byclass == 0] = 1
        gm_result = pow(np.prod(sensitivities), 1.0 / cm.shape[0])
        return gm_result


def mae(y, y_pred):
    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        return abs(y - y_pred).sum() / len(y)


def mmae(y, y_pred):
    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y, y_pred)
        n_class = cm.shape[0]
        costes = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
        costes = np.abs(costes - np.transpose(costes))
        errores = costes * cm
        
        sum_rows = np.sum(cm, ).astype("float")
        with np.errstate(divide='ignore', invalid='ignore'):
            amaes = np.sum(errores, ) / sum_rows
            
        amaes = amaes[~np.isnan(amaes)]
        if len(amaes) == 0: return 0.0
        return amaes.max()


def recall(y, y_pred, average='macro'):
    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y, y_pred)
        sum_byclass = np.sum(cm, ).astype("float")
        
        with np.errstate(divide='ignore', invalid='ignore'):
            sensitivities = np.diag(cm) / sum_byclass
        
        sensitivities = np.nan_to_num(sensitivities)
        
        if average is None:
            return sensitivities
        else:
            return np.mean(sensitivities)


def ms(y, y_pred):
    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred

    sensitivities = recall(y, y_pred, average=None)
    return np.min(sensitivities)


def mze(y, y_pred):
    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        confusion = confusion_matrix(y, y_pred)
        return 1 - np.diagonal(confusion).sum() / confusion.sum()


def tkendall(y, y_pred):
    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, pvalue = scipy.stats.kendalltau(y, y_pred)
        return corr


def wkappa(y, y_pred):
    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y, y_pred)
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


def spearman(y, y_pred):
    y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, pvalue = scipy.stats.spearmanr(y, y_pred)
        if np.isnan(corr):
            return 0.0
        return corr