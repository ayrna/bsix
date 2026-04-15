import numpy as np

from .classification_metrics import mae, amae, ms, ccr, recall
from .survival_metrics import concordanceIndexHarrel, concordanceIndexIPCW, cumulativeDinamicAUC

def get_metrics(targets, predictions):

    """
    Compute metrics for given targets and predictions.
    """
    
    # targets = (y_train, y_test)
    # predictions = survival_pred 'or' (survival_pred, binary_pred, ...)

    metrics = {}
    
    if len(predictions) == 1:
        metrics.update({
            "C-Index Harrel": concordanceIndexHarrel(targets, predictions[0]),
            "C-Index IPCW": concordanceIndexIPCW(targets, predictions[0]),
            "Cumulative Dinamic AUC": cumulativeDinamicAUC(targets, predictions[0]),
        })
    elif len(predictions) == 2:
        metrics.update({
            "C-Index Harrel": concordanceIndexHarrel(targets, predictions[0]),
            "C-Index IPCW": concordanceIndexIPCW(targets, predictions[0]),
            "Cumulative Dinamic AUC": cumulativeDinamicAUC(targets, predictions[0]),
        })

        binary_predictions = np.where(predictions[1] >= 0.5, 1.0, 0.0)
        metrics.update({
            "MAE": mae(targets[1]["event"], binary_predictions),
            "AMAE": amae(targets[1]["event"], binary_predictions),
            "MS": ms(targets[1]["event"], binary_predictions),
            "CCR": ccr(targets[1]["event"], binary_predictions),
        })

        sensitivities = np.array(recall(targets[1]["event"], binary_predictions, average=None))

        for i, sens in enumerate(sensitivities):
            metrics.update({
                f"RECALL{i}": sens,
            })

    return metrics

def get_metric_confidence_interval(y, prediction, metric_name, n_iterations=1000, confidence_level=0.95, random_state=0):

    """
    Compute confidence interval using bootstrapping.
    """

    import numpy.lib.recfunctions as rfn
    from sksurv.metrics import concordance_index_censored
    
    rng = np.random.default_rng(seed=random_state)

    if all(name in y.dtype.names for name in ["time_start", "time_stop"]):
        y = rfn.drop_fields(y, ["time_start", "time"])
        y = rfn.rename_fields(y, {'time_stop': 'time'})
        
    event = np.array([evento for evento, _ in y], np.bool_)
    time = np.array([tiempo for _, tiempo in y], np.float64)
    n_samples = len(time)
    
    bootstrapped_c_indices = []
    for _ in range(n_iterations):
        # Reindex with replacement
        indices = rng.choice(range(n_samples), size=int(n_samples * 0.9), replace=True)
        
        # Filter the data for the current sample
        sample_time = time[indices]
        sample_event = event[indices]
        sample_prediction = prediction[indices]
        
        # Compute metric
        try:
            if metric_name == "cindex":
                value = concordance_index_censored(sample_event, sample_time, sample_prediction)[0]
            elif metric_name == "amae":
                value = amae(sample_event, sample_prediction)
            bootstrapped_c_indices.append(value)
        except Exception:
            # Ignorar muestras anómalas
            pass
                
    # Extract percentiles
    alpha = (1.0 - confidence_level) / 2.0
    lower_percentile = alpha * 100
    upper_percentile = (1.0 - alpha) * 100
    
    confidence_lower = np.percentile(bootstrapped_c_indices, lower_percentile)
    confidence_upper = np.percentile(bootstrapped_c_indices, upper_percentile)
    metric_mean = np.mean(bootstrapped_c_indices)
    
    return metric_mean, confidence_lower, confidence_upper, confidence_level, metric_name