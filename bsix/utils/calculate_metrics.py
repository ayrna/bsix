import numpy as np

from .classification_metrics import mae, amae, ms, ccr, recall
from .survival_metrics import concordanceIndexHarrel, concordanceIndexIPCW, cumulativeDinamicAUC

def format_predictions(preds):

    """
    Format predictions to be a list of arrays, one per progression. If the model only has one progression, wrap it in a list.
    """
    
    claves = ["survival", "binary"]
    preds_dict = dict(zip(claves, preds if isinstance(preds, tuple) else [preds]))

    return preds_dict

def from_results_to_metrics(targets, predictions):

    """
    Format results to compute metrics.
    """

    return compute_metrics(targets[0], targets[1], predictions)

def compute_survival_metrics(train_targets, evaluation_targets, predictions):

    """
    Compute survival metrics for given targets and predictions.
    """
    
    metrics = {}

    number_progressions = predictions.shape[1] if predictions.ndim > 1 else 1
    progressions = ["EXTENT_PROGRESS", "NEW_EIMSFUP", "DYSPL_NEO"] #["EXTENT_PROGRESS", "NEW_EIMSFUP", "COLECTOMY_FUP", "DYSPL_NEO"]
    has_progressions = number_progressions > 1

    for p in range(number_progressions):
        prefix = f"{progressions[p]} " if has_progressions else ""
        
        targets_survival = [train_targets[:, p], evaluation_targets[:, p]] if has_progressions else [train_targets, evaluation_targets]
        predictions_survival = predictions[:, p] if has_progressions else predictions

        metrics.update({
            f"{prefix}C-Index Harrel": concordanceIndexHarrel(targets_survival, predictions_survival),
            f"{prefix}C-Index IPCW": concordanceIndexIPCW(targets_survival, predictions_survival),
            f"{prefix}Cumulative Dinamic AUC": cumulativeDinamicAUC(targets_survival, predictions_survival),
        })

    return metrics

def compute_binary_metrics(evaluation_targets, predictions):

    """
    Compute binary metrics for given targets and predictions.
    """

    metrics = {}

    number_progressions = predictions.shape[1] if predictions.ndim > 1 else 1
    progressions = ["EXTENT_PROGRESS", "NEW_EIMSFUP", "DYSPL_NEO"] # ["EXTENT_PROGRESS", "NEW_EIMSFUP", "COLECTOMY_FUP", "DYSPL_NEO"]
    has_progressions = number_progressions > 1

    for p in range(number_progressions):
        prefix = f"{progressions[p]} " if has_progressions else ""
        
        targets_binary = evaluation_targets[:, p]["event"] if has_progressions else evaluation_targets["event"]
        predictions_binary = np.where(predictions[:, p] >= 0.5, 1.0, 0.0) if has_progressions else np.where(predictions >= 0.5, 1.0, 0.0)

        metrics.update({
            f"{prefix}MAE": mae(targets_binary, predictions_binary),
            f"{prefix}AMAE": amae(targets_binary, predictions_binary),
            f"{prefix}MS": ms(targets_binary, predictions_binary),
            f"{prefix}CCR": ccr(targets_binary, predictions_binary),
        })

        sensitivities = np.array(recall(targets_binary, predictions_binary, average=None))
        for i, sens in enumerate(sensitivities):
            metrics[f"{prefix}RECALL{i}"] = sens

    return metrics

def compute_metrics(train_targets, evaluation_targets, predictions):

    """
    Compute metrics for given targets and predictions (experiments).
    """
    
    try:
        # If ndarray.dtype = object, extract the element (dict)
        predictions = predictions.item()
    except Exception:
        # Ignore
        pass

    if not isinstance(predictions, dict):
        predictions  = format_predictions(predictions)
    
    metrics = {}

    has_binary = "binary" in list(predictions.keys())

    if has_binary:
        metrics.update(compute_survival_metrics(train_targets, evaluation_targets, predictions["survival"]))
        metrics.update(compute_binary_metrics(evaluation_targets, predictions["binary"]))
    else:
        metrics.update(compute_survival_metrics(train_targets, evaluation_targets, predictions["survival"]))    

    return metrics

def compute_metric_confidence_interval(y, prediction, metric_name, n_iterations=1000, confidence_level=0.95, seed=0):

    """
    Compute confidence interval using bootstrapping.
    """

    import numpy.lib.recfunctions as rfn
    from sksurv.metrics import concordance_index_censored
    
    rng = np.random.default_rng(seed=seed)

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
            # Ignore
            pass
                
    # Extract percentiles
    alpha = (1.0 - confidence_level) / 2.0
    lower_percentile = alpha * 100
    upper_percentile = (1.0 - alpha) * 100
    
    confidence_lower = np.percentile(bootstrapped_c_indices, lower_percentile)
    confidence_upper = np.percentile(bootstrapped_c_indices, upper_percentile)
    metric_mean = np.mean(bootstrapped_c_indices)
    
    return metric_mean, confidence_lower, confidence_upper, confidence_level, metric_name