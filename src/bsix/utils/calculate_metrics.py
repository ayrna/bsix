import numpy as np

from .classification_metrics import mae, amae, ms, ccr, recall
from .survival_metrics import integratedBrierScore, concordanceIndexHarrel, concordanceIndexIPCW, cumulativeDinamicAUC


def _validate_structured_target(target, name):
    """
    Validate that the target is a structured NumPy array with required fields.
    """
    if target is None:
        raise ValueError(f"{name} cannot be None.")
    if not hasattr(target, "dtype") or target.dtype.names is None:
        raise ValueError(f"{name} must be a NumPy structured array.")

    required_fields = {"event", "time", "time_stop"}
    missing_fields = required_fields - set(target.dtype.names)
    if len(missing_fields) >= 2:
        raise ValueError(f"{name} is missing required fields: {missing_fields}.")

    if len(target) <= 1:
        raise ValueError(f"{name} must contain at least two samples.")

    if np.all(target["event"] == 0):
        raise ValueError(f"All samples in {name} are censored.")


def _validate_input(train_targets, evaluation_targets, predictions, survival_function = None):
    """
    Validate input data for compute_metrics in survival analysis.

    Parameters
    ----------
    train_targets : np.ndarray
        Structured array with 'event' and 'time' fields.
    evaluation_targets : np.ndarray
        Structured array with 'event' and 'time' fields.
    predictions : dict
        Dictionary containing 'survival' predictions and optional 'binary' predictions.
    survival_function : array-like of StepFunction, optional
        Baseline survival step function objects.

    Raises
    ------
    ValueError or TypeError
        If any validation check fails.
    """
    
    if not isinstance(predictions, dict):
        raise TypeError("Predictions must be a dictionary-like object.")

    if "survival" not in predictions or predictions["survival"] is None:
        raise ValueError("Predictions must contain a non-None 'survival' key.")

    _validate_structured_target(train_targets, "train_targets")
    _validate_structured_target(evaluation_targets, "evaluation_targets")

    for key in ("survival", "binary"):
        if key not in predictions or predictions[key] is None:
            continue

        pred_array = np.asarray(predictions[key])

        if len(pred_array) != len(evaluation_targets):
            raise ValueError(f"Length of predictions['{key}'] ({len(pred_array)}) does not match evaluation_targets ({len(evaluation_targets)}).")

        if not np.all(np.isfinite(pred_array)):
            raise ValueError(f"Predictions['{key}'] contains infinite or NaN values.")

    if survival_function is not None:
        from bsix.models.metodologies.utils import StepFunction

        sf_array = np.asarray(survival_function)
        if sf_array.size == 0:
            raise ValueError("Survival_function cannot be empty.")

        if not all(isinstance(s, StepFunction) for s in sf_array.flat):
            raise ValueError("All elements in survival_function must be StepFunction instances.")
        
def format_predictions(preds):

    """
    Format predictions to survival or binary mode.
    """
    
    claves = ["survival", "binary"]
    preds_dict = dict(zip(claves, preds if isinstance(preds, tuple) else [preds]))

    return preds_dict

def from_results_to_metrics(targets, predictions):

    """
    Format results to compute metrics.
    """

    return compute_metrics(targets[0], targets[1], predictions)

def compute_survival_metrics(train_targets, evaluation_targets, predictions, survival_function=None):

    """
    Compute survival metrics for given targets and predictions.
    If survival_function is provided, additional metrics are computed.
    """
    
    metrics = {}

    number_progressions = predictions.shape[1] if predictions.ndim > 1 else 1
    progressions = [f"P{k} " for k in range(number_progressions)] #["EXTENT_PROGRESS", "NEW_EIMSFUP", "COLECTOMY_FUP", "DYSPL_NEO"]
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

        if survival_function is not None:
            survival_function_p = survival_function[p] if has_progressions else survival_function

            metrics.update({
                f"{prefix}Brier Score": integratedBrierScore(targets_survival, survival_function_p),
            })

    return metrics

def compute_binary_metrics(evaluation_targets, predictions):

    """
    Compute binary metrics for given targets and predictions.
    """

    metrics = {}

    number_progressions = predictions.shape[1] if predictions.ndim > 1 else 1
    progressions = [f"P{k} " for k in range(number_progressions)] # ["EXTENT_PROGRESS", "NEW_EIMSFUP", "COLECTOMY_FUP", "DYSPL_NEO"]
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

def compute_metrics(train_targets, evaluation_targets, predictions, survival_function=None):

    """
    Compute metrics for given targets and predictions (experiments).
    If survival_function is provided, additional metrics are computed.
    """

    try:
        # If ndarray.dtype = object, extract the element (dict)
        predictions = predictions.item()
    except Exception:
        # Ignore
        pass

    if not isinstance(predictions, dict):
        predictions = format_predictions(predictions)

    _validate_input(train_targets, evaluation_targets, predictions, survival_function)
    
    metrics = {}

    has_binary = "binary" in list(predictions.keys())

    if has_binary:
        metrics.update(compute_survival_metrics(train_targets, evaluation_targets, predictions["survival"], survival_function))
        metrics.update(compute_binary_metrics(evaluation_targets, predictions["binary"]))
    else:
        metrics.update(compute_survival_metrics(train_targets, evaluation_targets, predictions["survival"], survival_function))

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
    time = np.array([tiempo for _, tiempo in y], np.float32)
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
