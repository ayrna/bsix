import numpy as np

def getTau(survival_train, survival_test):

    """
    Get the time point up to which the evaluation is performed.
    """
    
    tau = min(np.percentile(survival_train["time"], 90), np.percentile(survival_test["time"], 90))

    return tau

def getTimes(survival_test, num_division=10):

    """
    Get the time points for performing the evaluation.
    """
    
    times = np.linspace(np.percentile(survival_test["time"], 10), np.percentile(survival_test["time"], 90), num_division)

    return times
