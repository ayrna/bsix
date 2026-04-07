import numpy as np

def getTau(survival_train, survival_test, risk):

    """
    Get the time point up to which the evaluation is performed.
    """
    
    tau = min(survival_train["time"].max() - 1, survival_test["time"].max() - 1)
    mask = survival_test["time"] < tau

    survival_test = survival_test[mask]
    risk = risk[mask]

    return tau, survival_train, survival_test, risk

def getTimes(survival_test, num_division=3):

    """
    Get the time points for performing the evaluation.
    """

    t = np.array([tiempo for _, tiempo in survival_test], np.float64)
    range_t = t.max() - t.min()
    
    times = np.linspace(t.min() + (0.25 * range_t), t.max() - (0.25 * range_t), num_division)

    return times