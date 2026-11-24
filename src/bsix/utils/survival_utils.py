import numpy as np


def _upper_bound(survival):
    """
    Get the upper bound of the time variable in the survival data.
    """

    return survival["time"][survival["time"] < survival["time"].max()].max()


def getTau(survival_train, survival_test):
    """
    Get the time point up to which the evaluation is performed.
    """

    if survival_test.shape[0] == 1:
        return survival_test["time"][0]

    tau = min(
        np.percentile(survival_train["time"], 90),
        np.percentile(survival_test["time"], 90),
    )
    upper = min(_upper_bound(survival_train), _upper_bound(survival_test))

    return min(tau, upper)


def getTimes(survival_test, num_division=10):
    """
    Get the time points for performing the evaluation.
    """

    if survival_test.shape[0] == 1:
        return survival_test["time"]

    times = np.linspace(
        np.percentile(survival_test["time"], 10),
        min(np.percentile(survival_test["time"], 90), _upper_bound(survival_test)),
        num_division,
    )

    return times[
        (times >= survival_test["time"].min()) & (times < survival_test["time"].max())
    ]
