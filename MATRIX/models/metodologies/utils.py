import numpy as np

class StepFunction:

    """
    StepFunction.
    """

    def __init__(self, X, y, is_survival=True):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.is_survival = is_survival
        
    def __call__(self, t):
        t = np.asarray(t)
        scalar_input = t.ndim == 0
        if scalar_input:
            t = np.array([t])
            
        res = np.zeros_like(t, dtype=float)
        for i, t_val in enumerate(t):
            if len(self.X) == 0 or t_val < self.X[0]:
                res[i] = 1.0 if self.is_survival else 0.0
            else:
                idx = np.searchsorted(self.X, t_val, side='right') - 1
                idx = min(max(idx, 0), len(self.y) - 1)
                res[i] = self.y[idx]
                
        return res[0] if scalar_input else res

class BreslowEstimator:

    """
    Breslow estimator.
    """

    def __init__(self):
        self.times_ = None
        self.baseline_hazard_ = None
        self.baseline_survival_ = None

    def fit(self, risk_scores, events, times):
        times = np.asarray(times)
        events = np.asarray(events).astype(bool)
        risk_scores = np.asarray(risk_scores)
        
        log_risk = np.exp(risk_scores)
        
        # Sort unique times
        unique_times = np.unique(times[events])
        unique_times.sort()
        self.times_ = unique_times
        
        baseline_hazard = []
        for t in unique_times:
            risk_set = times >= t
            events_at_t = np.sum((times == t) & events)
            sum_exp_risk = np.sum(log_risk[risk_set])
            
            if sum_exp_risk > 0:
                baseline_hazard.append(events_at_t / sum_exp_risk)
            else:
                baseline_hazard.append(0.0)
                
        self.baseline_hazard_ = np.array(baseline_hazard)
        cum_baseline_hazard = np.cumsum(self.baseline_hazard_)
        self.baseline_survival_ = np.exp(-cum_baseline_hazard)
        
        return self

    def get_survival_function(self, risk_scores):
        log_risk = np.exp(np.asarray(risk_scores))

        return np.array([StepFunction(self.times_, np.power(self.baseline_survival_, er), is_survival=True) for er in log_risk])

    def get_cumulative_hazard_function(self, risk_scores):
        log_risk = np.exp(np.asarray(risk_scores))
        cum_baseline_hazard = -np.log(self.baseline_survival_)
        
        return np.array([StepFunction(self.times_, cum_baseline_hazard * er, is_survival=False) for er in log_risk])