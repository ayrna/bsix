import numpy as np

class StepFunction:

    """
    StepFunction.
    """

    def __init__(self, X, y, is_survival=True):
        self.X = X
        self.y = y
        self.is_survival = is_survival
        
    def __call__(self, t):
        scalar_input = np.ndim(t) == 0
        t = np.atleast_1d(t)
        
        res = np.zeros_like(t, dtype=float)
        if len(self.X) == 0:
            res[:] = 1.0 if self.is_survival else 0.0
            return res[0] if scalar_input else res
            
        indices = np.searchsorted(self.X, t, side='right') - 1
        
        before_start = t < self.X[0]
        
        indices = np.clip(indices, 0, len(self.y) - 1)
        
        res = self.y[indices]
        res[before_start] = 1.0 if self.is_survival else 0.0
        
        return res[0] if scalar_input else res
    
    def __repr__(self):
        x_str = repr(self.X)
        y_str = repr(self.y)
        
        return f"StepFunction(x={x_str}, y={y_str})"

class BreslowEstimator:

    """
    Breslow estimator.
    """

    def __init__(self):
        self.times_ = None
        self.baseline_hazard_ = None
        self.baseline_survival_ = None

    def fit(self, risk_scores, events, times):
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
        log_risk = np.exp(risk_scores)

        return np.array([StepFunction(self.times_, np.power(self.baseline_survival_, er), is_survival=True) for er in log_risk])

    def get_cumulative_hazard_function(self, risk_scores):
        log_risk = np.exp(risk_scores)
        cum_baseline_hazard = -np.log(self.baseline_survival_)
        
        return np.array([StepFunction(self.times_, cum_baseline_hazard * er, is_survival=False) for er in log_risk])