import numpy as np
from sksurv.ensemble import RandomSurvivalForest

def impute_censored_times(X, y, n_estimators=100, random_state=0):

    """
    Estimate survival times for censored patients using restricted conditional life expectancy.
    """
    
    estimator = RandomSurvivalForest(n_estimators=n_estimators, random_state=random_state)
    estimator.fit(X, y)
    
    survival_f = estimator.predict_survival_function(X)

    events = y["event"].astype(bool)
    times = y["time"].astype(np.float32)
    
    imputed_times = np.copy(times)
    
    for i in range(len(y)):
        e_i = events[i]
        t_i = times[i]
        
        if not e_i:
            f = survival_f[i]
            
            mask_residual = f.x >= t_i
            t_residual = f.x[mask_residual]
            
            if len(t_residual) > 1:
                survival_t_i = f(t_i)
                
                if survival_t_i > 0:
                    survival_t_residual = f(t_residual)
                    integral = np.trapezoid(y=survival_t_residual, x=t_residual)
                    
                    imputed_times[i] = t_i + (integral / survival_t_i)
                    
    return imputed_times

def compute_jackknife_soft_labels(time, event, t_max):

    """
    Compute the soft labels based on pseudo-Jackknife values at t_max
    """

    n_samples = len(time)
    mask_events = (time <= t_max) & (event == 1)
    unique_t, d_j = np.unique(time[mask_events], return_counts=True)

    if len(unique_t) == 0:
        raise ValueError(f"When using `compute_jackknife_soft_labels`, there are no events before t_max={t_max}")
    
    n_j = np.array([np.sum(time >= t) for t in unique_t])
    
    global_km = np.prod(1.0 - d_j / n_j)
    
    at_risk_mask = time[:, None] >= unique_t
    n_matrix = n_j - at_risk_mask.astype(np.float32)
    
    event_mask = (time[:, None] == unique_t) & (event[:, None] == 1)
    d_matrix = d_j - event_mask.astype(np.float32)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        hazard_factors = 1.0 - (d_matrix / n_matrix)
        hazard_factors[n_matrix == 0] = 1.0
        
    km_minus_i = np.prod(hazard_factors, axis=1)
    
    pseudo_values = n_samples * global_km - (n_samples - 1) * km_minus_i
    
    pseudo_values = np.clip(pseudo_values, 0.0, 1.0)
    soft_labels = 1.0 - pseudo_values
    
    return soft_labels

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