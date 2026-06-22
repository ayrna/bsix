import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
import warnings

from ..base import BaseSurvival
from .utils import BreslowEstimator

warnings.filterwarnings("ignore")

class CoxRegression(BaseSurvival):

    """
    Cox Regression model.

    Parameters
    ----------
    alpha : float, default =0.0
        Regularization strength.
    ties : str, default = ``"breslow"``
        Method for handling tied event times. ``"breslow"`` or ``"efron"``.
    n_iter : int, default =100
        Number of iterations for the Newton-Raphson algorithm.

    Attributes
    ----------
    coef_ : array-like, shape (n_features,)
        Estimated coefficients for the model.
    breslow : BreslowEstimator
        Breslow estimator for baseline hazards.
    survival_function : array-like, shape (n_samples, n_times)
        Estimated survival function.
    cumulative_hazard_function : array-like, shape (n_samples, n_times)
        Estimated cumulative hazard function.
    shap_explainer : shap.Explainer
        SHAP explainer for model interpretability.

    Examples
    --------
    .. code:: python

        from bsix.models.metodologies import CoxRegression
        model = CoxRegression(alpha=0.1, ties="efron", n_iter=200)
        model.fit(X_train, y_train)
    """

    def __init__(self, alpha=0.0, ties="breslow", n_iter=100):

        """
        Initialise model with specified parameters.
        """

        # Parameters
        self.alpha = alpha
        self.ties = ties
        self.n_iter = n_iter
        
        self.coef_ = None
        self.breslow = None

        self.labels_covariables = ["event", "time"]

    def fit(self, X, y):

        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : structured array-like, shape (n_samples,)
            Target training values (events, times).

        Returns
        -------
        self : CoxRegression
            Fitted estimator.
        """

        # Sort by time
        X, y = self._sort(X, y)
        events = y["event"]
        times = y["time"]
        
        n_features = X.shape[1]
        distinct_times = np.unique(times[events])

        self.coef_ = np.zeros(n_features)
        
        # Newton-Raphson algorithm
        for _ in range(self.n_iter):
            risk = np.dot(X, self.coef_)
            # Prevent overflow in exp by clipping the risk
            risk = np.clip(risk, -250, 250) 
            log_risk = np.exp(risk)
            
            gradient = np.zeros(n_features)
            hessian = np.zeros((n_features, n_features))
            
            for t in distinct_times:
                risk_set = times >= t
                events_t = (times == t) & events
                d_i = np.sum(events_t)
                
                X_risk = X[risk_set]
                risk_risk = log_risk[risk_set]
                
                sum_risk = np.sum(risk_risk) + 1e-15 
                    
                sum_X_risk = np.sum(X_risk * risk_risk[:, None], axis=0)
                sum_X_events = np.sum(X[events_t], axis=0)
                
                XX_risk = np.dot(X_risk.T, X_risk * risk_risk[:, None])

                if self.ties == "efron" and d_i > 1:
                    sum_risk_ties = np.sum(log_risk[events_t])
                    sum_X_ties = np.sum(X[events_t] * log_risk[events_t][:, None], axis=0)
                    XX_ties = np.dot(X[events_t].T, X[events_t] * log_risk[events_t][:, None])
                    
                    grad_term = np.zeros(n_features)
                    hess_term = np.zeros((n_features, n_features))
                    
                    for j in range(d_i):
                        fraction = j / d_i
                        den = (sum_risk - fraction * sum_risk_ties) + 1e-15
                        num = sum_X_risk - fraction * sum_X_ties
                        grad_term += num / den
                        
                        num2 = XX_risk - fraction * XX_ties
                        term1 = num2 / den
                        term2 = np.outer(num, num) / (den ** 2)
                        hess_term += (term1 - term2)
                        
                    gradient += sum_X_events - grad_term
                    hessian -= hess_term    
                else: # Breslow approximation
                    gradient += sum_X_events - d_i * (sum_X_risk / sum_risk)
                    term1 = XX_risk / sum_risk
                    term2 = np.outer(sum_X_risk, sum_X_risk) / (sum_risk ** 2)
                    hessian -= d_i * (term1 - term2)
                
            if self.alpha > 0:
                gradient -= self.alpha * self.coef_
                hessian -= self.alpha * np.eye(n_features)
            
            # Solve for parameter updates
            try:
                delta = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                delta = np.linalg.solve(hessian - 1e-6 * np.eye(n_features), -gradient)
                
            # Prevent Newton-Raphson jumps to NaN
            if np.any(np.isnan(delta)):
                logging.warning("Convergence Warning: NaN values in delta.")
                break
                
            self.coef_ += delta
            
            # Convergence criteria
            if np.max(np.abs(delta)) < 1e-6:
                break
        
        # Breslow estimator for baseline hazards
        self.breslow = BreslowEstimator()
        self.breslow.fit(self.predict(X), y["event"], y["time"])
        
        return self

    def predict(self, X):

        """
        Predict risk scores for the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        risk : array-like, shape (n_samples,)
            Predicted risk scores.
        """
        
        risk = np.dot(X, self.coef_)

        return risk
    
    def score(self, X, y):

        return None
    
    # ----------------------
    # Base Survival methods
    # ----------------------
    def predict_survival_function(self, X, index, dataset, seed, plot=False):

        """ 
        Predict the survival function for the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        index : array-like, shape (n_samples,)
            Index for the samples.
        dataset : str
            Name of the dataset.
        seed : int
            Random seed for reproducibility.
        plot : bool, default = ``False``
            Whether to plot the survival function.

        Returns
        -------
        survival_function : array-like, shape (n_samples, n_times)
            Predicted survival function.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_survival_function` with a model, the seed must be an integer. Value received: {seed}")
        
        risk = self.predict(X)

        self.survival_function = self.breslow.get_survival_function(risk)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.survival_function, index, "Cox Regression", dataset, "Survival", seed)
            plt.show()
            
        return self.survival_function

    def predict_cumulative_hazard_function(self, X, index, dataset, seed, plot=False):
        
        """
        Predict the cumulative hazard function for the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        index : array-like, shape (n_samples,)
            Index for the samples.
        dataset : str
            Name of the dataset.
        seed : int
            Random seed for reproducibility.
        plot : bool, default = ``False``
            Whether to plot the cumulative hazard function.

        Returns
        -------
        cumulative_hazard_function : array-like, shape (n_samples, n_times)
            Predicted cumulative hazard function.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_cumulative_hazard_function` with a model, the seed must be an integer. Value received: {seed}")

        risk = self.predict(X)
        
        self.cumulative_hazard_function = self.breslow.get_cumulative_hazard_function(risk)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.cumulative_hazard_function, index, "Cox Regression", dataset, "CumulativeRisk", seed)
            plt.show()
        
        return self.cumulative_hazard_function
    
    # ----------------------
    # XAI
    # ----------------------
    def calculate_xai(self, X, index, scaler, dataset, seed, feature_names, background=False, plot=False):

        """
        Calculate XAI values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        index : array-like, shape (n_samples,)
            Index for the samples.
        scaler : object
            Scaler used for the data.
        dataset : str
            Name of the dataset.
        seed : int
            Random seed for reproducibility.
        feature_names : list of str
            Names of the features.
        background : bool, default = ``False``
            Whether to use background data for SHAP.
        plot : bool, default = ``False``
            Whether to plot the XAI values.

        Returns
        -------
        shap_explainer : shap.Explainer
            SHAP explainer for model interpretability.
        coefficients : dict
            Dictionary of feature coefficients sorted by absolute value.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `calculate_xai` with a model, the seed must be an integer. Value received: {seed}")
        
        logging.getLogger("xai").setLevel(logging.WARNING)

        # Applying Explainer (model type)
        masker = shap.maskers.Independent(X, max_samples=X.shape[0])
        explainer_risk = shap.Explainer(self.predict, masker, feature_names=feature_names, seed=seed)
        
        # Background (faster)
        X_background = X.copy()
        if background:
            X_background = pd.DataFrame(shap.kmeans(X, background).data, columns=feature_names)

        self.shap_explainer = explainer_risk(X_background)

        coefficients = {feature_names[i]: round(coef, 8) for i, coef in enumerate(self.coef_)}
        self.coefficients = {k: v for k, v in sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)}

        if plot:
            figure, ax = BaseSurvival.plot_coefficients(self.coefficients, "Cox Regression", dataset, seed)
            figure, ax = BaseSurvival.plot_shap(self.shap_explainer, index, scaler, "Cox Regression", dataset, seed)
            
            plt.show()

        return self.shap_explainer, self.coefficients