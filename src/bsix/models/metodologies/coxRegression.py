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
    Cox proportional hazards model estimated via a Newton-Raphson optimization.

    This implementation fits a Cox proportional hazards model by maximizing the
    partial log-likelihood over the observed covariates. It supports tied event
    times through either the Breslow or Efron approximation and reconstructs the
    baseline survival and cumulative hazard curves from the fitted risk scores.

    Parameters
    ----------
    alpha : float, default=0.0
        L2 regularization strength applied to the coefficients.
    ties : {"breslow", "efron"}, default="breslow"
        Method used to handle tied event times.
    n_iter : int, default=100
        Maximum number of Newton-Raphson iterations used during optimization.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated regression coefficients for each feature.
    breslow : BreslowEstimator
        Estimator used to compute the baseline hazard and survival functions.
    survival_function : ndarray of shape (n_samples, n_times)
        Estimated survival function for each sample.
    cumulative_hazard_function : ndarray of shape (n_samples, n_times)
        Estimated cumulative hazard function for each sample.
    shap_explainer : shap.Explainer
        SHAP explainer used to interpret the model output.
    coefficients : dict
        Feature coefficients sorted by absolute magnitude for explainability.

    Notes
    -----
    The model assumes a proportional hazards structure, i.e. the hazard ratio
    between two observations is constant over time after conditioning on the
    covariates. The risk score is computed as ``X @ coef_`` and is later mapped
    to survival curves through the Breslow estimator.

    Examples
    --------
    >>> from bsix.models.metodologies import CoxRegression
    >>> model = CoxRegression(alpha=0.1, ties="efron", n_iter=200)
    >>> model.fit(X_train, y_train)
    >>> risk = model.predict(X_test)
    """

    def __init__(self, alpha=0.0, ties="breslow", n_iter=100):

        """
        Initialize the Cox regression model.

        Parameters
        ----------
        alpha : float, default=0.0
            L2 regularization strength applied to the coefficients.
        ties : {"breslow", "efron"}, default="breslow"
            Method used to handle tied event times.
        n_iter : int, default=100
            Maximum number of Newton-Raphson iterations used during optimization.
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
        Fit the Cox proportional hazards model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : structured array-like of shape (n_samples,)
            Survival target with at least the fields ``event`` and ``time``.

        Returns
        -------
        CoxRegression
            The fitted estimator instance.
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
        Predict relative risk scores for a set of samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for which to compute the risk score.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted risk score for each sample.
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
        Predict the survival function for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for prediction.
        index : array-like of shape (n_samples,)
            Sample indices used for plotting and identification.
        dataset : str
            Name of the dataset used in the generated plot.
        seed : int
            Random seed used for reproducibility in plotting.
        plot : bool, default=False
            If ``True``, display the survival-function plot.

        Returns
        -------
        ndarray of shape (n_samples, n_times)
            Estimated survival function for each sample.
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
        Predict the cumulative hazard function for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for prediction.
        index : array-like of shape (n_samples,)
            Sample indices used for plotting and identification.
        dataset : str
            Name of the dataset used in the generated plot.
        seed : int
            Random seed used for reproducibility in plotting.
        plot : bool, default=False
            If ``True``, display the cumulative hazard plot.

        Returns
        -------
        ndarray of shape (n_samples, n_times)
            Estimated cumulative hazard function for each sample.
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
        Compute SHAP-based explainability values and coefficient ranking.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix used for explanation.
        index : array-like of shape (n_samples,)
            Sample indices used for plotting.
        scaler : object
            Data scaler used before model fitting, if any.
        dataset : str
            Name of the dataset used in the generated visualization.
        seed : int
            Random seed for reproducibility.
        feature_names : list of str
            Names of the model features.
        background : bool, default=False
            If ``True``, compute the SHAP background using k-means summary data.
        plot : bool, default=False
            If ``True``, display the coefficient and SHAP plots.

        Returns
        -------
        shap.Explainer
            SHAP explainer for the fitted model.
        dict
            Feature coefficients sorted by absolute value.
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