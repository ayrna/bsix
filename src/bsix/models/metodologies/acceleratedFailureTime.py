import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import warnings

from ..base import BaseSurvival
from .utils import StepFunction

from scipy.optimize import minimize
from scipy.special import gamma

warnings.filterwarnings("ignore")

class AcceleratedFailureTime(BaseSurvival):

    """
    Accelerated failure time (AFT) survival model for Weibull and log-logistic distributions.

    This model fits a parametric survival regression where the logarithm of time is modeled
    as a linear combination of the covariates plus a distribution-specific shape parameter.

    Parameters
    ----------
    type : {"WeibullAFT", "LogLogisticAFT"}, default="WeibullAFT"
        Distribution used for the accelerated failure time formulation.
    penalizer : float, default=0.0
        Global regularization strength applied to the regression coefficients.
    l1_ratio : float, default=0.0
        Relative weight of the L1 penalty versus the L2 penalty. Values close to 0.0
        indicate ridge regularization, while values close to 1.0 indicate lasso-like
        sparsity.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated regression coefficients for the covariates.
    intercept_ : float
        Estimated intercept of the linear predictor.
    shape_ : float
        Estimated shape parameter of the selected AFT distribution.
    params_ : ndarray of shape (n_features + 2,)
        Full optimizer parameter vector containing coefficients, intercept, and log-shape.
    time_grid : ndarray
        Sorted unique observed times used to evaluate survival and hazard curves.
    labels_covariables : list of str
        Labels used by the event/time structured output.

    Notes
    -----
    The model supports elastic-net regularization on the coefficients and exposes survival,
    cumulative hazard, and SHAP-based explanations.

    Examples
    --------
    >>> from bsix.models.metodologies import AcceleratedFailureTime
    >>> model = AcceleratedFailureTime(type="WeibullAFT", penalizer=0.0, l1_ratio=0.0)
    >>> model.fit(X_train, y_train)
    >>> risk = model.predict(X_test)
    """

    def __init__(self, type="LogLogisticAFT", penalizer=0.0, l1_ratio=0.0):

        """
        Initialize the accelerated failure time model.

        Parameters
        ----------
        type : {"WeibullAFT", "LogLogisticAFT"}, default="WeibullAFT"
            Distribution used by the AFT model.
        penalizer : float, default=0.0
            Regularization strength applied to the coefficients.
        l1_ratio : float, default=0.0
            Fraction of the penalty assigned to the L1 term; the remaining fraction is
            assigned to the L2 term.
        """

        # Parameters
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.type = type

        # Fitted parameters (populated in fit())
        self.coef_ = None
        self.intercept_ = None
        self.shape_ = None
        self.params_ = None
        self.time_grid = None

        self.labels_covariables = ["event", "time"]

    def _linear_predictor(self, params, X):

        """
        Compute the linear predictor for the AFT scale parameter.

        Parameters
        ----------
        params : ndarray of shape (n_features + 2,)
            Optimizer parameters containing the coefficients, intercept, and log-shape.
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Linear predictor η(x) = Xβ + intercept used in the scale submodel.
        """

        n_features = X.shape[1]
        beta = params[:n_features]
        intercept = params[n_features]

        return X @ beta + intercept

    def _neg_log_likelihood_weibull(self, params, X, time, event):

        """
        Compute the negative log-likelihood for a Weibull AFT model.

        Parameters
        ----------
        params : ndarray of shape (n_features + 2,)
            Parameter vector containing regression coefficients, intercept, and log-shape.
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.
        time : ndarray of shape (n_samples,)
            Observed time-to-event values.
        event : ndarray of shape (n_samples,)
            Event indicator, where 1 denotes an observed event and 0 denotes censoring.

        Returns
        -------
        float
            Negative log-likelihood of the Weibull AFT model with elastic-net regularization.
        """

        n_features = X.shape[1]
        beta = params[:n_features]

        Xb = self._linear_predictor(params, X)
        rho = np.exp(params[n_features + 1])
        log_t = np.log(time)

        log_hazard = np.log(rho) + (rho - 1.0) * log_t - rho * Xb
        cumulative_hazard = np.exp(rho * (log_t - Xb))

        log_likelihood = np.sum(event * log_hazard - cumulative_hazard)

        penalty = self.penalizer * (
            (1.0 - self.l1_ratio) * 0.5 * np.sum(beta ** 2) + self.l1_ratio * np.sum(np.abs(beta))
        )

        return -log_likelihood + penalty

    def _neg_log_likelihood_loglogistic(self, params, X, time, event):

        """
        Compute the negative log-likelihood for a log-logistic AFT model.

        Parameters
        ----------
        params : ndarray of shape (n_features + 2,)
            Parameter vector containing regression coefficients, intercept, and log-shape.
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.
        time : ndarray of shape (n_samples,)
            Observed time-to-event values.
        event : ndarray of shape (n_samples,)
            Event indicator, where 1 denotes an observed event and 0 denotes censoring.

        Returns
        -------
        float
            Negative log-likelihood of the log-logistic AFT model with elastic-net regularization.
        """

        n_features = X.shape[1]
        beta = params[:n_features]

        Xb = self._linear_predictor(params, X)
        shape = np.exp(params[n_features + 1])
        log_t = np.log(time)
        z = shape * (log_t - Xb)

        log_hazard = np.log(shape) + (shape - 1.0) * log_t - shape * Xb - np.logaddexp(0.0, z)
        cumulative_hazard = np.logaddexp(0.0, z)

        log_likelihood = np.sum(event * log_hazard - cumulative_hazard)

        penalty = self.penalizer * (
            (1.0 - self.l1_ratio) * 0.5 * np.sum(beta ** 2) + self.l1_ratio * np.sum(np.abs(beta))
        )

        return -log_likelihood + penalty

    def fit(self, X, y):

        """
        Fit the AFT model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : structured array-like of shape (n_samples,)
            Survival target with fields ``event`` and ``time``.

        Returns
        -------
        AcceleratedFailureTime
            The fitted estimator instance.
        """

        # Sort by time
        X, y = self._sort(X, y)
        events = y["event"]
        times = y["time"]

        n_features = X.shape[1]

        # Initial guess: β = 0, intercept = log(mean(time)), shape = 1 (log_shape = 0)
        init_params = np.zeros(n_features + 2)
        init_params[n_features] = np.log(np.mean(times))
        init_params[n_features + 1] = 0.0

        neg_log_likelihood = (
            self._neg_log_likelihood_loglogistic
            if self.type == "LogLogisticAFT"
            else self._neg_log_likelihood_weibull
        )

        result = minimize(
            neg_log_likelihood,
            init_params,
            args=(X, times, events),
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )

        self.params_ = result.x
        self.coef_ = self.params_[:n_features]
        self.intercept_ = self.params_[n_features]
        self.shape_ = np.exp(self.params_[n_features + 1])
        self.time_grid = np.unique(times)

        return self

    def _predict_expectation(self, Xb):

        """
        Compute the conditional expected survival time under the fitted AFT distribution.

        Parameters
        ----------
        Xb : ndarray of shape (n_samples,)
            Linear predictor values associated with each sample.

        Returns
        -------
        ndarray of shape (n_samples,)
            Expected survival time for each sample under the fitted distribution.
        """

        lambda_ = np.exp(Xb)

        if self.type == "LogLogisticAFT":
            shape = self.shape_
            if shape > 1.0:
                expectation = lambda_ * (np.pi / shape) / np.sin(np.pi / shape)
            else:
                # Mean is undefined for shape <= 1; fall back to the median survival time.
                expectation = lambda_
        else:
            rho = self.shape_
            expectation = lambda_ * gamma(1.0 + 1.0 / rho)

        return expectation

    def predict(self, X):

        """
        Predict risk scores for the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix used to compute the model risk score.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted risk values, defined as the negative expected survival time.
        """

        Xb = X @ self.coef_ + self.intercept_

        risk = self._predict_expectation(Xb) * -1

        return risk

    # ----------------------
    # Distribution functions
    # ----------------------
    def _survival_function(self, t, Xb):

        """
        Evaluate the survival function for a set of times under the fitted AFT model.

        Parameters
        ----------
        t : array-like
            Time points at which the survival function is evaluated.
        Xb : ndarray of shape (n_samples,)
            Linear predictor values for the samples.

        Returns
        -------
        ndarray
            Survival probability at the requested time points for each sample.
        """

        lambda_ = np.exp(Xb)

        if self.type == "LogLogisticAFT":
            shape = self.shape_
            return 1.0 / (1.0 + (t / lambda_) ** shape)
        else:
            rho = self.shape_
            return np.exp(-((t / lambda_) ** rho))

    def _cumulative_hazard(self, t, Xb):

        """
        Evaluate the cumulative hazard function for a set of times under the fitted AFT model.

        Parameters
        ----------
        t : array-like
            Time points at which the cumulative hazard is evaluated.
        Xb : ndarray of shape (n_samples,)
            Linear predictor values for the samples.

        Returns
        -------
        ndarray
            Cumulative hazard at the requested time points for each sample.
        """

        lambda_ = np.exp(Xb)

        if self.type == "LogLogisticAFT":
            shape = self.shape_
            return np.log1p((t / lambda_) ** shape)
        else:
            rho = self.shape_
            return (t / lambda_) ** rho

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
        
        Xb = X @ self.coef_ + self.intercept_

        survival = np.stack([self._survival_function(t, Xb) for t in self.time_grid])
        self.survival_function = np.array([StepFunction(X=self.time_grid, y=individual_survival, is_survival=True) for individual_survival in survival])
        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.survival_function, index, "Accelerated Failure Time", dataset, "Survival", seed)
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

        Xb = X @ self.coef_ + self.intercept_

        cumulative_hazard = np.stack([self._cumulative_hazard(t, Xb) for t in self.time_grid])
        self.cumulative_hazard_function = np.array([StepFunction(X=self.time_grid, y=individual_survival, is_survival=True) for individual_survival in cumulative_hazard])
        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.cumulative_hazard_function, index, "Accelerated Failure Time", dataset, "CumulativeRisk", seed)
            plt.show()
        
        return self.cumulative_hazard_function
    
    # ----------------------
    # XAI
    # ----------------------
    def calculate_xai(self, X, index, scaler, dataset, seed, feature_names, background=False, plot=False):

        """
        Compute SHAP-based explanations and coefficient rankings for the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix used for explanation.
        index : array-like of shape (n_samples,)
            Sample indices used for plotting.
        scaler : object
            Scaler used in the preprocessing pipeline, if any.
        dataset : str
            Name of the dataset used in the generated visualization.
        seed : int
            Random seed used for reproducibility.
        feature_names : list of str
            Names of the model features.
        background : bool, default=False
            If ``True``, compute a SHAP background using k-means summary data.
        plot : bool, default=False
            If ``True``, display the SHAP and coefficient plots.

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
            figure, ax = BaseSurvival.plot_coefficients(self.coefficients, "Accelerated Failure Time", dataset, seed)
            figure, ax = BaseSurvival.plot_shap(self.shap_explainer, index, scaler, "Accelerated Failure Time", dataset, seed)
            
            plt.show()

        return self.shap_explainer, self.coefficients
