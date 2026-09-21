import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shap
 
from ..base import BaseSurvival
from .utils import BreslowEstimator

from scipy.stats import rankdata
from tabpfn import TabPFNClassifier, TabPFNRegressor

warnings.filterwarnings("ignore")

ruta_relativa_clf = "src/bsix/models/metodologies/tabpfn-v2-classifier.ckpt"
ruta_relativa_reg = "src/bsix/models/metodologies/tabpfn-v2-regressor.ckpt"
ruta_absoluta_clf = os.path.abspath(ruta_relativa_clf)
ruta_absoluta_reg = os.path.abspath(ruta_relativa_reg)

class SurvivalTabPFN(BaseSurvival):
 
    """
    Probabilistic survival ensemble based on TabPFN models.

    This estimator combines a classifier that estimates the event probability
    with a regressor trained on event-time ranks to produce a risk score used in
    a survival-analysis workflow. The resulting risk is then converted into
    survival and cumulative hazard curves through a Breslow baseline estimator.

    Parameters
    ----------
    n_estimators : int, default=1
        Number of TabPFN estimators to use for the underlying models.
    classifier : object, optional
        Pretrained or user-provided TabPFN classifier. If ``None``, a new
        classifier is created during ``fit``.
    regressor : object, optional
        Pretrained or user-provided TabPFN regressor. If ``None``, a new
        regressor is created during ``fit``.
    seed : int, default=0
        Random seed used for reproducibility.
    n_jobs : int, default=-1
        Number of worker threads used by TabPFN preprocessing and fitting.

    Attributes
    ----------
    breslow : BreslowEstimator
            Estimator used to compute the baseline hazard and survival functions.
    survival_function : ndarray of shape (n_samples, n_times)
            Estimated survival function for each sample.
    cumulative_hazard_function : ndarray of shape (n_samples, n_times)
        Estimated cumulative hazard function for each sample.
    shap_explainer : shap.Explainer
        SHAP explainer used to interpret the model output.

    Notes
    -----
    The model combines an event classifier and a time-ranking regressor to
    build a risk score that captures both the probability of observing an event
    and the relative ordering of event times. This risk estimate is then mapped
    to survival and cumulative hazard curves through the Breslow estimator.

    Examples
    --------
    >>> from bsix.models.metodologies import SurvivalTabPFN
    >>> model = SurvivalTabPFN(n_estimators=1)
    >>> model.fit(X_train, y_train)
    >>> risk = model.predict(X_test)
    """
 
    def __init__(self, n_estimators=1, classifier=None, regressor=None, seed=0, n_jobs=-1):
 
        """
        Initialize the SurvivalTabPFN model.

        Parameters
        ----------
        n_estimators : int, default=1
            Number of TabPFN estimators used for the classifier and regressor.
        classifier : object, optional
            Pretrained or user-provided classifier instance. If omitted, a
            `TabPFNClassifier` is created during fitting.
        regressor : object, optional
            Pretrained or user-provided regressor instance. If omitted, a
            `TabPFNRegressor` is created during fitting.
        seed : int, default=0
            Random seed for reproducibility.
        n_jobs : int, default=-1
            Number of jobs used by TabPFN for preprocessing and fitting.
        """
 
        # Parameters for the ensemble
        self.n_estimators = n_estimators
        self.classifier = classifier
        self.regressor = regressor
 
        self.seed = seed
        self.n_jobs = n_jobs
 
        self.labels_covariables = ["event", "time"]
    
    def fit(self, X, y):
 
        """
        Fit the SurvivalTabPFN ensemble to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : structured array-like of shape (n_samples,)
            Target values containing the fields ``event`` and ``time``.

        Returns
        -------
        SurvivalTabPFN
            The fitted estimator instance.
        """
 
        # Sort by time
        X, y = self._sort(X, y)
        
        event = y["event"].astype(bool)
        time = y["time"].astype(np.float32)

        if self.classifier is None:
            # Model
            self.classifier = TabPFNClassifier(n_estimators=self.n_estimators, random_state=self.seed, ignore_pretraining_limits=True, model_path=ruta_absoluta_clf, show_progress_bar=True, n_preprocessing_jobs=self.n_jobs)

        self.classifier.fit(X, event)

        X_event = X[event]
        y_time_event = time[event]
        reversed_y_time_event = - y_time_event
        y_ranked_risk = (rankdata(reversed_y_time_event) - 1) / (reversed_y_time_event.shape[0] - 1)

        if self.regressor is None:
            # Model
            self.regressor = TabPFNRegressor(n_estimators=self.n_estimators ,random_state=self.seed, ignore_pretraining_limits=True, model_path=ruta_absoluta_reg, show_progress_bar=True, n_preprocessing_jobs=self.n_jobs)

        self.regressor.fit(X_event, y_ranked_risk)

        self.breslow = BreslowEstimator()
        self.breslow.fit(self.predict(X), event, time)
 
        return self
 
    def predict(self, X):
 
        """
        Predict risk scores for the given input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for which risk predictions are required.

        Returns
        -------
        ndarray of shape (n_samples,)
            Estimated risk scores for each sample.
        """
        
        p_a = self.classifier.predict_proba(X)[:, 1]
        p_b = self.regressor.predict(X)
        risk = p_a * p_b

        return risk
 
    # ----------------------
    # Base Survival methods
    # ----------------------
    def predict_survival_function(self, X, index, dataset, seed, plot=False):
 
        """
        Predict the survival function for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix used to estimate survival probabilities.
        index : array-like of shape (n_samples,)
            Sample indices used for plotting and identification.
        dataset : str
            Dataset name used when generating the visualization.
        seed : int
            Random seed for deterministic plotting behavior.
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
            figure, ax = self._plot_survival_hazard_functions(self.survival_function, index, "SurvivalTabPFN", dataset, "Survival", seed)
            plt.show()
 
        return self.survival_function
 
    def predict_cumulative_hazard_function(self, X, index, dataset, seed, plot=False):
 
        """
        Predict the cumulative hazard function for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix used to estimate cumulative hazard values.
        index : array-like of shape (n_samples,)
            Sample indices used for plotting and identification.
        dataset : str
            Dataset name used when generating the visualization.
        seed : int
            Random seed for deterministic plotting behavior.
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
            figure, ax = self._plot_survival_hazard_functions(self.cumulative_hazard_function, index, "SurvivalTabPFN", dataset, "CumulativeRisk", seed)
            plt.show()
 
        return self.cumulative_hazard_function
 
    # ----------------------
    # XAI
    # ----------------------
    def calculate_xai(self, X, index, scaler, dataset, seed, feature_names, background=None, plot=False):
 
        """
        Compute SHAP-based explainability values for the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix used to compute SHAP explanations.
        index : array-like of shape (n_samples,)
            Sample indices used for plotting and identification.
        scaler : object
            Preprocessing scaler used in the pipeline, if any.
        dataset : str
            Name of the dataset used in the generated visualization.
        seed : int
            Random seed for reproducibility.
        feature_names : list of str
            Names of the model features.
        background : int or None, default=None
            Number of k-means clusters used to summarize the SHAP background.
            If ``None`` or ``0``, the full dataset is used as the background.
        plot : bool, default=False
            If ``True``, display the SHAP explanation plot.

        Returns
        -------
        shap.Explainer
            SHAP explainer containing the explanations computed for the input
            samples.
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
 
        if plot:
            figure, ax = BaseSurvival.plot_shap(self.shap_explainer, index, scaler, "SurvivalTabPFN", dataset, seed)
            plt.show()
 
        return self.shap_explainer
