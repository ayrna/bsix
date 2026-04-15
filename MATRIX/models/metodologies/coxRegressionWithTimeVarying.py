import logging
import numpy as np
import pandas as pd
import shap
import warnings

from ..base import BaseSurvival
from lifelines import CoxTimeVaryingFitter
from sksurv.linear_model.coxph import BreslowEstimator

warnings.filterwarnings("ignore")

class CoxRegressionWithTimeVarying(BaseSurvival):

    """
    Cox Regression with Time-Varying Covariates model.
    """

    def __init__(self, penalizer=0.0, l1_ratio=0.0, formula=None):

        """
        Initialise model with specified parameters.
        """

        # Parameters
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.formula = formula

        # Model (will be initialized in train())
        self.model = None

        self.labels_covariables = ["event", "time_start", "time_stop"]

    def _toDataframe(self, data, columns=None):

        """
        Convert data to DataFrame format.
        """

        if columns == None:
            dataframe = pd.DataFrame(data, columns=[str(l) for l in range(data.shape[1])])
        else:
            dataframe = pd.DataFrame(data, columns=columns)

        return dataframe
    
    def fit(self, X, y):

        """
        Fit the model to the data.
        """

        # Breslow estimator for baseline hazards
        self.breslow = BreslowEstimator()

        # Sort by time
        X, y = self._sort(X, y, "time_stop")

        dataframe = pd.concat([self._toDataframe(X), self._toDataframe(y[["event", "time_start", "time_stop"]], self.labels_covariables)], axis=1)
        dataframe["time_stop"] = np.where(dataframe["time_start"] == dataframe["time_stop"], dataframe["time_stop"] + 1e-6, dataframe["time_stop"])

        self.model = CoxTimeVaryingFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.model.fit(dataframe, id_col=dataframe.columns[0], start_col=self.labels_covariables[1], stop_col=self.labels_covariables[2], event_col=self.labels_covariables[0], show_progress=False)
        
        # Compute baseline hazards with training data
        self.breslow.fit(self.predict(X), y["event"], y["time_stop"])
        
        self.coef_ = self.model.params_
        
        return self

    def predict(self, X):

        """
        Predict risk scores for the given data.
        """

        risk = self.model.predict_log_partial_hazard(self._toDataframe(X)).to_numpy()

        return risk
    
    def score(self, X, y):

        """
        Calculate the score for the model.
        """
        
        return None
    
    # ----------------------
    # Base Survival methods
    # ----------------------
    def predict_survival_function(self, X, estimator_name, dataset, seed):

        """ 
        S(x, t) = exp(-H(x, t)).
        """

        risk = self.predict(X)
        
        survival_function = self.breslow.get_survival_function(risk)
        self._plot_survival_hazard_functions(survival_function, estimator_name, dataset, seed, "Survival")

        return survival_function

    def predict_cumulative_hazard_function(self, X, estimator_name, dataset, seed):
        
        """
        H(x,t) = H₀(t) × exp(βᵀx).
        """

        risk = self.predict(X)
        
        get_cumulative_hazard_function = self.breslow.get_cumulative_hazard_function(risk)
        self._plot_survival_hazard_functions(get_cumulative_hazard_function, estimator_name, dataset, seed, "CumulativeRisk")
        
        return get_cumulative_hazard_function
    
    # ----------------------
    # XAI
    # ----------------------
    def calculate_xai(self, X, estimator_name, dataset, seed, feature_names, background=False):

        """
        Calculate XAI values.
        """

        logging.getLogger("xai").setLevel(logging.WARNING)

        # Applying Explainer (model type)
        explainer_risk = shap.Explainer(self.predict, X, feature_names=feature_names, seed=seed)
        
        # Background (faster)
        X_background = X.copy()
        if background:
            X_background = pd.DataFrame(shap.kmeans(X, background).data, columns=feature_names)

        self.shap_explainer = explainer_risk(X_background)

        BaseSurvival.plot_shap(self.shap_explainer, estimator_name, dataset, seed)

        coefficients = {feature_names[i]: round(coef, 8) for i, coef in enumerate(self.coef_)}
        self.coefficients = {k: v for k, v in sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)}

        BaseSurvival.plot_coefficients(self.coefficients, estimator_name, dataset, seed)