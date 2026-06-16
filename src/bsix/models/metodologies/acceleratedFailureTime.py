import logging
import matplotlib.pyplot as plt
import pandas as pd
import shap
import warnings

from ..base import BaseSurvival
from lifelines import LogLogisticAFTFitter, WeibullAFTFitter

warnings.filterwarnings("ignore")

class AcceleratedFailureTime(BaseSurvival):

    """
    Weibull Accelerated Failure Time model.
    """

    def __init__(self, type="WeibullAFT", penalizer=0.0, l1_ratio=0.0):

        """
        Initialise model with specified parameters.
        """

        # Parameters
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.type = type

        # Model (will be initialized in train())
        self.model = None

        self.labels_covariables = ["event", "time"]

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

        # Sort by time
        X, y = self._sort(X, y, "time")

        dataframe = pd.concat([self._toDataframe(X), self._toDataframe(y[["event", "time"]], self.labels_covariables)], axis=1)

        if self.type == "LogLogisticAFT":
            self.model = LogLogisticAFTFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        else:
            self.model = WeibullAFTFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)

        self.model.fit(dataframe, duration_col=self.labels_covariables[1], event_col=self.labels_covariables[0], show_progress=False, fit_options={"step_size": 0.15})
        
        self.coef_ = self.model.params_.values[:X.shape[1]]
        
        return self

    def predict(self, X):

        """
        Predict risk scores for the given data.
        """

        risk = self.model.predict_expectation(self._toDataframe(X)).to_numpy() * -1

        return risk
    
    def score(self, X, y):

        """
        Calculate the score for the model.
        """
        
        return None
    
    # ----------------------
    # Base Survival methods
    # ----------------------
    def predict_survival_function(self, X, index, dataset, seed, plot=False):

        """ 
        S(x, t) = exp(-H(x, t)).
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_survival_function` with a model, the seed must be an integer. Value received: {seed}")
        
        risk = self.predict(X)

        self.survival_function = self.model.predict_survival_function(self._toDataframe(X))

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.survival_function, index, "Accelerated Failure Time", dataset, "Survival", seed)
            plt.show()

        return self.survival_function

    def predict_cumulative_hazard_function(self, X, index, dataset, seed, plot=False):
        
        """
        H(x,t) = H₀(t) × exp(βᵀx).
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_cumulative_hazard_function` with a model, the seed must be an integer. Value received: {seed}")

        risk = self.predict(X)
        
        self.cumulative_hazard_function = self.model.predict_cumulative_hazard(self._toDataframe(X))

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.cumulative_hazard_function, index, "Accelerated Failure Time", dataset, "CumulativeRisk", seed)
            plt.show()
        
        return self.cumulative_hazard_function
    
    # ----------------------
    # XAI
    # ----------------------
    def calculate_xai(self, X, index, scaler, dataset, seed, feature_names, background=False, plot=False):

        """
        Calculate XAI values.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `calculate_xai` with a model, the seed must be an integer. Value received: {seed}")
        
        logging.getLogger("xai").setLevel(logging.WARNING)

        # Applying Explainer (model type)
        explainer_risk = shap.Explainer(self.predict, X, feature_names=feature_names, seed=seed)
        
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