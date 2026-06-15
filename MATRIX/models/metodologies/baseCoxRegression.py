import logging
import matplotlib.pyplot as plt
import pandas as pd
import shap
import warnings

from ..base import BaseSurvival
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.linear_model.coxph import BreslowEstimator

warnings.filterwarnings("ignore")

class BaseCoxRegression(BaseSurvival):

    """
    Cox Regression model.
    """

    def __init__(self, alpha=0.0, ties="breslow", n_iter=100):

        """
        Initialise model with specified parameters.
        """

        # Parameters 
        self.alpha = alpha
        self.ties = ties
        self.n_iter = n_iter

        # Model (will be initialized in train())
        self.model = None

        self.labels_covariables = ["event", "time"]

    def fit(self, X, y):

        """
        Fit the model to the data.
        """
                
        # Breslow estimator for baseline hazards
        self.breslow = BreslowEstimator()

        # Sort by time
        X, y = self._sort(X, y)

        self.model = CoxPHSurvivalAnalysis(alpha=self.alpha, ties=self.ties, n_iter=self.n_iter)
        self.model.fit(X, y)

        # Compute baseline hazards with training data
        self.breslow.fit(self.predict(X), y["event"], y["time"])
        
        self.coef_ = self.model.coef_
        
        return self

    def predict(self, X):

        """
        Predict risk scores for the given data.
        """
                
        risk = self.model.predict(X)

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

        self.survival_function = self.breslow.get_survival_function(risk)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.survival_function, index, "BaseCoxRegression", dataset, "Survival", seed)
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
        
        self.cumulative_hazard_function = self.breslow.get_cumulative_hazard_function(risk)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.cumulative_hazard_function, index, "BaseCoxRegression", dataset, "CumulativeRisk", seed)
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
            figure, ax = BaseSurvival.plot_coefficients(self.coefficients, "BaseCoxRegression", dataset, seed)
            figure, ax = BaseSurvival.plot_shap(self.shap_explainer, index, scaler, "BaseCoxRegression", dataset, seed)
            
            plt.show()

        return self.shap_explainer, self.coefficients