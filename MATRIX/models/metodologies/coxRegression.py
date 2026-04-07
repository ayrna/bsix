import logging
import pandas as pd
import shap
import warnings

from ..base import BaseSurvival
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.linear_model.coxph import BreslowEstimator

warnings.filterwarnings("ignore")

class CoxRegression(BaseSurvival):

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