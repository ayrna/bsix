import logging
import pandas as pd
import shap
import warnings

from ..base import BaseSurvival
from sksurv.ensemble import RandomSurvivalForest

warnings.filterwarnings("ignore")

class RandomSurvForest(BaseSurvival):

    """
    Random Survival Forest model.
    """

    def __init__(self, random_state, n_jobs=-1, n_estimators=100, max_depth=None, min_samples_split=6):

        """
        Initialise model with specified parameters.
        """
        
        # Parameters
        self.n_jobs=n_jobs
        self.random_state=random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        # Model (will be initialized in train())
        self.model = None

    def fit(self, X, y):

        """
        Fit the model to the data.
        """
                
        # Sort by time
        X, y = self._sort(X, y)

        self.model = RandomSurvivalForest(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_jobs=self.n_jobs, random_state=self.random_state)
        self.model.fit(X, y)
        
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
        S(x, t) = exp(-H(x, t)) 
        """

        survival_function = self.model.predict_survival_function(X)
        self._plot_survival_hazard_functions(survival_function, estimator_name, dataset, seed, "Survival")

        return survival_function

    def predict_cumulative_hazard_function(self, X, estimator_name, dataset, seed):
        
        """
        H(x,t) = H₀(t) × exp(βᵀx)
        """

        get_cumulative_hazard_function = self.model.predict_cumulative_hazard_function(X)
        self._plot_survival_hazard_functions(get_cumulative_hazard_function, estimator_name, dataset, seed, "CumulativeRisk")
        
        return get_cumulative_hazard_function
    
    # ----------------------
    # XAI
    # ----------------------
    def calculate_xai(self, X, estimator_name, dataset, seed, feature_names, background=False):

        """
        Calculate XAI values
        """

        logging.getLogger('xai').setLevel(logging.WARNING)

        # Applying Explainer (model type)
        explainer_risk = shap.Explainer(self.predict, X, feature_names=feature_names, seed=seed)
        
        # Background (faster)
        X_background = X.copy()
        if background:
            X_background = pd.DataFrame(shap.kmeans(X, background).data, columns=feature_names)

        self.shap_explainer = explainer_risk(X_background)

        BaseSurvival.plot_shap(self.shap_explainer, estimator_name, dataset, seed)
