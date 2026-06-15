import logging
import matplotlib.pyplot as plt
import pandas as pd
import shap
import warnings

from ..base import BaseSurvival
from sksurv.tree import SurvivalTree

warnings.filterwarnings("ignore")

class BaseSurvivalTree(BaseSurvival):

    """
    Survival Tree model.
    """

    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):

        """
        Initialise model with specified parameters.
        """

        # Parameters 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        # Model (will be initialized in train())
        self.model = None

        self.labels_covariables = ["event", "time"]

    def fit(self, X, y):

        """
        Fit the model to the data.
        """

        # Sort by time
        X, y = self._sort(X, y)

        self.model = SurvivalTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
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
    def predict_survival_function(self, X, index, dataset, seed, plot=False):

        """ 
        S(x, t) = exp(-H(x, t)).
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_survival_function` with a model, the seed must be an integer. Value received: {seed}")

        self.survival_function = self.model.predict_survival_function(X)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.survival_function, index, "BaseSurvivalTree", dataset, "Survival", seed)
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
        
        self.cumulative_hazard_function = self.model.predict_cumulative_hazard_function(X)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.cumulative_hazard_function, index, "BaseSurvivalTree", dataset, "CumulativeRisk", seed)
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

        if plot:
            figure, ax = BaseSurvival.plot_shap(self.shap_explainer, index, scaler, "BaseSurvivalTree", dataset, seed)
            
            plt.show()

        return self.shap_explainer