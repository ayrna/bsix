import logging
import matplotlib.pyplot as plt
import pandas as pd
import shap
import warnings

from ..base import BaseSurvival

from sksurv.ensemble import RandomSurvivalForest

warnings.filterwarnings("ignore")

class BaseRandomSurvivalForest(BaseSurvival):

    """
    Random Survival Forest model.
    """

    def __init__(self, seed, n_jobs=-1, n_estimators=100, max_depth=None, min_samples_leaf=3, min_samples_split=6):

        """
        Initialise model with specified parameters.
        """
        
        # Parameters
        self.n_jobs=n_jobs
        self.seed=seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

        # Model (will be initialized in train())
        self.model = None

    def fit(self, X, y):

        """
        Fit the model to the data.
        """
                
        # Sort by time
        X, y = self._sort(X, y)

        self.model = RandomSurvivalForest(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split, n_jobs=self.n_jobs, random_state=self.seed)
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
            figure, ax = self._plot_survival_hazard_functions(self.survival_function, index, "Random Survival Forest", dataset, "Survival", seed)
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
            figure, ax = self._plot_survival_hazard_functions(self.cumulative_hazard_function, index, "Random Survival Forest", dataset, "CumulativeRisk", seed)
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
        masker = shap.maskers.Independent(X, max_samples=X.shape[0])
        explainer_risk = shap.Explainer(self.predict, masker, feature_names=feature_names, seed=seed)
        
        # Background (faster)
        X_background = X.copy()
        if background:
            X_background = pd.DataFrame(shap.kmeans(X, background).data, columns=feature_names)

        self.shap_explainer = explainer_risk(X_background)

        if plot:
            figure, ax = BaseSurvival.plot_shap(self.shap_explainer, index, scaler, "Random Survival Forest", dataset, seed)
            plt.show()

        return self.shap_explainer