import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import warnings

from ..base import BaseSurvival
from .survTree import SurvTree
from .utils import StepFunction

from joblib import Parallel, delayed
from sklearn.utils.validation import check_random_state

warnings.filterwarnings("ignore")

class RandomSurvForest(BaseSurvival):

    """
    Random Survival Forest model.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_jobs : int, default =-1
        Number of jobs to run in parallel.
    n_estimators : int, default =100
        The number of trees in the forest.
    max_depth : int, default =´´None´´
        The maximum depth of the tree.
    min_samples_leaf : int, default =3
        The minimum number of samples required to be at a leaf node.
    min_samples_split : int, default =6
        The minimum number of samples required to split an internal node.

    Attributes
    ----------
    survival_function : array-like, shape (n_samples, n_times)
        Estimated survival function.
    cumulative_hazard_function : array-like, shape (n_samples, n_times)
        Estimated cumulative hazard function.
    shap_explainer : shap.Explainer
        SHAP explainer for model interpretability.

    Examples
    --------
    .. code:: python
    
        from bsix.models.metodologies import RandomSurvForest
        model = RandomSurvForest(seed=42, n_estimators=100, max_depth=5)
        model.fit(X_train, y_train)
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

    def _fit_single_tree(self, X, y, n_samples, tree_seed):

        """
        Fit a single survival tree on a bootstrap sample of the data.
        """
        
        random_state_obj = check_random_state(tree_seed)

        # Bagging
        indices = random_state_obj.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Survival tree
        tree = SurvTree(
            max_depth=self.max_depth, 
            min_samples_leaf=self.min_samples_leaf, 
            min_samples_split=self.min_samples_split, 
            seed=tree_seed
        )
        
        # Train survival tree
        tree.fit(X_boot, y_boot)
        return tree

    def fit(self, X, y):

        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : structured array-like, shape (n_samples,)
            Target training values (events, times).

        Returns
        -------
        self : RandomSurvForest
            Fitted estimator.
        """
                
        # Sort by time
        X, y = self._sort(X, y)
        n_samples = X.shape[0]

        # Generate random state object
        random_state_obj = check_random_state(self.seed)
        tree_seeds = random_state_obj.randint(0, np.iinfo(np.int32).max, size=self.n_estimators)
            
        # Parallelize the training of trees
        self.model = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_tree)(X, y, n_samples, tree_seed) 
            for tree_seed in tree_seeds
        )
        
        return self

    def predict(self, X):

        """
        Predict risk scores for the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        risk : array-like, shape (n_samples,)
            Predicted risk scores.
        """
        
        chfs = self._compute_cumulative_hazard_function(X, survival=False)
        risk = np.array([np.sum(chf.y) for chf in chfs])

        return risk
    
    def score(self, X, y):
        
        return None
    
    # ----------------------
    # Base Survival methods
    # ----------------------
    def _compute_cumulative_hazard_function(self, X, survival=False):
        
        """
        Auxiliary method for computing the cumulative hazard function.
        """

        if not self.model:
            raise ValueError(f"When computing `cumulative_hazard_function` with a model, first fit the model.")
            
        all_preds = []
        for tree in self.model:
            all_preds.append(tree._compute_survival_hazard_functions(X, survival))
            
        # Extract all unique time points across all trees
        all_times = np.unique(np.concatenate([fn.X for tree_preds in all_preds for fn in tree_preds]))
        
        n_samples = X.shape[0]
        functions = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            patient_evaluations = np.array([tree_preds[i](all_times) for tree_preds in all_preds])
            mean_y = np.mean(patient_evaluations, axis=0)
            
            # StepFunctions
            functions[i] = StepFunction(X=all_times, y=mean_y, is_survival=survival)
            
        return functions
    
    def predict_survival_function(self, X, index, dataset, seed, plot=False):

        """ 
        Predict the survival function for the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        index : array-like, shape (n_samples,)
            Index for the samples.
        dataset : str
            Name of the dataset.
        seed : int
            Random seed for reproducibility.
        plot : bool, default = ``False``
            Whether to plot the survival function.

        Returns
        -------
        survival_function : array-like, shape (n_samples, n_times)
            Predicted survival function.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_survival_function` with a model, the seed must be an integer. Value received: {seed}")
        
        self.survival_function = self._compute_cumulative_hazard_function(X, survival=True)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.survival_function, index, "Random Survival Forest", dataset, "Survival", seed)
            plt.show()

        return self.survival_function

    def predict_cumulative_hazard_function(self, X, index, dataset, seed, plot=False):
        
        """
        Predict the cumulative hazard function for the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        index : array-like, shape (n_samples,)
            Index for the samples.
        dataset : str
            Name of the dataset.
        seed : int
            Random seed for reproducibility.
        plot : bool, default = ``False``
            Whether to plot the cumulative hazard function.

        Returns
        -------
        cumulative_hazard_function : array-like, shape (n_samples, n_times)
            Predicted cumulative hazard function.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_cumulative_hazard_function` with a model, the seed must be an integer. Value received: {seed}")
    
        self.cumulative_hazard_function = self._compute_cumulative_hazard_function(X, survival=False)

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

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        index : array-like, shape (n_samples,)
            Index for the samples.
        scaler : object
            Scaler used for the data.
        dataset : str
            Name of the dataset.
        seed : int
            Random seed for reproducibility.
        feature_names : list of str
            Names of the features.
        background : bool, default = ``False``
            Whether to use background data for SHAP.
        plot : bool, default = ``False``
            Whether to plot the XAI values.

        Returns
        -------
        shap_explainer : shap.Explainer
            SHAP explainer for model interpretability.
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