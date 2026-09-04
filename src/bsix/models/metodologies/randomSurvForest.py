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
    Random survival forest estimator for nonparametric survival analysis.

    This model builds an ensemble of survival trees from bootstrap samples and
    aggregates their predictions to estimate both survival and cumulative hazard
    functions. It follows the same public interface as the other survival models
    in the package, while leveraging the flexibility of tree ensembles to capture
    nonlinear covariate effects and nonproportional risk patterns.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs used for tree training.
    n_estimators : int, default=100
        Number of survival trees in the forest.
    max_depth : int or None, default=None
        Maximum depth allowed for each tree.
    min_samples_leaf : int, default=3
        Minimum number of samples required at a leaf node.
    min_samples_split : int, default=6
        Minimum number of samples required to split an internal node.

    Attributes
    ----------
    model : list of SurvTree
        Trained forest composed of individual survival trees.
    survival_function : ndarray of shape (n_samples, n_times)
        Estimated survival function for each sample.
    cumulative_hazard_function : ndarray of shape (n_samples, n_times)
        Estimated cumulative hazard function for each sample.
    shap_explainer : shap.Explainer
        SHAP explainer used to interpret the model output.

    Notes
    -----
    Random survival forests aggregate the output of multiple survival trees by
    averaging their step-wise cumulative hazard or survival estimates. The
    resulting prediction can be interpreted as an ensemble risk score derived
    from the underlying tree structure.

    Examples
    --------
    >>> from bsix.models.metodologies import RandomSurvForest
    >>> model = RandomSurvForest(seed=42, n_estimators=100, max_depth=5)
    >>> model.fit(X_train, y_train)
    >>> risk = model.predict(X_test)
    """

    def __init__(self, seed, n_jobs=-1, n_estimators=100, max_depth=None, min_samples_leaf=3, min_samples_split=6):

        """
        Initialize the random survival forest model.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        n_jobs : int, default=-1
            Number of parallel jobs used during training.
        n_estimators : int, default=100
            Number of trees to train.
        max_depth : int or None, default=None
            Maximum depth of each tree.
        min_samples_leaf : int, default=3
            Minimum number of samples required to create a leaf node.
        min_samples_split : int, default=6
            Minimum number of samples required to split an internal node.
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
        Train a single survival tree on a bootstrap sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : structured array-like of shape (n_samples,)
            Survival labels with fields ``event`` and ``time``.
        n_samples : int
            Number of rows in the training dataset.
        tree_seed : int
            Random seed used for the bootstrap sample and tree fit.

        Returns
        -------
        SurvTree
            Fitted individual survival tree.
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
        Fit the random survival forest to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : structured array-like of shape (n_samples,)
            Target training values containing the fields ``event`` and ``time``.

        Returns
        -------
        RandomSurvForest
            The fitted estimator instance.
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
        X : array-like of shape (n_samples, n_features)
            Feature matrix used for prediction.

        Returns
        -------
        ndarray of shape (n_samples,)
            Aggregated risk score for each sample.
        """
        
        chfs = self._compute_cumulative_hazard_function(X, survival=False)
        risk = np.array([np.sum(chf.y) for chf in chfs])

        return risk
    
    def score(self, X, y):
        
        """
        Return the model score for the provided data.

        This method is kept for compatibility with the estimator interface but is
        not implemented for this class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : structured array-like of shape (n_samples,)
            Ground-truth survival labels.

        Returns
        -------
        None
            The score is currently not defined for this implementation.
        """
        
        return None
    
    # ----------------------
    # Base Survival methods
    # ----------------------
    def _compute_cumulative_hazard_function(self, X, survival=False):
        
        """
        Compute the aggregated cumulative hazard or survival function across the
        forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for which to estimate the forest response.
        survival : bool, default=False
            If ``True``, return the estimated survival function. Otherwise, return
            the cumulative hazard function.

        Returns
        -------
        ndarray of shape (n_samples,)
            Array of ``StepFunction`` objects representing the aggregated tree
            predictions for each sample.
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
        Predict the survival function for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix used for prediction.
        index : array-like of shape (n_samples,)
            Sample indices used for plotting.
        dataset : str
            Name of the dataset used in the generated plot.
        seed : int
            Random seed for reproducibility.
        plot : bool, default=False
            If ``True``, display the survival-function plot.

        Returns
        -------
        ndarray of shape (n_samples,)
            Array of ``StepFunction`` objects with the estimated survival curves.
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
        Predict the cumulative hazard function for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix used for prediction.
        index : array-like of shape (n_samples,)
            Sample indices used for plotting.
        dataset : str
            Name of the dataset used in the generated plot.
        seed : int
            Random seed for reproducibility.
        plot : bool, default=False
            If ``True``, display the cumulative hazard plot.

        Returns
        -------
        ndarray of shape (n_samples,)
            Array of ``StepFunction`` objects with the estimated cumulative hazard
            curves.
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
        Compute SHAP-based explainability values for the forest model.

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
            Random seed for reproducibility.
        feature_names : list of str
            Names of the model features.
        background : bool, default=False
            If ``True``, compute the SHAP background using k-means summary data.
        plot : bool, default=False
            If ``True``, display the SHAP plot.

        Returns
        -------
        shap.Explainer
            SHAP explainer for the fitted forest.
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