import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
import warnings

from ..base import BaseSurvival
from .utils import StepFunction

from numba import njit
from sklearn.utils.validation import check_random_state

warnings.filterwarnings("ignore")

@njit(fastmath=True, cache=True)
def _calculate_log_rank_njit(times_left, events_left, unique_times, n_j, d_j):

    """
    Calculate the Log-Rank statistic between two groups defined by left mask.
    """
    
    # At-risk count in the left group (nleft_j) at each unique event time
    idx_left = np.searchsorted(times_left, unique_times, side="left")
    nleft_j = (len(times_left) - idx_left).astype(np.float64)

    # Event count in the left group (dleft_j) at each unique event time
    mask = events_left != 0 
    tleft_events = times_left[mask]
    
    if len(tleft_events) > 0:
        dleft_j = np.bincount(np.searchsorted(unique_times, tleft_events), minlength=len(unique_times)).astype(np.float64)
    else:
        dleft_j = np.zeros(len(unique_times), dtype=np.float64)

    safe_n_j = np.where(n_j > 0, n_j, 1.0)
    # Calculate U statistic (Observed - Expected events)
    U = np.sum(np.where(n_j > 0, dleft_j - d_j * (nleft_j / safe_n_j), 0.0))

    # Calculate Variance (V) assuming a hypergeometric distribution
    valid = n_j > 1.0
    nv = n_j[valid]
    V = np.sum(d_j[valid] * nleft_j[valid] * (nv - nleft_j[valid]) * (nv - d_j[valid]) / (nv ** 2 * (nv - 1.0)))

    # Return Log-Rank score
    return (U ** 2) / V if V > 0.0 else 0.0

@njit(fastmath=True, cache=True)
def _best_split_njit(X, events, times, unique_times, n_j, d_j, features, min_samples_leaf):

    """
    Return (best_feature_idx, best_threshold) maximising log-rank score.
    """

    best_score = -1.0
    best_feature = -1
    best_threshold = np.nan

    for fi in features:
        col = X[:, fi]
        uniq = np.unique(col)

        # Skip the very last threshold since it would create an empty right node
        for i in range(len(uniq) - 1):
            thresh = uniq[i]
            left_mask = col <= thresh
            num_left = int(left_mask.sum())
            num_right = len(left_mask) - num_left
            
            # Check constraints
            if num_left < min_samples_leaf or num_right < min_samples_leaf:
                continue

            score = _calculate_log_rank_njit(times[left_mask], events[left_mask], unique_times, n_j, d_j)

            # Update best split
            if score > best_score:
                best_score = score
                best_feature = fi
                best_threshold = thresh

    return best_feature, best_threshold

class LeafEstimator:

    """
    Local estimator for leaf nodes in the Survival Tree.
    """

    def __init__(self):

        """
        Initialise with specified parameters.
        """

        # Parameters
        self.times = None
        self.survival = None
        self.cumulative_hazard = None

    def fit(self, events, times, global_times):

        """
        Fit the estimator to the data.
        """
                
        self.times = global_times

        # Sort by time (already sorted?)
        sort_idx = np.argsort(times)
        t_sorted = times[sort_idx]
        e_sorted = events[sort_idx].astype(bool)
 
        # Risk set (n_i) at each global time point
        risk_set = len(times) - np.searchsorted(t_sorted, self.times, side="left")
 
        # Count the exact number of events (d_i) at each global time point
        d_events = np.zeros(len(self.times), dtype=np.float64)
        event_times = t_sorted[e_sorted]

        if len(event_times) > 0:
            # Map local event times to their corresponding index in the global grid
            idx = np.searchsorted(self.times, event_times)
            # Counts multiple events happening at the same time
            np.add.at(d_events, idx, 1)
 
        safe_risk = np.where(risk_set > 0, risk_set, 1.0)
        # Calculate discrete hazards (d_i / n_i)
        hazards = np.where(risk_set > 0, d_events / safe_risk, 0.0)
 
        self.cumulative_hazard = np.cumsum(hazards)
        self.survival = np.cumprod(1.0 - hazards)

class TreeNode:

    """
    Node in the Survival Tree.
    """
 
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, is_leaf=False, risk_value=None, estimator=None):
        
        """
        Initialise with specified parameters.
        """
                
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.risk_value = risk_value
        self.estimator = estimator

class SurvTree(BaseSurvival):

    """
    Survival Tree model using Log-Rank test for node splitting.
    """

    def __init__(self, max_depth=None, min_samples_split=6, min_samples_leaf=3, seed=0):

        """
        Initialise model with specified parameters.
        """

        # Parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        
        self.root = None
        self.unique_times = None
        self.labels_covariables = ["event", "time"]
    
    def _best_split(self, X, events, times, n_features):

        """
        Return (best_feature_idx, best_threshold) maximising log-rank score.
        """

        # Extract unique event times and counts for the parent node
        event_mask = events.astype(bool)
        unique_times, d_j_int = np.unique(times[event_mask], return_counts=True)

        # If no events occurred in this node, it cannot be split
        if len(unique_times) == 0:
            return None, None
 
        # Pre-compute parent at-risk counts (n_j) and events (d_j).
        n_j = (len(times) - np.searchsorted(times, unique_times, side="left")).astype(np.float64)
        d_j = d_j_int.astype(np.float64)

        # Shuffle features to ensure random, reproducible tie-breaking
        features = np.arange(n_features)
        self.rng.shuffle(features)
 
        best_feature, best_threshold = _best_split_njit(X, events, times, unique_times, n_j, d_j, features, self.min_samples_leaf)

        if best_feature == -1:
            return None, None
        
        return best_feature, best_threshold

    def _create_leaf(self, events, times):

        """
        Instantiate a terminal node and fit the local survival estimators.
        """

        estimator = LeafEstimator()
        estimator.fit(events, times, self.unique_times)
        
        # Risk value defined as the area under the cumulative hazard curve
        risk_value = float(np.sum(estimator.cumulative_hazard))
        
        return TreeNode(is_leaf=True, risk_value=risk_value, estimator=estimator)
    
    def _build_tree(self, X, events, times, depth):

        """
        Recursively build the tree.
        """

        n_samples, n_features = X.shape
        
        stop = (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or int(events.sum()) == 0
        )

        # Evaluate stopping criteria
        if stop:
            return self._create_leaf(events, times)

        # Search for the optimal split
        best_feature, best_threshold = self._best_split(X, events, times, n_features)

        # If no valid split was found convert to leaf
        if best_feature is None:
            return self._create_leaf(events, times)

        # Create boolean mask for the left branch
        left_mask = X[:, best_feature] <= best_threshold

        # Recursively construct left and right branches
        return TreeNode(feature=best_feature, threshold=best_threshold, left=self._build_tree(X[left_mask], events[left_mask], times[left_mask], depth + 1), right=self._build_tree(X[~left_mask], events[~left_mask], times[~left_mask], depth + 1))

    def fit(self, X, y):

        """
        Fit the model to the data.
        """
        
        X, y = self._sort(X, y)
        
        events = y["event"]
        times = y["time"]

        self.rng = check_random_state(self.seed)

        self.unique_times = np.unique(times)
        self.root = self._build_tree(X, events, times, depth=0)
        
        return self

    def predict(self, X):

        """
        Predict risk scores for the given data.
        """
    
        leaves = self._get_leaves(X)
        
        risks = np.empty(len(leaves), dtype=np.float64)
        for i, node in enumerate(leaves):
            risks[i] = node.risk_value
            
        return risks

    def score(self, X, y):

        """
        Calculate the score for the model.
        """

        return None
    
    def _get_leaves(self, X):

        """
        Finds and returns the leaf node for each sample.
        """
        
        leaves = np.empty(X.shape[0], dtype=object)
        for i in range(X.shape[0]):
            node = self.root

            while not node.is_leaf:
                if X[i, node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            leaves[i] = node
            
        return leaves
    
    # ----------------------
    # Base Survival methods
    # ----------------------
    def _compute_survival_hazard_functions(self, X, survival=True):
        
        """
        Auxiliary method for computing the cumulative hazard function.
        """

        if not self.root:
            raise ValueError(f"When computing `cumulative_hazard_function` with a model, first fit the model.")
            
        leaves = self._get_leaves(X)
        
        functions = []
        for node in leaves:
            if survival:
                functions.append(StepFunction(node.estimator.times, np.exp(-node.estimator.cumulative_hazard), is_survival=survival))
            else:
                functions.append(StepFunction(node.estimator.times, node.estimator.cumulative_hazard, is_survival=survival))           
            
        return np.array(functions, dtype=object)
    
    def predict_survival_function(self, X, index, dataset, seed, plot=False):

        """ 
        Survival function.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_survival_function`, the seed must be an integer. Value received: {seed}")
        
        self.survival_function = self._compute_survival_hazard_functions(X, survival=True)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.survival_function, index, "Survival Tree", dataset, "Survival", seed)
            plt.show()
            
        return self.survival_function

    def predict_cumulative_hazard_function(self, X, index, dataset, seed, plot=False):

        """
        Cumulative hazard function.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_cumulative_hazard_function`, the seed must be an integer. Value received: {seed}")
        
        self.cumulative_hazard_function = self._compute_survival_hazard_functions(X, survival=False)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.cumulative_hazard_function, index, "Survival Tree", dataset, "CumulativeRisk", seed)
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
            figure, ax = BaseSurvival.plot_shap(self.shap_explainer, index, scaler, "Survival Tree", dataset, seed)
            plt.show()

        return self.shap_explainer