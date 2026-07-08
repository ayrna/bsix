import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import shap
import torch
import torch.nn as nn
import torchtuples as tt
import warnings

from ..base import BaseSurvival
from pycox.models import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

warnings.filterwarnings("ignore")

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _get_activation(activation):

    """
    Resolve an activation-function name to its nn.Module class.
    """
    try:
        return _ACTIVATIONS[activation]
    except KeyError:
        raise ValueError(f"Unknown activation function: {activation}")
    
class BaseDeepHit(BaseSurvival):

    """
    DeepHit Regression model (using pycox and PyTorch).
    """

    def __init__(self, num_durations, epochs=50, learning_rate=1e-3, num_nodes=None, alpha=0.2, sigma=0.1, activation="relu", dropout=0.0, seed=None, time_threshold=None):

        """
        Initialise model with specified parameters.
        """

        self.num_durations = num_durations
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.sigma = sigma
        self.activation = activation
        self.dropout = dropout
        self.seed = seed
        self.time_threshold = num_durations - 1 if time_threshold is None else max(0, min(time_threshold, num_durations - 1))

        self.model = None
        self.labtrans = None
        self.labels_covariables = ["event", "time"]

    def _set_seeds(self):

        """
        Initialise random seeds for reproducibility.
        """

        if self.seed is not None:
            seed = self.seed

            # Python
            random.seed(seed)

            # NumPy
            np.random.seed(seed)

            # PyTorch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def fit(self, X, y):

        """
        Fit the model to the data.
        """
        
        # Set random seeds
        self._set_seeds()

        X_np = X.values.astype('float32') if isinstance(X, pd.DataFrame) else X.astype('float32')
        
        time = np.array(y["time"]).astype('float32')
        event = np.array(y["event"]).astype('int32')

        self.labtrans = LabTransDiscreteTime(self.num_durations, scheme="quantiles")
        y_train = self.labtrans.fit_transform(time, event)

        in_features = X_np.shape[1]
        out_features = self.labtrans.out_features
        
        layers = []
        in_dim = in_features
        for out_dim in self.num_nodes:
            layers.append(torch.nn.Linear(in_dim, out_dim))
            layers.append(_get_activation(self.activation)())
            layers.append(torch.nn.Dropout(self.dropout))
            in_dim = out_dim
        layers.append(torch.nn.Linear(in_dim, out_features))

        net = torch.nn.Sequential(*layers)

        self.model = DeepHitSingle(
            net, 
            tt.optim.Adam, 
            alpha=self.alpha, 
            sigma=self.sigma, 
            duration_index=self.labtrans.cuts
        )
        self.model.optimizer.set_lr(self.learning_rate)

        self.model.fit(X_np, y_train, epochs=self.epochs, verbose=False)
        
        return self

    def predict(self, X):

        """
        Predict risk scores for the given data.
        """
        X_np = X.values.astype('float32') if isinstance(X, pd.DataFrame) else X.astype('float32')
        
        surv = self.model.predict_surv_df(X_np).iloc[self.time_threshold - 1, :]
        risk = 1.0 - surv.values

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
        S(x, t) estimation natively by DeepHit.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_survival_function` with a model, the seed must be an integer. Value received: {seed}")
        
        X_np = X.values.astype('float32') if isinstance(X, pd.DataFrame) else X.astype('float32')
        
        self.survival_function = self.model.predict_surv_df(X_np)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.survival_function, index, "BaseDeepHit", dataset, "Survival", seed)
            plt.show()
            
        return self.survival_function

    def predict_cumulative_hazard_function(self, X, index, dataset, seed, plot=False):
        
        """
        H(x,t) estimation natively by DeepHit.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_cumulative_hazard_function` with a model, the seed must be an integer. Value received: {seed}")

        X_np = X.values.astype('float32') if isinstance(X, pd.DataFrame) else X.astype('float32')

        surv = self.model.predict_surv_df(X_np)
        self.cumulative_hazard_function = -np.log(surv + 1e-8)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.cumulative_hazard_function, index, "BaseDeepHit", dataset, "CumulativeRisk", seed)
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

        # Usar un wrapper para SHAP (ya que predict necesita un array de numpy/tensor)
        def predict_fn(x):
            x_tensor = torch.tensor(x).float() if not isinstance(x, torch.Tensor) else x
            return self.predict(pd.DataFrame(x_tensor.numpy(), columns=feature_names))
            
        masker = shap.maskers.Independent(X, max_samples=X.shape[0])
        explainer_risk = shap.Explainer(predict_fn, masker, feature_names=feature_names, seed=seed)
        
        X_background = X.copy()
        if background:
            X_background = pd.DataFrame(shap.kmeans(X, background).data, columns=feature_names)

        self.shap_explainer = explainer_risk(X_background)

        mean_shap_values = np.abs(self.shap_explainer.values).mean(axis=0)
        coefficients = {feature_names[i]: round(val, 8) for i, val in enumerate(mean_shap_values)}
        self.coefficients = {k: v for k, v in sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)}

        if plot:
            figure, ax = BaseSurvival.plot_coefficients(self.coefficients, "BaseDeepHit", dataset, seed)
            figure, ax = BaseSurvival.plot_shap(self.shap_explainer, index, scaler, "BaseDeepHit", dataset, seed)
            
            plt.show()

        return self.shap_explainer, self.coefficients