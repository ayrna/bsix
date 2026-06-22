import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torchtuples as tt
import time
import torch
import random
import warnings

from ..base import BaseSurvival
from ..loggers.deepSurvLogger import DeepSurvLogger
from ..nets.deepNets import DeepSurvFFNN
from .utils import BreslowEstimator

from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")

class DeepTimeVarying(BaseSurvival):

    """
    Deep Time-Varying model.

    Parameters
    ----------
    num_inputs : int
        Number of input features.
    valid_data : dict, default = ``None``
        Validation data in the form of a dictionary with keys "x", "e", and "t" for features, events, and times, respectively.
    hidden_layers : list of int, default = ``None``
        List specifying the number of units in each hidden layer.
    epochs : int, default =500
        Number of training epochs.
    learn_rate : float, default =0.0
        Learning rate for the optimizer.
    lr_decay : float, default =0.0
        Learning rate decay factor.
    l1_reg : float, default =0.0
        L1 regularization strength.
    l2_reg : float, default =0.0
        L2 regularization strength.
    momentum : float, default =0.9
        Momentum for the optimizer.
    activation : str, default = ``"relu"``
        Activation function to use in the hidden layers. ``relu``, ``selu``, ``tanh`` or ``sigmoid``.
    dropout : float, default =0.0
        Dropout rate for regularization.
    standardize : bool, default = ``True``
        Whether to standardize input features.
    ties : str, default = ``"cox"``
        Method for handling tied event times. ``"cox"`` or ``"breslow"``.
    device : torch.device, default = ``None``
        Device to run the model on (e.g., "cpu" or "cuda").
    validation_frequency : int, default =10
        Frequency (in epochs) to perform validation.
    patience : int, default =2000
        Number of epochs to wait for improvement before early stopping.
    improvement_threshold : float, default =0.99999
        Threshold for considering an improvement in validation loss.
    patience_increase : int, default =2
        Factor by which to increase patience when an improvement is observed.
    logger : DeepSurvLogger, default = ``None``
        Logger for tracking training progress.
    verbose : bool, default = ``True``
        Whether to print training progress.
    seed : int, default = ``None``
        Random seed for reproducibility.

    Attributes
    ----------
    breslow : BreslowEstimator
        Breslow estimator for baseline hazards.
    survival_function : array-like, shape (n_samples, n_times)
        Estimated survival function.
    cumulative_hazard_function : array-like, shape (n_samples, n_times)
        Estimated cumulative hazard function.
    shap_explainer : shap.Explainer
        SHAP explainer for model interpretability.

    Examples
    --------
    .. code:: python

        from bsix.models.metodologies import DeepTimeVarying
        model = DeepTimeVarying(num_inputs=10, hidden_layers=[32,], epochs=200, learn_rate=0.01)
        model.fit(X_train, y_train)
    """

    def __init__(self, num_inputs, valid_data=None, hidden_layers=None, epochs=500, learn_rate=0.0, lr_decay=0.0, l1_reg=0.0, l2_reg=0.0, momentum=0.9, 
                 activation="relu", dropout=0.0, standardize=True, ties="cox", device=None, validation_frequency=10, patience=2000, 
                 improvement_threshold=0.99999, patience_increase=2, logger=None, verbose=True, seed=None):
          
        """
        Initialise model with specified parameters.
        """
                
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Standardization parameters
        self.offset = torch.zeros(num_inputs, dtype=torch.float32, device=self.device)
        self.scale = torch.ones(num_inputs, dtype=torch.float32, device=self.device)
        self.standardize = standardize
    
        # Parameters
        self.num_inputs = num_inputs
        self.learn_rate = learn_rate
        self.lr_decay = lr_decay
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.momentum = momentum
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.ties = ties

        self.epochs = epochs
        self.valid_data = valid_data
        self.validation_frequency = validation_frequency
        self.patience = patience
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold

        self.logger = logger
        
        self.verbose = verbose

        self.seed = seed

        # Network (will be initialized in train())
        self.network = None

        # Optimizer (will be initialized in train())
        self.optimizer = None
    
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
    
    def _negative_log_likelihood(self, risk, t_start, t_stop, e):

        """
        Compute the negative partial log-likelihood for Cox proportional hazards.
        """

        # start_j < stop_i <= stop_j (j in risk set of i)
        t_stop_i = t_stop.view(-1, 1)
        t_start_j = t_start.view(1, -1)
        t_stop_j = t_stop.view(1, -1)

        mask = (t_start_j < t_stop_i) & (t_stop_i <= t_stop_j)
        risk_mask = torch.where(mask, risk.view(1, -1), torch.tensor(-float('inf'), device=risk.device))

        log_risk = torch.logsumexp(risk_mask, dim=1)
        
        uncensored_likelihood = risk.view(-1) - log_risk.view(-1)
        censored_likelihood = uncensored_likelihood * e.view(-1)
        num_observed_events = torch.sum(e)
        
        if num_observed_events == 0:
            return torch.tensor(0.0, device=risk.device, requires_grad=True)
        neg_likelihood = - (torch.sum(censored_likelihood) / num_observed_events)

        return neg_likelihood
    
    def _compute_l1_loss(self):

        """
        Compute L1 regularization loss.
        """

        l1_loss = 0.0
        for param in self.network.parameters():
            l1_loss += torch.sum(torch.abs(param))

        return l1_loss
    
    def _compute_l2_loss(self):

        """
        Compute L2 regularization loss.
        """

        l2_loss = 0.0
        for param in self.network.parameters():
            l2_loss += torch.sum(param ** 2)

        return l2_loss
    
    def _get_loss(self, x, e, t_start, t_stop):

        """
        Compute total loss including regularization.
        """

        risk = self.network(x)
        cox_loss = self._negative_log_likelihood(risk, t_start, t_stop, e)

        l1_loss = self._compute_l1_loss() if self.l1_reg > 0.0 else 0.0
        l2_loss = self._compute_l2_loss() if self.l2_reg > 0.0 else 0.0
        
        total_loss = cox_loss + (self.l1_reg * l1_loss) + (self.l2_reg * l2_loss)

        return total_loss
    
    def _get_concordance_index(self, x, t, e, **kwargs):

        """
        Calculate concordance index (C-index) for model predictions.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Input data.
        t : array-like, shape (n_samples,)
            Censoring times.
        e : array-like, shape (n_samples,)
            Event indicators.

        Returns
        -------
        c_index : float
            Concordance index.
        """

        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            if self.standardize:
                x_tensor = self._standardize_x(x_tensor)
            risk = np.ravel(self.network(x_tensor).cpu().numpy())

        return concordance_index_censored(e, t, risk)[0]
    
    def _standardize_x(self, x):

        """
        Standardize input features.
        """

        return (x - self.offset) / (self.scale + 1e-15)
    
    def fit(self, X_train, y_train, **kwargs):
        
        """
        Fit the model to the data.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Training data.
        y_train : structured array-like, shape (n_samples,)
            Target training values (events, start times, stop times).

        Returns
        -------
        self : DeepTimeVarying
            Fitted estimator.
        """
        
        # Set random seeds
        self._set_seeds()

        # Breslow estimator for baseline hazards
        self.breslow = BreslowEstimator()

        # Sort by time
        X_train, y_train = self._sort(X_train, y_train, "time_stop")

        # Apply y_train supervision
        y_train["time_stop"] = np.where(y_train["time_start"] == y_train["time_stop"], y_train["time_stop"] + 1e-15, y_train["time_stop"])

        if self.logger is None:
            logger = DeepSurvLogger("DeepTimeVarying")
        
        # Build network
        self.network = DeepSurvFFNN(
            num_inputs=self.num_inputs,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout=self.dropout,
        ).to(self.device)

        # Set standardization parameters
        if self.standardize:
            self.offset = torch.tensor(
                X_train.mean(axis=0), 
                dtype=torch.float32, 
                device=self.device
            )
            self.scale = torch.tensor(
                X_train.std(axis=0),
                dtype=torch.float32,
                device=self.device
            )
        
        # Events and Times
        e_train = np.array([event[0] for event in y_train], np.bool_)
        t_start_train = np.array([time[1] for time in y_train], np.float32)
        t_stop_train = np.array([time[2] for time in y_train], np.float32)

        if self.valid_data:
            X_valid = np.array(self.valid_data["x"], np.float64)
            e_valid = np.array(self.valid_data["e"], np.bool_)
            t_start_valid = np.array(self.valid_data["t"][0], np.float64)
            t_stop_valid = np.array(self.valid_data["t"][1], np.float64)
        
        # Convert to tensors
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        e_train_tensor = torch.tensor(e_train, dtype=torch.long, device=self.device)
        t_start_train_tensor = torch.tensor(t_start_train, dtype=torch.float32, device=self.device)
        t_stop_train_tensor = torch.tensor(t_stop_train, dtype=torch.float32, device=self.device)

        if self.valid_data:
            x_valid_tensor = torch.tensor(X_valid, dtype=torch.float32, device=self.device)
            e_valid_tensor = torch.tensor(e_valid, dtype=torch.long, device=self.device)
            t_start_valid_tensor = torch.tensor(t_start_valid, dtype=torch.float32, device=self.device)
            t_stop_valid_tensor = torch.tensor(t_stop_valid, dtype=torch.float32, device=self.device)
        
        if self.standardize:
            x_train_tensor = self._standardize_x(x_train_tensor)

            if self.valid_data:
                x_valid_tensor = self._standardize_x(x_valid_tensor)
        
        # Initialize optimizer with weight decay for L2 regularization
        self.optimizer = tt.optim.SGD(
            params=self.network.parameters(),
            lr=self.learn_rate,
            momentum=self.momentum,
        )
        
        # Training metrics
        best_validation_loss = np.inf
        best_params = None
        best_params_idx = -1
        
        start = time.time()
        
        for epoch in range(self.epochs):
            # Learning rate decay
            lr = self.learn_rate / (1 + epoch * self.lr_decay)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            
            logger.logValue("lr", lr, epoch)
            
            # Training step
            self.network.train()
            self.optimizer.zero_grad()
            
            loss = self._get_loss(x_train_tensor, e_train_tensor, t_start_train_tensor, t_stop_train_tensor)
            loss.backward()
            self.optimizer.step()
            
            logger.logValue("loss", loss.item(), epoch)
            
            # Calculate training C-index
            ci_train = self._get_concordance_index(X_train, t_stop_train, e_train)
            logger.logValue("c-index", ci_train, epoch)
            
            # Validation
            patience = self.patience
            if self.valid_data and (epoch % self.validation_frequency == 0):
                self.network.eval()
                with torch.no_grad():
                    validation_loss = self._get_loss(x_valid_tensor, e_valid_tensor, t_start_valid_tensor, t_stop_valid_tensor)
                    logger.logValue("valid_loss", validation_loss.item(), epoch)
                
                ci_valid = self._get_concordance_index(X_valid, t_stop_valid, e_valid)
                logger.logValue("valid_c-index", ci_valid, epoch)
                
                if validation_loss.item() < best_validation_loss:
                    if validation_loss.item() < best_validation_loss * self.improvement_threshold:
                        patience = max(patience, epoch * self.patience_increase)
                    
                    # Save best parameters
                    best_params = {
                        "model_state_dict": self.network.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                    }
                    best_params_idx = epoch
                    best_validation_loss = validation_loss.item()
            
            if self.verbose and (epoch % self.validation_frequency == 0):
                if self.valid_data:
                    logger.print_progress_bar(epoch, self.epochs, loss.item(), validation_loss.item(), ci_train, ci_valid)
                else:
                    logger.print_progress_bar(epoch, self.epochs, loss=loss.item(), ci=ci_train)
            
            if patience <= epoch:
                print("Early stopping at epoch %d" % epoch)
                break
        
        if self.verbose:
            logger.logMessage(f"Finished Training with {epoch + 1} iterations in {time.time() - start:.2f}s")
        
        # Compute baseline hazards with training data
        self.breslow.fit(self.predict(X_train), e_train, t_stop_train)

        logger.shutdown()
        
        logger.history["best_valid_loss"] = best_validation_loss
        logger.history["best_params"] = best_params
        logger.history["best_params_idx"] = best_params_idx
        
        self.history = logger.history
        
        return self
    
    def predict(self, x):

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

        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            if self.standardize:
                x_tensor = self._standardize_x(x_tensor)
            risk = self.network(x_tensor).cpu().numpy().flatten()
            
        return risk

    # ----------------------
    # Base Survival methods
    # ----------------------
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
        
        risk = self.predict(X)

        self.survival_function = self.breslow.get_survival_function(risk)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.survival_function, index, "DeepSurv Time-Varying", dataset, "Survival", seed)
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
        
        risk = self.predict(X)
        
        self.cumulative_hazard_function = self.breslow.get_cumulative_hazard_function(risk)

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.cumulative_hazard_function, index, "DeepSurv Time-Varying", dataset, "CumulativeRisk", seed)
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
            figure, ax = BaseSurvival.plot_shap(self.shap_explainer, index, scaler, "DeepSurv Time-Varying", dataset, seed)
            plt.show()

        return self.shap_explainer