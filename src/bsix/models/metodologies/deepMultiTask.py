"""Multi-task deep survival model.

This estimator combines several survival heads or likelihood terms to learn a
shared latent representation while preserving compatibility with the package's
survival prediction API.
"""

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
from ..loggers.deepMultiTaskLogger import DeepMultiTaskLogger
from ..nets.deepNets import DeepMultiTaskFFNN
from .utils import BreslowEstimator

from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")

class DeepMultiTask(BaseSurvival):

    """
    Multi-task deep survival model for competing-risk learning.

    This implementation builds a shared latent representation from the input
    covariates and then fits multiple Cox-style survival heads in parallel. Each
    head models a different progression or competing-risk channel, while the
    overall network is optimized with a composite loss combining the Cox partial
    likelihood terms and regularization penalties. The fitted risk scores are then
    transformed into survival and cumulative hazard curves through the Breslow
    baseline estimator associated with each event type.

    Parameters
    ----------
    number_inputs : int
        Number of input features.
    valid_data : dict, optional
        Validation dataset containing the keys ``x``, ``e`` and ``t``.
    hidden_layers : list of int, optional
        Hidden-layer widths used by the network.
    epochs : int, default=500
        Number of training epochs.
    learn_rate : float, default=0.0
        Learning rate for the optimizer.
    lr_decay : float, default=0.0
        Learning-rate decay factor.
    l1_reg : float, default=0.0
        L1 regularization strength.
    l2_reg : float, default=0.0
        L2 regularization strength.
    cox_reg : float, default=0.0
        Weight applied to the Cox-oriented loss in the total objective.
    momentum : float, default=0.9
        Momentum for the optimizer.
    activation : str, default="relu"
        Activation function used in the hidden layers.
    dropout : float, default=0.0
        Dropout probability applied inside the model.
    standardize : bool, default=True
        Whether to standardize the input features before training.
    ties : {"cox", "breslow"}, default="cox"
        Tie-handling rule used inside the partial log-likelihood.
    device : torch.device, optional
        Device used for training and inference, e.g. ``cpu`` or ``cuda``.
    validation_frequency : int, default=10
        Validation interval in epochs.
    patience : int, default=500
        Maximum number of epochs to wait for validation improvement before early
        stopping.
    improvement_threshold : float, default=0.99999
        Minimal relative improvement required to count as progress.
    patience_increase : int, default=25
        Factor by which patience is increased after improvement.
    logger : object, optional
        Logger used to track training metrics.
    verbose : bool, default=True
        Whether to print training progress.
    seed : int, optional
        Random seed for reproducibility.
    coef_likelihood : list of float, default=[1.0]
        Weights assigned to the likelihood term of each competing event.

    Attributes
    ----------
    number_events : int
        Number of competing-risk heads learned by the model.
    network : object
        Trained neural network model.
    optimizer : object
        Optimizer used during training.
    breslow : list of BreslowEstimator
        Baseline hazard estimator associated with each risk head.
    survival_functions : list or ndarray
        Estimated survival function(s) for each sample and risk channel.
    cumulative_hazard_functions : list or ndarray
        Estimated cumulative hazard function(s) for each sample and risk channel.
    shap_explainer : list of shap.Explainer
        SHAP explainers used to interpret the model output.

    Notes
    -----
    The model assumes a proportional hazards structure within each risk channel,
    and the risk score is computed as the output of the event-specific head after
    the shared latent representation is learned. The baseline survival curves are
    reconstructed by fitting a Breslow estimator for each event type.

    Examples
    --------
    >>> from bsix.models.metodologies import DeepMultiTask
    >>> model = DeepMultiTask(
    ...     number_inputs=10,
    ...     hidden_layers=[32, 16],
    ...     epochs=200,
    ...     learn_rate=0.01,
    ... )
    >>> model.fit(X_train, y_train)
    >>> risk = model.predict(X_test)
    """
    
    def __init__(self, number_inputs, valid_data=None, hidden_layers=None, epochs=500, learn_rate=0.0, lr_decay=0.0, l1_reg=0.0, l2_reg=0.0, cox_reg=0.0,
                 momentum=0.9, activation="relu", dropout=0.0, standardize=True, ties="cox", device=None, validation_frequency=10, 
                 patience=500, improvement_threshold=0.99999, patience_increase=25, logger=None, verbose=True, seed=None, coef_likelihood=[1.0]):
        
        """
        Initialize the multi-task deep survival model.

        Parameters
        ----------
        number_inputs : int
            Number of input features.
        valid_data : dict, optional
            Validation dataset with keys ``x``, ``e`` and ``t``.
        hidden_layers : list of int, optional
            Hidden-layer widths used by the network.
        epochs : int, default=500
            Number of training epochs.
        learn_rate : float, default=0.0
            Learning rate for the optimizer.
        lr_decay : float, default=0.0
            Learning-rate decay factor.
        l1_reg : float, default=0.0
            L1 regularization strength.
        l2_reg : float, default=0.0
            L2 regularization strength.
        cox_reg : float, default=0.0
            Weight applied to the Cox loss in the total objective.
        momentum : float, default=0.9
            Momentum for the optimizer.
        activation : str, default="relu"
            Activation function in the hidden layers.
        dropout : float, default=0.0
            Dropout probability.
        standardize : bool, default=True
            Whether to standardize the input features.
        ties : {"cox", "breslow"}, default="cox"
            Method used to handle tied event times.
        device : torch.device, optional
            Device used for training and inference.
        validation_frequency : int, default=10
            Validation interval in epochs.
        patience : int, default=500
            Maximum number of epochs to wait for improvement.
        improvement_threshold : float, default=0.99999
            Minimal relative improvement threshold for early stopping.
        patience_increase : int, default=25
            Increase factor applied to patience after improvement.
        logger : object, optional
            Logger used to track optimization metrics.
        verbose : bool, default=True
            Whether to print training progress.
        seed : int, optional
            Random seed for reproducibility.
        coef_likelihood : list of float, default=[1.0]
            Weight of the likelihood term for each risk channel.
        """
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Standardization parameters
        self.offset = torch.zeros(number_inputs, dtype=torch.float32, device=self.device)
        self.scale = torch.ones(number_inputs, dtype=torch.float32, device=self.device)
        self.standardize = standardize
        
        # Parameters
        self.number_inputs = number_inputs
        self.learn_rate = learn_rate
        self.lr_decay = lr_decay
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.cox_reg = cox_reg
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
    
        # Loss coefficients
        self.coef_likelihood = coef_likelihood

        # Network (will be initialized in train())
        self.network = None

        # Optimizer (will be initialized in train())
        self.optimizer = None
    
    def _set_seeds(self):

        """
        Initialize random seeds for reproducibility.

        Returns
        -------
        None
            This method updates the Python, NumPy and PyTorch random generators.
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
    
    def _negative_log_likelihood(self, risk, t, e):

        """
        Compute the negative partial log-likelihood for a Cox-type risk head.

        Parameters
        ----------
        risk : torch.Tensor of shape (n_samples,)
            Predicted log-risk score for each sample.
        t : torch.Tensor of shape (n_samples,)
            Observation time for each sample.
        e : torch.Tensor of shape (n_samples,)
            Event indicator, where 1 denotes event and 0 denotes censoring.

        Returns
        -------
        torch.Tensor
            Scalar negative partial log-likelihood for the supplied risk head.
        """
        
        risk, t, e = self._sort_multitask(risk, t, e)

        t_i = t.view(-1, 1)
        t_j = t.view(1, -1)

        if self.ties == "cox":
            log_risk = torch.logsumexp(risk, dim=0)
        elif self.ties == "breslow":
            mask = t_i <= t_j
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
        Compute the L1 regularization penalty over the network weights.

        Returns
        -------
        torch.Tensor
            Total L1 loss for the network parameters.
        """

        l1_loss = 0.0
        for param in self.network.parameters():
            l1_loss += torch.sum(torch.abs(param))

        return l1_loss
    
    def _compute_l2_loss(self):

        """
        Compute the L2 regularization penalty over the network weights.

        Returns
        -------
        torch.Tensor
            Total L2 loss for the network parameters.
        """

        l2_loss = 0.0
        for param in self.network.parameters():
            l2_loss += torch.sum(param ** 2)

        return l2_loss
    
    def _get_loss(self, x, e, t):

        """
        Compute the total training loss for the multi-task survival model.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
            Input feature matrix for the current batch.
        e : torch.Tensor of shape (n_samples, n_events)
            Event indicator matrix for each competing-risk head.
        t : torch.Tensor of shape (n_samples, n_events)
            Observation time matrix for each competing-risk head.

        Returns
        -------
        torch.Tensor
            Scalar total loss value, including Cox terms and regularization.
        """

        risk = self.network(x)[1][:, :self.number_events]

        cox_loss = []
        for k in range(self.number_events):
            cox_loss.append(self._negative_log_likelihood(risk[:, k], t[:, k], e[:, k]) * self.coef_likelihood[k])
        cox_loss = torch.stack(cox_loss)
        
        l1_loss = self._compute_l1_loss() if self.l1_reg > 0.0 else 0.0
        l2_loss = self._compute_l2_loss() if self.l2_reg > 0.0 else 0.0
        
        total_loss = (self.cox_reg * torch.sum(cox_loss)) + (self.l1_reg * l1_loss) + (self.l2_reg * l2_loss)

        return total_loss
    
    def _get_concordance_index(self, x, t, e, **kwargs):

        """
        Compute the concordance index for each risk head.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Feature matrix used for prediction.
        t : array-like of shape (n_samples, n_events)
            Observation times for each event head.
        e : array-like of shape (n_samples, n_events)
            Event indicators for each event head.
        **kwargs
            Additional keyword arguments accepted for compatibility.

        Returns
        -------
        torch.Tensor of shape (n_events,)
            Concordance index value for each risk head.
        """

        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            if self.standardize:
                x_tensor = self._standardize_x(x_tensor)
            risk = self.network(x_tensor)[1][:, :self.number_events].cpu().numpy()
            
        c_index_censored = []
        for k in range(self.number_events):
            c_index_censored.append(torch.tensor(concordance_index_censored(e[:, k], t[:, k], risk[:, k])[0], dtype=torch.float32, device=self.device))
        c_index_censored = torch.stack(c_index_censored)
        
        return c_index_censored
    
    def _standardize_x(self, x):

        """
        Standardize input features using the training-time offset and scale.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
            Input feature matrix to standardize.

        Returns
        -------
        torch.Tensor
            Standardized feature matrix.
        """

        return (x - self.offset) / (self.scale + 1e-15)
    
    def fit(self, X_train, y_train, **kwargs):
        
        """
        Fit the model to the training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : structured array-like of shape (n_events, n_samples)
            Target values for each event head, where each column contains the
            event indicator and time pairs for one progression channel.

        Returns
        -------
        DeepMultiTask
            The fitted estimator instance.
        """
        
        # Set random seeds
        self._set_seeds()

        # Set the number of competing events
        self.number_events = y_train.shape[1]

        # Breslow estimator for baseline hazards
        self.breslow = [BreslowEstimator() for _ in range(self.number_events)]

        if self.logger is None:
            logger = DeepMultiTaskLogger("DeepMultiTask")
        
        # Build network
        self.network = DeepMultiTaskFFNN(
            number_inputs=self.number_inputs,
            number_events=self.number_events,
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
        e_train = []
        t_train = []
        for k in range(self.number_events):
            e_train.append(np.array([evento for evento, _ in y_train[:, k]], np.bool_))
            t_train.append(np.array([tiempo for _, tiempo in y_train[:, k]], np.float32))
        e_train = np.array(e_train, np.bool_).T
        t_train = np.array(t_train, np.float32).T
        
        if self.valid_data:
            X_valid = np.array(self.valid_data["x"], np.float32)
            e_valid = []
            t_valid = []
            
            for k in range(self.number_events):
                e_valid.append(np.array(self.valid_data["e"][:, k], np.bool_))
                t_valid.append(np.array(self.valid_data["t"][:, k], np.float32))
            e_valid = np.array(e_valid, np.bool_).T
            t_valid = np.array(t_valid, np.float32).T
        
        # Convert to tensors
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        e_train_tensor = torch.tensor(e_train, dtype=torch.long, device=self.device)
        t_train_tensor = torch.tensor(t_train, dtype=torch.float32, device=self.device)

        if self.valid_data:
            x_valid_tensor = torch.tensor(X_valid, dtype=torch.float32, device=self.device)
            e_valid_tensor = torch.tensor(e_valid, dtype=torch.long, device=self.device)
            t_valid_tensor = torch.tensor(t_valid, dtype=torch.float32, device=self.device)

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
            
            loss = self._get_loss(x_train_tensor, e_train_tensor, t_train_tensor)
            loss.backward()
            ###torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            logger.logValue("loss", loss.item(), epoch)
            
            # Calculate training C-index
            ci_train = self._get_concordance_index(X_train, t_train, e_train)
            logger.logValue("c-index", ci_train, epoch)
            
            # Validation
            patience = self.patience
            if self.valid_data and (epoch % self.validation_frequency == 0):
                self.network.eval()
                with torch.no_grad():
                    validation_loss = self._get_loss(x_valid_tensor, e_valid_tensor, t_valid_tensor)
                    logger.logValue("valid_loss", validation_loss.item(), epoch)
                
                ci_valid = self._get_concordance_index(X_valid, t_valid, e_valid)
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
        for k in range(self.number_events):
            self.breslow[k].fit(self.predict(X_train)[:, k], e_train[:, k], t_train[:, k])

        logger.shutdown()
        
        logger.history["best_val_loss"] = best_validation_loss
        logger.history["best_params"] = best_params
        logger.history["best_params_idx"] = best_params_idx
        
        self.history = logger.history
        
        return self
    
    def predict(self, x):

        """
        Predict risk scores for the given data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        ndarray of shape (n_samples, n_events)
            Predicted relative risk for each event channel.
        """

        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            if self.standardize:
                x_tensor = self._standardize_x(x_tensor)
            risk = self.network(x_tensor)[1][:, :self.number_events].cpu().numpy()
            
        return risk
    
    def predict_outputs(self, x):

        """
        Predict the raw model outputs for the given data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        ndarray
            Raw output tensor from the neural network, before any post-processing.
        """

        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            if self.standardize:
                x_tensor = self._standardize_x(x_tensor)
            outputs = self.network(x_tensor)[0].cpu().numpy()
            
        return outputs
    
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
        list or ndarray
            Estimated survival functions for each event head. If there is only one
            event channel, a single array is returned.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_survival_function` with a model, the seed must be an integer. Value received: {seed}")
        risk = self.predict(X)

        self.survival_functions = []
        for k in range(self.number_events):
            survival_function = self.breslow[k].get_survival_function(risk[:, k])
            self.survival_functions.append(survival_function)

            if plot:
                figure, ax = self._plot_survival_hazard_functions(survival_function, index, "DeepSurv Multi-Task", dataset, "Survival", seed, k)
                plt.show()
        
        self.survival_functions = self.survival_functions[0] if self.number_events == 1 else self.survival_functions

        return self.survival_functions

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
        list or ndarray
            Estimated cumulative hazard functions for each event head. If there is
            only one event channel, a single array is returned.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_cumulative_hazard_function` with a model, the seed must be an integer. Value received: {seed}")
        
        risk = self.predict(X)
        
        self.cumulative_hazard_functions = []
        for k in range(self.number_events):
            cumulative_hazard_function = self.breslow[k].get_cumulative_hazard_function(risk[:, k])
            self.cumulative_hazard_functions.append(cumulative_hazard_function)

            if plot:
                figure, ax = self._plot_survival_hazard_functions(cumulative_hazard_function, index, "DeepSurv Multi-Task", dataset, "CumulativeRisk", seed, k)
                plt.show()

        self.cumulative_hazard_functions = self.cumulative_hazard_functions[0] if self.number_events == 1 else self.cumulative_hazard_functions

        return self.cumulative_hazard_functions

    # ----------------------
    # XAI
    # ----------------------
    def calculate_xai(self, X, index, scaler, dataset, seed, feature_names, background=False, plot=False):

        """
        Compute SHAP-based explainability values for the model.

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
            SHAP explainer for the fitted model.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `calculate_xai` with a model, the seed must be an integer. Value received: {seed}")
        
        logging.getLogger("xai").setLevel(logging.WARNING)

        self.shap_explainer = [None] * self.number_events
        for k in range(self.number_events):
            # Applying Explainer (model type)
            masker = shap.maskers.Independent(X, max_samples=X.shape[0])
            explainer_risk = shap.Explainer(self.predict, masker, feature_names=feature_names, seed=seed)
            
            # Background (faster)
            X_background = X.copy()
            if background:
                X_background = pd.DataFrame(shap.kmeans(X, background).data, columns=feature_names)

            self.shap_explainer[k] = explainer_risk(X_background)

            if plot:
                figure, ax = BaseSurvival.plot_shap(self.shap_explainer[k], index, scaler, "DeepSurv Multi-Task", dataset, seed, k)
                plt.show()

        return self.shap_explainer