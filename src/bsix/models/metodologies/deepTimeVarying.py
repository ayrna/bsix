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
    Deep survival model for time-varying covariates.

    This implementation adapts the DeepSurv formulation to interval-based
    longitudinal observations, where each sample is represented by a start time,
    stop time and event indicator. The network learns a nonlinear risk score from
    the covariates, while the partial log-likelihood is evaluated only over the
    risk intervals active at each time point. The resulting risk is then mapped to
    survival and cumulative hazard curves using a Breslow baseline estimator.

    Parameters
    ----------
    number_inputs : int
        Number of input features.
    valid_data : dict, optional
        Validation dataset containing the keys ``x``, ``e``, and ``t``.
    hidden_layers : list of int, optional
        Hidden-layer widths of the network.
    epochs : int, default=500
        Number of training epochs.
    learn_rate : float, default=0.0
        Learning rate used by the optimizer.
    lr_decay : float, default=0.0
        Learning-rate decay factor.
    l1_reg : float, default=0.0
        L1 regularization strength.
    l2_reg : float, default=0.0
        L2 regularization strength.
    momentum : float, default=0.9
        Momentum for the optimizer.
    activation : str, default="relu"
        Activation function used by the hidden layers.
    dropout : float, default=0.0
        Dropout probability.
    standardize : bool, default=True
        Whether to standardize the input features before training.
    ties : {"cox", "breslow"}, default="cox"
        Method used to handle tied event times.
    device : torch.device, optional
        Device used for training and inference.
    validation_frequency : int, default=10
        Validation interval in epochs.
    patience : int, default=2000
        Maximum number of epochs to wait for validation improvement before early
        stopping.
    improvement_threshold : float, default=0.99999
        Minimal relative improvement required to count as progress.
    patience_increase : int, default=2
        Factor by which patience is increased after improvement.
    logger : object, optional
        Logger used to track training metrics.
    verbose : bool, default=True
        Whether to print training progress.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    breslow : BreslowEstimator
        Estimator used to compute the baseline hazard and survival functions.
    network : object
        Trained neural-network model.
    optimizer : object
        Optimizer used during training.
    survival_function : ndarray of shape (n_samples, n_times)
        Estimated survival function for each sample.
    cumulative_hazard_function : ndarray of shape (n_samples, n_times)
        Estimated cumulative hazard function for each sample.
    shap_explainer : shap.Explainer
        SHAP explainer used to interpret the model output.

    Notes
    -----
    The model assumes a proportional hazards structure across time intervals,
    which is appropriate for longitudinal observations recorded as risk sets over
    ``[time_start, time_stop]``. The risk score is learned by the network and then
    converted into survival curves through the Breslow estimator.

    Examples
    --------
    >>> from bsix.models.metodologias import DeepTimeVarying
    >>> model = DeepTimeVarying(
    ...     number_inputs=10,
    ...     hidden_layers=[32, 16],
    ...     epochs=200,
    ...     learn_rate=0.01,
    ... )
    >>> model.fit(X_train, y_train)
    >>> risk = model.predict(X_test)
    """

    def __init__(self, number_inputs, valid_data=None, hidden_layers=None, epochs=500, learn_rate=0.0, lr_decay=0.0, l1_reg=0.0, l2_reg=0.0, momentum=0.9, 
                 activation="relu", dropout=0.0, standardize=True, ties="cox", device=None, validation_frequency=10, patience=2000, 
                 improvement_threshold=0.99999, patience_increase=2, logger=None, verbose=True, seed=None):
          
        """
        Initialize the time-varying DeepSurv model.

        Parameters
        ----------
        number_inputs : int
            Number of input features.
        valid_data : dict, optional
            Validation data with keys ``x``, ``e`` and ``t``.
        hidden_layers : list of int, optional
            Hidden-layer widths for the network.
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
        momentum : float, default=0.9
            Momentum for the optimizer.
        activation : str, default="relu"
            Activation function used in the hidden layers.
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
        patience : int, default=2000
            Maximum number of epochs to wait for improvement.
        improvement_threshold : float, default=0.99999
            Minimal relative improvement threshold for early stopping.
        patience_increase : int, default=2
            Factor by which patience is increased after improvement.
        logger : object, optional
            Logger used to track training metrics.
        verbose : bool, default=True
            Whether to print training progress.
        seed : int, optional
            Random seed for reproducibility.
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
    
    def _negative_log_likelihood(self, risk, t_start, t_stop, e):

        """
        Compute the negative partial log-likelihood for time-varying Cox hazards.

        Parameters
        ----------
        risk : torch.Tensor of shape (n_samples,)
            Predicted log-risk score for each sample.
        t_start : torch.Tensor of shape (n_samples,)
            Start time of each risk interval.
        t_stop : torch.Tensor of shape (n_samples,)
            End time of each risk interval.
        e : torch.Tensor of shape (n_samples,)
            Event indicator, where 1 denotes event and 0 denotes censoring.

        Returns
        -------
        torch.Tensor
            Scalar negative partial log-likelihood for the supplied interval data.
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
    
    def _get_loss(self, x, e, t_start, t_stop):

        """
        Compute the total training loss for the time-varying survival model.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
            Input feature matrix for the current batch.
        e : torch.Tensor of shape (n_samples,)
            Event indicator vector.
        t_start : torch.Tensor of shape (n_samples,)
            Start time of each risk interval.
        t_stop : torch.Tensor of shape (n_samples,)
            End time of each risk interval.

        Returns
        -------
        torch.Tensor
            Scalar total loss value including the Cox term and regularization.
        """

        risk = self.network(x)
        cox_loss = self._negative_log_likelihood(risk, t_start, t_stop, e)

        l1_loss = self._compute_l1_loss() if self.l1_reg > 0.0 else 0.0
        l2_loss = self._compute_l2_loss() if self.l2_reg > 0.0 else 0.0
        
        total_loss = cox_loss + (self.l1_reg * l1_loss) + (self.l2_reg * l2_loss)

        return total_loss
    
    def _get_concordance_index(self, x, t, e, **kwargs):

        """
        Compute the concordance index for the trained model.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Feature matrix used for prediction.
        t : array-like of shape (n_samples,)
            Observation times.
        e : array-like of shape (n_samples,)
            Event indicators.

        Returns
        -------
        float
            Concordance index value.
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
        Fit the time-varying DeepSurv model to the training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : structured array-like of shape (n_samples,)
            Target values containing the fields ``event``, ``time_start`` and
            ``time_stop``.

        Returns
        -------
        DeepTimeVarying
            The fitted estimator instance.
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
            number_inputs=self.number_inputs,
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
            X_valid = np.array(self.valid_data["x"], np.float32)
            e_valid = np.array(self.valid_data["e"], np.bool_)
            t_start_valid = np.array(self.valid_data["t"][0], np.float32)
            t_stop_valid = np.array(self.valid_data["t"][1], np.float32)
        
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
        x : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
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
        Predict the survival function for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for prediction.
        index : array-like of shape (n_samples,)
            Sample indices used for plotting and identification.
        dataset : str
            Name of the dataset used in the generated plot.
        seed : int
            Random seed used for reproducibility in plotting.
        plot : bool, default=False
            If ``True``, display the survival-function plot.

        Returns
        -------
        ndarray of shape (n_samples, n_times)
            Estimated survival function for each sample.
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
        Predict the cumulative hazard function for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for prediction.
        index : array-like of shape (n_samples,)
            Sample indices used for plotting and identification.
        dataset : str
            Name of the dataset used in the generated plot.
        seed : int
            Random seed used for reproducibility in plotting.
        plot : bool, default=False
            If ``True``, display the cumulative hazard plot.

        Returns
        -------
        ndarray of shape (n_samples, n_times)
            Estimated cumulative hazard function for each sample.
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