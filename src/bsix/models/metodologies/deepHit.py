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
from ..nets.deepNets import DeepHitFFNN

from numba import njit
from sksurv.metrics import concordance_index_censored
from torch.utils.data import TensorDataset, DataLoader
from .utils import StepFunction

warnings.filterwarnings("ignore")

@njit(fastmath=True, cache=True)
def _discretize_time_njit(t_ravel, time_grid, number_categories):

    """
    Map continuous times to their discrete bin index.

    Parameters
    ----------
    t_ravel : ndarray of shape (n_samples,)
        Array of observed times.
    time_grid : ndarray of shape (n_bins,)
        Bin edges used to discretize the original time scale.
    number_categories : int
        Total number of discrete time bins.

    Returns
    -------
    ndarray of shape (n_samples,)
        Integer bin index assigned to each time value.
    """

    idx = np.searchsorted(time_grid, t_ravel, side="right")
    
    return np.minimum(idx, number_categories - 1)

@njit(fastmath=True, cache=True)
def _get_fc_mask1_njit(e_ravel, t_bin_ravel, number_events, number_categories):

    """
    Build the one-hot mask for the event-time likelihood term.

    Parameters
    ----------
    e_ravel : ndarray of shape (n_samples,)
        Event indicator array, where values greater than zero denote observed
        events.
    t_bin_ravel : ndarray of shape (n_samples,)
        Discretized time-bin index for each sample.
    number_events : int
        Number of competing event types.
    number_categories : int
        Total number of discretized time bins.

    Returns
    -------
    ndarray of shape (n_samples, number_events, number_categories)
        One-hot mask selecting the observed event and time bin.
    """

    n = len(e_ravel)
    mask = np.zeros((n, number_events, number_categories), dtype=np.float32)
    for i in range(n):
        event = e_ravel[i]
        if event > 0:
            mask[i, event - 1, t_bin_ravel[i]] = 1.0
    return mask

@njit(fastmath=True, cache=True)
def _get_fc_mask2_njit(t_bin_ravel, number_categories):

    """
    Build the risk-set mask used in the survival ranking terms.

    Parameters
    ----------
    t_bin_ravel : ndarray of shape (n_samples,)
        Discretized time-bin index for each sample.
    number_categories : int
        Total number of discretized time bins.

    Returns
    -------
    ndarray of shape (n_samples, number_categories)
        Mask assigning ones to all time bins strictly after the sample's censoring
        or event time.
    """

    n = len(t_bin_ravel)
    mask = np.zeros((n, number_categories), dtype=np.float32)
    for i in range(n):
        for j in range(t_bin_ravel[i] + 1, number_categories):
            mask[i, j] = 1.0
    return mask

class DeepHit(BaseSurvival):

    """
    DeepHit model for discrete-time survival with competing risks.

    This implementation follows the DeepHit architecture: a shared representation
    is learned from the covariates, and one output head is trained per competing
    event to approximate the discrete probability mass function over time. The
    model combines a log-likelihood term, a ranking term and a calibration term
    in a multi-objective objective, allowing it to model the event-time
    distribution while accounting for right-censoring and competing risks.

    Parameters
    ----------
    number_inputs : int
        Number of input features.
    number_events : int
        Number of competing-event outputs, excluding the censoring class.
    number_categories : int
        Number of discrete time bins used to represent the time horizon.
    time_threshold : int, optional
        Maximum time-bin index used during prediction and training. If ``None``,
        the last available category is used.
    valid_data : dict, optional
        Validation dataset containing the keys ``x``, ``e`` and ``t``.
    hidden_layers_shared : list of int, optional
        Hidden-layer sizes of the shared sub-network.
    hidden_layers_specific : list of int, optional
        Hidden-layer sizes of the cause-specific sub-networks.
    epochs : int, default=50
        Number of training epochs.
    learn_rate : float, default=1e-4
        Learning rate for the Adam optimizer.
    lr_decay : float, default=0.0
        Learning-rate decay factor.
    alpha : float, default=1.0
        Weight of the log-likelihood loss.
    beta : float, default=1.0
        Weight of the ranking loss.
    gamma : float, default=1.0
        Weight of the calibration loss.
    ranking_sigma : float, default=0.1
        Temperature parameter used inside the ranking loss.
    l2_reg_hidden : float, default=1e-4
        L2 regularization strength applied to hidden layers.
    l1_reg_output : float, default=1e-4
        L1 regularization strength applied to the output layer.
    momentum : float, default=0.9
        Momentum for the optimizer.
    activation : str, default="relu"
        Activation function used by the hidden layers.
    dropout : float, default=0.0
        Dropout probability applied to the network.
    standardize : bool, default=True
        Whether to standardize the input features before training.
    device : torch.device, optional
        Device used for training and inference, e.g. ``cpu`` or ``cuda``.
    validation_frequency : int, default=10
        Frequency of validation checks during training.
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
    batch_size : int, default=256
        Batch size used during training.

    Attributes
    ----------
    time_grid : ndarray or None
        Discretization points used to map continuous times to bins.
    network : object
        Trained neural network model.
    optimizer : object
        Optimizer used during training.
    survival_function : ndarray of shape (n_samples, number_categories)
        Estimated survival function for each sample.
    cumulative_hazard_function : ndarray of shape (n_samples, number_categories)
        Estimated cumulative hazard function for each sample.
    shap_explainer : shap.Explainer
        SHAP explainer used for model interpretability.

    Notes
    -----
    DeepHit estimates a discrete distribution over the time axis rather than a
    single scalar risk score. The final survival curve is obtained by aggregating
    the event-specific densities over time and applying the standard survival
    transformation.

    Examples
    --------
    >>> from bsix.models.metodologies import DeepHit
    >>> model = DeepHit(
    ...     number_inputs=10,
    ...     number_events=2,
    ...     number_categories=30,
    ...     epochs=200,
    ...     learn_rate=1e-4,
    ... )
    >>> model.fit(X_train, y_train)
    >>> risk = model.predict(X_test)
    """

    def __init__(self, number_inputs, number_events, number_categories, time_threshold=None, valid_data=None, hidden_layers_shared=None, hidden_layers_specific=None,
                 epochs=50, learn_rate=1e-4, lr_decay=0.0, alpha=1.0, beta=1.0, gamma=1.0, ranking_sigma=0.1, l2_reg_hidden=1e-4, l1_reg_output=1e-4, momentum=0.9,
                 activation="relu", dropout=0.0, standardize=True, device=None, validation_frequency=10, patience=2000, improvement_threshold=0.99999,
                 patience_increase=2, logger=None, verbose=True, seed=None, batch_size=256):

        """
        Initialize the DeepHit model.

        Parameters
        ----------
        number_inputs : int
            Number of input features.
        number_events : int
            Number of competing event types.
        number_categories : int
            Number of discrete time bins.
        time_threshold : int, optional
            Maximum time-bin index to use in the output. If ``None``, the last
            category is used.
        valid_data : dict, optional
            Validation data with keys ``x``, ``e`` and ``t``.
        hidden_layers_shared : list of int, optional
            Hidden-layer widths for the shared network.
        hidden_layers_specific : list of int, optional
            Hidden-layer widths for the cause-specific networks.
        epochs : int, default=50
            Number of training epochs.
        learn_rate : float, default=1e-4
            Learning rate for the optimizer.
        lr_decay : float, default=0.0
            Learning-rate decay factor.
        alpha : float, default=1.0
            Weight of the log-likelihood loss.
        beta : float, default=1.0
            Weight of the ranking loss.
        gamma : float, default=1.0
            Weight of the calibration loss.
        ranking_sigma : float, default=0.1
            Temperature used in the ranking objective.
        l2_reg_hidden : float, default=1e-4
            L2 regularization strength for the hidden layers.
        l1_reg_output : float, default=1e-4
            L1 regularization strength for the output layer.
        momentum : float, default=0.9
            Momentum used by the optimizer.
        activation : str, default="relu"
            Activation function in the hidden layers.
        dropout : float, default=0.0
            Dropout probability.
        standardize : bool, default=True
            Whether to standardize the input features.
        device : torch.device, optional
            Device to use for training and inference.
        validation_frequency : int, default=10
            Validation interval in epochs.
        patience : int, default=2000
            Maximum number of epochs to wait for improvement.
        improvement_threshold : float, default=0.99999
            Relative improvement threshold for early stopping.
        patience_increase : int, default=2
            Multiplier applied to patience after improvement.
        logger : object, optional
            Logger used to track training metrics.
        verbose : bool, default=True
            Whether to print training progress.
        seed : int, optional
            Random seed for reproducibility.
        batch_size : int, default=256
            Size of each training batch.
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
        self.number_events = number_events
        self.number_categories = number_categories
        self.time_threshold = number_categories - 1 if time_threshold is None else max(0, min(time_threshold, number_categories - 1))

        self.momentum = momentum
        self.hidden_layers_shared = hidden_layers_shared
        self.hidden_layers_specific = hidden_layers_specific
        self.activation = activation
        self.dropout = dropout

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ranking_sigma = ranking_sigma
        self.l2_reg_hidden = l2_reg_hidden
        self.l1_reg_output = l1_reg_output

        self.epochs = epochs
        self.learn_rate = learn_rate
        self.lr_decay = lr_decay

        self.valid_data = valid_data
        self.validation_frequency = validation_frequency
        self.patience = patience
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold

        self.logger = logger
        self.verbose = verbose
        self.seed = seed
        self.batch_size = batch_size
    
        # Time discretisation (computed in fit())
        self.time_grid = None

        # Network (will be initialized in fit())
        self.network = None

        # Optimizer (will be initialized in fit())
        self.optimizer = None

    def _set_seeds(self):

        """
        Initialize random seeds for reproducibility.

        Returns
        -------
        None
            This method updates the global random generators used by Python,
            NumPy and PyTorch.
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

    def _build_time_grid(self, t):

        """
        Build the quantile-based time grid used for discretization.

        Parameters
        ----------
        t : array-like of shape (n_samples,)
            Observed times used to compute the discrete bins.

        Returns
        -------
        None
            The internal attribute ``self.time_grid`` is updated in-place.
        """

        quantiles = np.linspace(0, 1, self.number_categories + 1)[1:-1]
        self.time_grid = np.unique(np.quantile(t, quantiles)) if self.number_categories > 1 else np.array([])

    def _discretize_time(self, t):

        """
        Map continuous times to integer-based time bins.

        Parameters
        ----------
        t : array-like
            Time values to convert into discrete bins.

        Returns
        -------
        ndarray
            Discretized time indices with the same shape as the input array.
        """

        t_ravel = np.ravel(t)
        idx = _discretize_time_njit(t_ravel, self.time_grid, self.number_categories)

        return idx.reshape(t.shape).astype(np.int64)

    def _get_fc_mask1(self, e, t_bin):

        """
        Build the event-time one-hot mask used in the likelihood objective.

        Parameters
        ----------
        e : array-like of shape (n_samples,)
            Event indicator for each sample.
        t_bin : array-like of shape (n_samples,)
            Discretized time-bin index for each sample.

        Returns
        -------
        ndarray of shape (n_samples, number_events, number_categories)
            One-hot mask selecting the event and time-bin pairs for uncensored
            observations.
        """
        e_ravel = np.ravel(e).astype(np.int64)
        t_bin_ravel = np.ravel(t_bin).astype(np.int64)

        return _get_fc_mask1_njit(e_ravel, t_bin_ravel, self.number_events, self.number_categories)

    def _get_fc_mask2(self, t_bin):

        """
        Build the risk-set mask used in the censoring and ranking terms.

        Parameters
        ----------
        t_bin : array-like of shape (n_samples,)
            Discretized time-bin index for each sample.

        Returns
        -------
        ndarray of shape (n_samples, number_categories)
            Mask containing ones for every time bin strictly after the sample's
            observed time.
        """
        t_bin_ravel = np.ravel(t_bin).astype(np.int64)

        return _get_fc_mask2_njit(t_bin_ravel, self.number_categories)

    def _loss_log_likelihood(self, pmf, mask1, mask2, e):

        """
        Compute the log-likelihood loss for the DeepHit model.

        Parameters
        ----------
        pmf : torch.Tensor of shape (n_samples, number_events, number_categories)
            Estimated probability mass function for each competing event and time
            bin.
        mask1 : torch.Tensor of shape (n_samples, number_events, number_categories)
            One-hot mask for observed events.
        mask2 : torch.Tensor of shape (n_samples, number_categories)
            Risk-set mask for censored samples.
        e : torch.Tensor of shape (n_samples,)
            Event indicator tensor.

        Returns
        -------
        torch.Tensor
            Scalar log-likelihood loss contribution.
        """

        e_ocurred = (e > 0).float().view(-1, 1)

        # Uncensored: log P(T=t, K=k|x)
        uncensored = (mask1 * pmf).sum(dim=2).sum(dim=1, keepdim=True)
        uncensored = e_ocurred * torch.log(uncensored + 1e-15)

        # Censored: log sum P(T>t|x), marginalised over every competing cause.
        pmf_marginal = pmf.sum(dim=1)
        censored = (mask2 * pmf_marginal).sum(dim=1, keepdim=True)

        # Phantom logit
        survival_past_tmax = 1.0 - pmf.sum(dim=(1, 2), keepdim=True)
        survival_past_tmax = torch.clamp(survival_past_tmax, min=0.0)
        
        censored = censored + survival_past_tmax
        censored = (1.0 - e_ocurred) * torch.log(censored + 1e-15)

        return -torch.mean(uncensored + censored)

    def _loss_ranking(self, pmf, t_bin, e):

        """
        Compute the ranking loss used to enforce time-ordering consistency.

        Parameters
        ----------
        pmf : torch.Tensor of shape (n_samples, number_events, number_categories)
            Estimated event-time probability mass function.
        t_bin : torch.Tensor of shape (n_samples,)
            Discretized time-bin index for each sample.
        e : torch.Tensor of shape (n_samples,)
            Event indicator tensor.

        Returns
        -------
        torch.Tensor
            Scalar ranking loss term.
        """
        
        t_bin = t_bin.view(-1).long()
        e_col = e.view(-1, 1)
        t_col = t_bin.view(-1, 1)

        any_event_i = e_col > 0
        earlier = t_col < t_col.t()
        tie_j_censored = (t_col == t_col.t()) & (e_col.t() == 0)
        
        comparable_pairs = (any_event_i & (earlier | tie_j_censored)).float()

        # Cumulative distribution function (CDF)
        cdf = torch.cumsum(pmf, dim=2)

        cause_specific_losses = []
        for k in range(self.number_events):
            cdf_k = cdf[:, k, :]
            # Cross risk: F_i(T_j)
            cross_risk = cdf_k[:, t_bin]
            # Own risk: F_i(T_i)
            own_risk = cross_risk.diagonal()
            # Risks difference: (F_i(T_i) - F_j(T_i))
            risk_diff = own_risk.view(-1, 1) - cross_risk.t()

            cause_mask = (e == k + 1).float().view(-1, 1)
            loss_k = (comparable_pairs * cause_mask * torch.exp(-risk_diff / self.ranking_sigma)).mean(dim=1, keepdim=True)
            cause_specific_losses.append(loss_k)

        eta = torch.stack(cause_specific_losses, dim=1).reshape(-1, self.number_events).sum(dim=1, keepdim=True)
        return torch.sum(eta)

    def _compute_l2_loss_hidden(self):

        """
        Compute the L2 regularization contribution from hidden layers.

        Returns
        -------
        float or torch.Tensor
            Total L2 penalty over the regularized hidden parameters.
        """

        l2_loss = 0.0
        for param in self._l2_reg_params:
            l2_loss += torch.sum(param ** 2)

        return l2_loss

    def _compute_l1_loss_output(self):

        """
        Compute the L1 regularization contribution from the output layer.

        Returns
        -------
        torch.Tensor
            Total L1 penalty for the output weights.
        """

        return torch.sum(torch.abs(self.network.output_layer.weight))

    def _get_loss(self, x, e, t, mask1, mask2):

        """
        Compute the full DeepHit training loss, including regularization.

        Parameters
        ----------
        x : torch.Tensor of shape (n_batch, n_features)
            Batch of input covariates.
        e : torch.Tensor of shape (n_batch,)
            Event indicator tensor.
        t : torch.Tensor of shape (n_batch,)
            Discretized event or censoring times.
        mask1 : torch.Tensor of shape (n_batch, number_events, number_categories)
            Event-time mask used in the likelihood term.
        mask2 : torch.Tensor of shape (n_batch, number_categories)
            Risk-set mask used in the censoring and ranking terms.

        Returns
        -------
        torch.Tensor
            Scalar total loss value.
        """

        pmf = self.network(x)

        loss1 = self._loss_log_likelihood(pmf, mask1, mask2, e)
        loss2 = self._loss_ranking(pmf, t, e)

        total_loss = (self.alpha * loss1) + (self.beta * loss2)

        if self.l2_reg_hidden > 0.0:
            total_loss += self.l2_reg_hidden * self._compute_l2_loss_hidden()
        if self.l1_reg_output > 0.0:
            total_loss += self.l1_reg_output * self._compute_l1_loss_output()

        return total_loss

    def _get_concordance_index(self, x, t, e):

        """
        Compute the concordance index for a given batch of samples.

        Parameters
        ----------
        x : array-like or torch.Tensor
            Feature matrix used to infer the model output.
        t : array-like of shape (n_samples,)
            Observation times.
        e : array-like of shape (n_samples,)
            Event indicators.

        Returns
        -------
        torch.Tensor
            Concordance index value as a tensor on the model device.
        """

        self.network.eval()
        with torch.no_grad():
            pmf = self.network(x).cpu().numpy()

        cumulative_risk = pmf[:, :, :self.time_threshold].sum(axis=2).sum(axis=1)

        e_occurred = e > 0

        c_index = concordance_index_censored(e_occurred, t, cumulative_risk)[0]

        return torch.tensor(c_index, dtype=torch.float32, device=self.device)

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
        Fit the DeepHit model to the training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : structured array-like of shape (n_samples,)
            Target data containing the fields ``event`` and ``time``.
        **kwargs
            Additional keyword arguments accepted for compatibility with the
            training interface.

        Returns
        -------
        DeepHit
            The fitted estimator instance.
        """

        # Set random seeds
        self._set_seeds()

        logger = self.logger if self.logger is not None else DeepMultiTaskLogger("DeepHit")

        # Build network
        self.network = DeepHitFFNN(
            number_inputs=self.number_inputs,
            number_events=self.number_events,
            number_categories=self.number_categories,
            hidden_layers_shared=self.hidden_layers_shared,
            hidden_layers_specific=self.hidden_layers_specific,
            activation=self.activation,
            dropout=self.dropout,
        ).to(self.device)

        # Cache the parameters subject to L2 regularization.
        self._l2_reg_params = [
            param for name, param in self.network.named_parameters()
            if "output_layer" not in name and param.dim() > 1
        ]

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
        e_train = np.array([item[0] for item in y_train], dtype=np.int64)
        t_train = np.array([item[1] for item in y_train], dtype=np.float32)

        # Discretise time
        self._build_time_grid(t_train)
        tb_train = self._discretize_time(t_train)

        # Build masks for the loss functions
        mask1_train = self._get_fc_mask1(e_train, tb_train)
        mask2_train = self._get_fc_mask2(tb_train)

        if self.valid_data:
            X_val = np.array(self.valid_data["x"], np.float32)
            e_val = np.array(self.valid_data["e"], np.int64)
            t_val = np.array(self.valid_data["t"], np.float32)

            tb_valid = self._discretize_time(t_val)

            mask1_valid = self._get_fc_mask1(e_val, tb_valid)
            mask2_valid = self._get_fc_mask2(tb_valid)

        # Convert to tensors
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        e_train_tensor = torch.tensor(e_train, dtype=torch.float32, device=self.device)
        #t_train_tensor = torch.tensor(t_train, dtype=torch.float32, device=self.device)
        tb_train_tensor = torch.tensor(tb_train, dtype=torch.float32, device=self.device)
        m1_train_tensor = torch.tensor(mask1_train, dtype=torch.float32, device=self.device)
        m2_train_tensor = torch.tensor(mask2_train, dtype=torch.float32, device=self.device)

        if self.valid_data:
            x_valid_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            e_valid_tensor = torch.tensor(e_val, dtype=torch.float32, device=self.device)
            #t_valid_tensor = torch.tensor(t_val, dtype=torch.float32, device=self.device)
            tb_valid_tensor = torch.tensor(tb_valid, dtype=torch.float32, device=self.device)
            m1_valid_tensor = torch.tensor(mask1_valid, dtype=torch.float32, device=self.device)
            m2_valid_tensor = torch.tensor(mask2_valid, dtype=torch.float32, device=self.device)

        if self.standardize:
            x_train_tensor = self._standardize_x(x_train_tensor)

            if self.valid_data:
                x_valid_tensor = self._standardize_x(x_valid_tensor)

        # Dataloader for training
        train_dataset = TensorDataset(x_train_tensor, e_train_tensor, tb_train_tensor, m1_train_tensor, m2_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize optimizer
        self.optimizer = tt.optim.Adam(
            params=self.network.parameters(),
            lr=self.learn_rate,
        )

        # Training metrics
        best_validation_loss = np.inf
        best_params = None
        best_params_idx = -1

        patience = self.patience

        start = time.time()

        for epoch in range(self.epochs):
            # Learning rate decay
            lr = self.learn_rate / (1 + epoch * self.lr_decay)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            logger.logValue("lr", lr, epoch)

            # Batches
            self.network.train()
            epoch_loss = 0.0
            
            for b_x, b_e, b_t, b_m1, b_m2 in train_loader:
                self.optimizer.zero_grad()
                loss = self._get_loss(b_x, b_e, b_t, b_m1, b_m2)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.logValue("loss", avg_epoch_loss, epoch)

            # Validation
            if epoch % self.validation_frequency == 0:
                
                # Calculate training C-index
                ci_train = self._get_concordance_index(x_train_tensor, t_train, e_train)
                logger.logValue("c-index", ci_train, epoch)

                if self.valid_data:
                    self.network.eval()
                    with torch.no_grad():
                        validation_loss = self._get_loss(
                            x_valid_tensor, e_valid_tensor, tb_valid_tensor, 
                            m1_valid_tensor, m2_valid_tensor
                        )
                        logger.logValue("valid_loss", validation_loss.item(), epoch)

                    ci_valid = self._get_concordance_index(x_valid_tensor, t_val, e_val)
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

                if self.verbose:
                    if self.valid_data:
                        logger.print_progress_bar(epoch, self.epochs, avg_epoch_loss, validation_loss.item(), ci_train, ci_valid)
                    else:
                        logger.print_progress_bar(epoch, self.epochs, loss=avg_epoch_loss, ci=ci_train)

            if patience <= epoch:
                print("Early stopping at epoch %d" % epoch)
                break

        if self.verbose:
            logger.logMessage(f"Finished Training with {epoch + 1} iterations in {time.time() - start:.2f}s")

        logger.shutdown()

        logger.history["best_valid_loss"] = best_validation_loss
        logger.history["best_params"] = best_params
        logger.history["best_params_idx"] = best_params_idx

        self.history = logger.history

        return self

    def predict(self, x):

        """
        Predict the overall risk for the given data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Input feature matrix for prediction.

        Returns
        -------
        ndarray of shape (n_samples,)
            Estimated cumulative risk for each sample.
        """

        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            if self.standardize:
                x_tensor = self._standardize_x(x_tensor)
            pmf = self.network(x_tensor).cpu().numpy()

        cumulative_risk = pmf[:, :, :self.time_threshold].sum(axis=2).sum(axis=1)
        
        return cumulative_risk

    # ----------------------
    # Base Survival methods
    # ----------------------
    def predict_survival_function(self, X, index, dataset, seed, plot=False):

        """
        Predict the overall survival function for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
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
            Array of ``StepFunction`` objects with the estimated survival curve
            for each sample.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_survival_function` with a model, the seed must be an integer. Value received: {seed}")

        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            if self.standardize:
                x_tensor = self._standardize_x(x_tensor)
            pmf = self.network(x_tensor).cpu().numpy()

        cumulative_incidence = np.cumsum(pmf.sum(axis=1), axis=1)

        survival_function = 1.0 - cumulative_incidence
        self.survival_function = np.array([StepFunction(X=self.time_grid, y=individual_survival, is_survival=True) for individual_survival in survival_function])
        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.survival_function, index, "DeepHit", dataset, "Survival", seed)
            plt.show()

        return self.survival_function

    def predict_cumulative_hazard_function(self, X, index, dataset, seed, event=None, plot=False):

        """
        Predict the cumulative incidence or cumulative hazard function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        index : array-like of shape (n_samples,)
            Sample indices used for plotting.
        dataset : str
            Name of the dataset used in the generated plot.
        seed : int
            Random seed for reproducibility.
        event : int, optional
            Competing event index to use for cause-specific estimation. If
            ``None``, the overall cumulative incidence is returned.
        plot : bool, default=False
            If ``True``, display the cumulative hazard plot.

        Returns
        -------
        ndarray of shape (n_samples, number_categories)
            Estimated cumulative incidence function, either overall or cause-
            specific depending on ``event``.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_cumulative_hazard_function` with a model, the seed must be an integer. Value received: {seed}")

        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            if self.standardize:
                x_tensor = self._standardize_x(x_tensor)
            pmf = self.network(x_tensor).cpu().numpy()

        pmf_for_event = pmf.sum(axis=1) if event is None else pmf[:, event - 1, :]
        cumulative_incidence = np.cumsum(pmf_for_event, axis=1)

        self.cumulative_hazard_function = cumulative_incidence

        if plot:
            figure, ax = self._plot_survival_hazard_functions(self.cumulative_hazard_function, index, "DeepHit", dataset, "CumulativeRisk", seed)
            plt.show()

        return self.cumulative_hazard_function

    # ----------------------
    # XAI
    # ----------------------
    def calculate_xai(self, X, index, scaler, dataset, seed, feature_names, event=1, background=False, plot=False):

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
            figure, ax = BaseSurvival.plot_shap(self.shap_explainer, index, scaler, "DeepHit", dataset, seed)
            plt.show()

        return self.shap_explainer