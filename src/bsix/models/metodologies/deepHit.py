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

from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")

class DeepHit(BaseSurvival):

    """
    Deep Survival model with competing risks (DeepHit).

    Unlike :class:`DeepSurv`, DeepHit does not assume a Cox proportional
    hazards form and therefore needs no Breslow estimator: the network
    directly outputs the joint probability mass function
    P(T = t, K = k | x) over every competing event ``k`` and discrete time
    bin ``t``. Survival and cumulative-incidence curves are obtained
    directly from that distribution.

    Parameters
    ----------
    number_inputs : int
        Number of input features.
    number_events : int
        Number of competing events (this does not include the censoring
        label).
    number_categories : int
        Number of discrete time bins, i.e. the dimension of the time
        horizon of interest |T| (the output dimension of the network).
    valid_data : dict, default = ``None``
        Validation data in the form of a dictionary with keys "x", "e", and
        "t" for features, (cause) events and times, respectively. ``e``
        should be an integer in ``{0, 1, ..., number_events}``, with ``0``
        denoting censoring.
    h_dim_shared : int, default =32
        Number of hidden units per layer in the shared sub-network.
    num_layers_shared : int, default =1
        Number of fully-connected layers in the shared sub-network.
    h_dim_cs : int, default =32
        Number of hidden units per layer in each cause-specific sub-network.
    num_layers_cs : int, default =1
        Number of fully-connected layers in each cause-specific sub-network.
    epochs : int, default =500
        Number of training epochs.
    learn_rate : float, default =1e-4
        Learning rate for the Adam optimizer.
    lr_decay : float, default =0.0
        Learning rate decay factor.
    alpha : float, default =1.0
        Weight of the log-likelihood loss (Loss 1).
    beta : float, default =1.0
        Weight of the ranking loss (Loss 2).
    gamma : float, default =1.0
        Weight of the calibration loss (Loss 3). The calibration loss is an
        experimental addition that is not part of the original DeepHit
        paper; set ``gamma=0`` to disable it.
    ranking_sigma : float, default =0.1
        Temperature used inside the ranking loss.
    l2_reg_hidden : float, default =1e-4
        L2 regularisation strength applied to the shared and cause-specific
        hidden layers.
    l1_reg_output : float, default =1e-4
        L1 regularisation strength applied to the output layer.
    activation : str, default = ``"relu"``
        Activation function to use in the hidden layers. ``"relu"``,
        ``"elu"`` or ``"tanh"``.
    dropout : float, default =0.0
        Dropout rate for regularisation.
    standardize : bool, default = ``True``
        Whether to standardize input features.
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
    time_grid_ : array-like, shape (number_categories - 1,)
        Bin edges (in the original time scale) used to discretise
        continuous event/censoring times into ``number_categories`` bins.
    survival_function : array-like, shape (n_samples, number_categories)
        Estimated overall survival function, marginalised over every
        competing cause.
    cumulative_hazard_function : array-like, shape (n_samples, number_categories)
        Estimated cumulative incidence function (overall, or cause-specific
        when ``event`` is given).
    shap_explainer : shap.Explainer
        SHAP explainer for model interpretability.

    Examples
    --------
    .. code:: python

        from bsix.models.metodologies import DeepHit
        model = DeepHit(number_inputs=10, number_events=2, number_categories=30, epochs=200, learn_rate=1e-4)
        model.fit(X_train, y_train)
    """

    def __init__(self, number_inputs, number_events, number_categories, time_threshold=None, valid_data=None,
                 hidden_layers_shared=None, hidden_layers_specific=None,
                 epochs=500, learn_rate=1e-4, lr_decay=0.0,
                 alpha=1.0, beta=1.0, gamma=1.0, ranking_sigma=0.1,
                 l2_reg_hidden=1e-4, l1_reg_output=1e-4, momentum=0.9,
                 activation="relu", dropout=0.0, standardize=True, device=None,
                 validation_frequency=10, patience=2000, improvement_threshold=0.99999,
                 patience_increase=2, logger=None, verbose=True, seed=None):

        """
        Initialise model with specified parameters.
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

        self.time_threshold = (number_categories - 20) if time_threshold is None else time_threshold # Calculate at 80

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

        # Time discretisation (computed in fit())
        self.time_grid_ = None

        # Network (will be initialized in fit())
        self.network = None

        # Optimizer (will be initialized in fit())
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

    # ------------------------------------------------------------------
    # Time discretisation helpers
    # ------------------------------------------------------------------
    def _build_time_grid(self, t):

        """
        Build the bin edges used to discretise continuous event/censoring
        times into ``self.number_categories`` discrete bins (quantile-based).
        """

        quantiles = np.linspace(0, 1, self.number_categories + 1)[1:-1]
        self.time_grid_ = np.unique(np.quantile(t, quantiles)) if self.number_categories > 1 else np.array([])

    def _discretize_time(self, t):

        """
        Map continuous times to an integer bin index in
        ``[0, number_categories - 1]`` using ``self.time_grid_``.
        """

        idx = np.searchsorted(self.time_grid_, t, side="right")
        return np.clip(idx, 0, self.number_categories - 1).astype(np.int64)

    # ------------------------------------------------------------------
    # Mask construction (translated from the original DeepHit data
    # pre-processing utilities)
    # ------------------------------------------------------------------
    def _get_fc_mask1(self, e, t_bin):

        """
        One-hot mask over (event, time-bin), used to pick out
        P(T = t, K = k | x) for every *uncensored* subject in the
        log-likelihood loss (Loss 1).
        """
        e_ravel = np.ravel(e).astype(np.int64)
        t_bin_ravel = np.ravel(t_bin).astype(np.int64)

        n = len(e_ravel)
        mask = np.zeros((n, self.number_events, self.number_categories), dtype=np.float32)
        observed = np.where(e_ravel > 0)[0]
        mask[observed, e_ravel[observed] - 1, t_bin_ravel[observed]] = 1.0
        return mask

    def _get_fc_mask2(self, t_bin):

        """
        Risk-set mask: ``mask2[i, j] = 1`` for every time-bin ``j`` strictly
        after subject ``i``'s own time-bin. Used for the censored-survival
        term of the log-likelihood loss, the ranking loss and the
        calibration loss.
        """
        t_bin_ravel = np.ravel(t_bin).astype(np.int64)

        n = len(t_bin_ravel)
        mask = np.zeros((n, self.number_categories), dtype=np.float32)
        for i in range(n):
            mask[i, t_bin_ravel[i] + 1:] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    @staticmethod
    def _log(x):
        return torch.log(x + 1e-15)

    def _loss_log_likelihood(self, pmf, mask1, mask2, e):

        """
        Loss 1 -- Log-likelihood loss. Includes the log-likelihood of
        subjects who are censored.
        """

        I_1 = (e > 0).float().view(-1, 1)

        # Uncensored: log P(T=t, K=k|x)
        tmp1 = (mask1 * pmf).sum(dim=2).sum(dim=1, keepdim=True)
        tmp1 = I_1 * self._log(tmp1)

        # Censored: log sum P(T>t|x), marginalised over every competing cause.
        #
        # NOTE: the reference TensorFlow implementation multiplies a
        # (batch, number_categories) mask directly against the (batch,
        # number_events, number_categories) output tensor. That only broadcasts
        # correctly when number_events == 1; for true competing risks
        # (number_events > 1) it raises a shape mismatch at run time. Here we
        # marginalise `pmf` over the event axis first, which is well
        # defined for any number of competing risks and matches the
        # intended quantity, P(T>t|x).
        pmf_marginal = pmf.sum(dim=1)
        tmp2 = (mask2 * pmf_marginal).sum(dim=1, keepdim=True)
        tmp2 = (1.0 - I_1) * self._log(tmp2)

        return -torch.mean(tmp1 + tmp2)

    def _loss_ranking(self, pmf, mask2, t, e):

        """
        Loss 2 -- Ranking loss. Computed only for acceptable pairs (see the
        DeepHit paper for the definition).
        """

        batch = pmf.shape[0]
        one_vector = torch.ones((batch, 1), dtype=torch.float32, device=pmf.device)
        t_col = t.view(-1, 1)

        etas = []
        for i in range(self.number_events):
            I_2 = (e == i + 1).float().view(-1)
            I_2 = torch.diag(I_2)

            tmp_e = pmf[:, i, :]                                    # event-specific joint prob.

            R = tmp_e @ mask2.t()                                   # r_i(T_j)
            diag_R = torch.diag(R).view(-1, 1)
            R = one_vector @ diag_R.t() - R                         # R_{ij} = r_j(T_j) - r_i(T_j)
            R = R.t()                                               # R_{ij} = r_i(T_i) - r_j(T_i)

            T = torch.relu(torch.sign(one_vector @ t_col.t() - t_col @ one_vector.t()))
            T = I_2 @ T                                             # only event-i subjects keep T_{ij}=1

            tmp_eta = (T * torch.exp(-R / self.ranking_sigma)).mean(dim=1, keepdim=True)
            etas.append(tmp_eta)

        eta = torch.stack(etas, dim=1).reshape(-1, self.number_events).mean(dim=1, keepdim=True)

        return torch.sum(eta)

    def _loss_calibration(self, pmf, mask2, e):

        """
        Loss 3 -- Calibration loss. Reduces the calibration loss; this is
        an experimental addition that is not included in the original
        DeepHit paper.
        """

        etas = []
        for i in range(self.number_events):
            I_2 = (e == i + 1).float().view(-1, 1)
            tmp_e = pmf[:, i, :]
            r = (tmp_e * mask2).sum(dim=0)
            tmp_eta = ((r.unsqueeze(0) - I_2) ** 2).mean(dim=1, keepdim=True)
            etas.append(tmp_eta)

        eta = torch.stack(etas, dim=1).reshape(-1, self.number_events).mean(dim=1, keepdim=True)

        return torch.sum(eta)

    def _compute_l2_loss_hidden(self):

        """
        Compute L2 regularization loss over the shared and cause-specific
        hidden layers.
        """

        l2_loss = 0.0
        for name, param in self.network.named_parameters():
            if "output_layer" not in name and param.dim() > 1:
                l2_loss += torch.sum(param ** 2)

        return l2_loss

    def _compute_l1_loss_output(self):

        """
        Compute L1 regularization loss over the output layer.
        """

        return torch.sum(torch.abs(self.network.output_layer.weight))

    def _get_loss(self, x, e, t, mask1, mask2):

        """
        Compute total loss including regularization.
        """

        pmf = self.network(x)

        loss1 = []
        loss2 = []
        for p in range(self.number_progressions):
            loss1.append(self._loss_log_likelihood(pmf[:, p, :], mask1[:, p, :], mask2[:, p, :], e[:, p]))
            loss2.append(self._loss_ranking(pmf[:, p, :], mask2[:, p, :], t[:, p], e[:, p]))
        loss1 = torch.stack(loss1)
        loss2 = torch.stack(loss2)
        ###loss3 = self._loss_calibration(pmf, mask2, e)
        ###total_loss = self.alpha * loss1 + self.beta * loss2 ###+ self.gamma * loss3

        total_loss = (self.alpha * torch.sum(loss1)) + (self.beta * torch.sum(loss2))

        if self.l2_reg_hidden > 0.0:
            total_loss = total_loss + self.l2_reg_hidden * self._compute_l2_loss_hidden()
        if self.l1_reg_output > 0.0:
            total_loss = total_loss + self.l1_reg_output * self._compute_l1_loss_output()

        return total_loss

    def _get_concordance_index(self, x, t, e, event=1):

        """
        Calculate the cause-specific concordance index (C-index) for the
        model's predictions, treating every cause other than ``event``
        (including censoring) as censored.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Input data.
        t : array-like, shape (n_samples,)
            Times.
        e : array-like, shape (n_samples,)
            Cause labels (0 = censored, 1..number_events = competing causes).
        event : int, default =1
            Which competing event to evaluate.

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
            pmf = self.network(x_tensor).cpu().numpy()

        risk = pmf[:, :, :, :self.time_threshold + 1].sum(axis=(2, 3))
        
        c_index_censored = []
        for p in range(self.number_progressions):
            c_index_censored.append(torch.tensor(concordance_index_censored(e[:, p], t[:, p], risk[:, p])[0], dtype=torch.float32, device=self.device))
        c_index_censored = torch.stack(c_index_censored)
        
        return c_index_censored

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
            Target training values (cause, time). ``cause`` should be an
            integer in ``{0, 1, ..., number_events}``, with ``0`` denoting
            censoring.

        Returns
        -------
        self : DeepHit
            Fitted estimator.
        """

        # Set random seeds
        self._set_seeds()

        # Set the number of progressions
        self.number_progressions = y_train.shape[1]

        if self.logger is None:
            logger = DeepMultiTaskLogger("DeepHit")

        # Build network
        self.network = DeepHitFFNN(
            number_inputs=self.number_inputs,
            number_progressions=self.number_progressions,
            number_events=self.number_events,
            number_categories=self.number_categories,
            hidden_layers_shared=self.hidden_layers_shared,
            hidden_layers_specific=self.hidden_layers_specific,
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
        for p in range(self.number_progressions):
            e_train.append(np.array([evento for evento, _ in y_train[:, p]], np.bool_))
            t_train.append(np.array([tiempo for _, tiempo in y_train[:, p]], np.float32))
        
        e_train = np.array(e_train, np.bool_).T
        t_train = np.array(t_train, np.float32).T

        # Discretise time
        self._build_time_grid(t_train)
        tb_train = self._discretize_time(t_train)

        # Build masks for the loss functions
        mask1_train = []
        mask2_train = []
        for p in range(self.number_progressions):
            mask1_train.append(self._get_fc_mask1(e_train[:, p], tb_train[:, p]))
            mask2_train.append(self._get_fc_mask2(tb_train[:, p]))

        mask1_train = np.stack(mask1_train, axis=1).astype(np.float32)
        mask2_train = np.stack(mask2_train, axis=1).astype(np.float32)

        if self.valid_data:
            X_val = np.array(self.valid_data["x"], np.float32)
            e_val = []
            t_val = []
            
            for p in range(self.number_progressions):
                e_val.append(np.array(self.valid_data["e"][:, p], np.bool_))
                t_val.append(np.array(self.valid_data["t"][:, p], np.float32))
            e_val = np.array(e_val, np.bool_).T
            t_val = np.array(t_val, np.float32).T

            tb_valid = self._discretize_time(t_val)
            
            mask1_valid = []
            mask2_valid = []
            for p in range(self.number_progressions):
                mask1_valid.append(self._get_fc_mask1(e_val[:, p], tb_valid[:, p]))
                mask2_valid.append(self._get_fc_mask2(tb_valid[:, p]))

            mask1_valid = np.stack(mask1_valid, axis=1).astype(np.float32)
            mask2_valid = np.stack(mask2_valid, axis=1).astype(np.float32)

        # Convert to tensors
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        e_train_tensor = torch.tensor(e_train, dtype=torch.float32, device=self.device)
        t_train_tensor = torch.tensor(t_train, dtype=torch.float32, device=self.device)
        m1_train_tensor = torch.tensor(mask1_train, dtype=torch.float32, device=self.device)
        m2_train_tensor = torch.tensor(mask2_train, dtype=torch.float32, device=self.device)

        if self.valid_data:
            x_valid_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            e_valid_tensor = torch.tensor(e_val, dtype=torch.float32, device=self.device)
            t_valid_tensor = torch.tensor(t_val, dtype=torch.float32, device=self.device)
            m1_valid_tensor = torch.tensor(mask1_valid, dtype=torch.float32, device=self.device)
            m2_valid_tensor = torch.tensor(mask2_valid, dtype=torch.float32, device=self.device)

        if self.standardize:
            x_train_tensor = self._standardize_x(x_train_tensor)

            if self.valid_data:
                x_valid_tensor = self._standardize_x(x_valid_tensor)

        # Initialize optimizer (the original DeepHit paper uses Adam)
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

            loss = self._get_loss(x_train_tensor, e_train_tensor, t_train_tensor, m1_train_tensor, m2_train_tensor)
            loss.backward()
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
                    validation_loss = self._get_loss(x_valid_tensor, e_valid_tensor, t_valid_tensor, m1_valid_tensor, m2_valid_tensor)
                    logger.logValue("valid_loss", validation_loss.item(), epoch)

                ci_valid = self._get_concordance_index(X_val, t_val, e_val)
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

        logger.shutdown()

        logger.history["best_valid_loss"] = best_validation_loss
        logger.history["best_params"] = best_params
        logger.history["best_params_idx"] = best_params_idx

        self.history = logger.history

        return self

    def predict(self, x):

        """
        Predict the joint probability mass function P(T = t, K = k | x) for
        the given data.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        pmf : array-like, shape (n_samples, number_events, number_categories)
            Predicted joint probability of failing from cause k at
            time-bin t.
        """

        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            if self.standardize:
                x_tensor = self._standardize_x(x_tensor)
            pmf = self.network(x_tensor).cpu().numpy()

        risk = pmf[:, :, :, :self.time_threshold + 1].sum(axis=(2, 3))
        
        return risk

    # ----------------------
    # Base Survival methods
    # ----------------------
    def predict_survival_function(self, X, index, dataset, seed, plot=False):

        """
        Predict the overall survival function (marginalised over every
        competing cause) for the given data.

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
        survival_function : array-like, shape (n_samples, number_categories)
            Predicted overall survival function.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_survival_function` with a model, the seed must be an integer. Value received: {seed}")

        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            pmf = self.network(x_tensor).cpu().numpy()
        
        self.survival_functions = []
        for p in range(self.number_progressions):
            cumulative_incidence = pmf[:, p, :, :self.time_threshold + 1].sum(axis=(1, 2))
            survival_function = 1.0 - cumulative_incidence
            self.survival_functions.append(survival_function)

            if plot:
                figure, ax = self._plot_survival_hazard_functions(survival_function, index, "DeepHit", dataset, "Survival", seed, p)
                plt.show()

        return self.survival_functions

    def predict_cumulative_hazard_function(self, X, index, dataset, seed, event=None, plot=False):

        """
        Predict the cumulative incidence function for the given data.

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
        event : int, default = ``None``
            Which competing cause to compute the cumulative incidence
            function for. If ``None``, returns the overall cumulative
            incidence (1 - overall survival), summed across every cause.
        plot : bool, default = ``False``
            Whether to plot the cumulative incidence function.

        Returns
        -------
        cumulative_hazard_function : array-like, shape (n_samples, number_categories)
            Predicted (overall or cause-specific) cumulative incidence
            function.
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_cumulative_hazard_function` with a model, the seed must be an integer. Value received: {seed}")

        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            pmf = self.network(x_tensor).cpu().numpy()
        
        self.cumulative_hazard_functions = []
        for p in range(self.number_progressions):
            cumulative_hazard_function = pmf[:, p, :, :self.time_threshold + 1].sum(axis=(1, 2))
            self.cumulative_hazard_functions.append(cumulative_hazard_function)

            if plot:
                figure, ax = self._plot_survival_hazard_functions(cumulative_hazard_function, index, "DeepHit", dataset, "CumulativeRisk", seed, p)
                plt.show()

        return self.cumulative_hazard_functions

    # ----------------------
    # XAI
    # ----------------------
    def calculate_xai(self, X, index, scaler, dataset, seed, feature_names, event=1, background=False, plot=False):

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
        event : int, default =1
            Which competing cause to explain.
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
            figure, ax = BaseSurvival.plot_shap(self.shap_explainer, index, scaler, "DeepHit", dataset, seed)
            plt.show()

        return self.shap_explainer