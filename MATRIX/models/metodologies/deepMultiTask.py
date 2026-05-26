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
    Deep Multi-Task model.
    """
        
    def __init__(self, num_inputs, valid_data=None, hidden_layers=None, epochs=500, learn_rate=0.0, lr_decay=0.0, l1_reg=0.0, l2_reg=0.0, cox_reg=0.0,
                 momentum=0.9, activation="relu", dropout=0.0, standardize=True, ties="cox", device=None, validation_frequency=10, 
                 patience=500, improvement_threshold=0.99999, patience_increase=25, logger=None, verbose=True, seed=None, coef_likelihood=[1.0]):
        
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
    
    def _negative_log_likelihood(self, risk, t, e):

        """
        Compute the negative partial log-likelihood for Cox proportional hazards.
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
    
    def _get_loss(self, x, e, t):

        """
        Compute total loss including regularization.
        """

        risk = self.network(x)[:, :self.number_progressions]
        
        cox_loss = []
        for p in range(self.number_progressions):
            cox_loss.append(self._negative_log_likelihood(risk[:, p], t[:, p], e[:, p]) * self.coef_likelihood[p])
        cox_loss = torch.stack(cox_loss)
        
        l1_loss = self._compute_l1_loss() if self.l1_reg > 0.0 else 0.0
        l2_loss = self._compute_l2_loss() if self.l2_reg > 0.0 else 0.0
        
        total_loss = (self.cox_reg * torch.sum(cox_loss)) + (self.l1_reg * l1_loss) + (self.l2_reg * l2_loss)

        return total_loss
    
    def _get_concordance_index(self, x, t, e, **kwargs):

        """
        Calculate concordance index (C-index) for model predictions.
        """

        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            if self.standardize:
                x_tensor = self._standardize_x(x_tensor)
            risk = self.network(x_tensor)[:, :self.number_progressions].cpu().numpy()
            
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
        """
        
        # Set random seeds
        self._set_seeds()

        # Set the number of progressions
        self.number_progressions = y_train.shape[1]

        # Breslow estimator for baseline hazards
        self.breslow = [BreslowEstimator() for _ in range(self.number_progressions)]

        if self.logger is None:
            logger = DeepMultiTaskLogger("DeepMultiTask")
        
        # Build network
        self.network = DeepMultiTaskFFNN(
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
        e_train = []
        t_train = []
        for p in range(self.number_progressions):
            e_train.append(np.array([evento for evento, _ in y_train[:, p]], np.bool_))
            t_train.append(np.array([tiempo for _, tiempo in y_train[:, p]], np.float32))
        e_train = np.array(e_train, np.bool_).T
        t_train = np.array(t_train, np.float32).T
        
        if self.valid_data:
            X_val = np.array(self.valid_data["x"], np.float32)
            e_val = []
            t_val = []
            
            for p in range(self.number_progressions):
                e_val.append(np.array(self.valid_data["e"][:, p], np.bool_))
                t_val.append(np.array(self.valid_data["t"][:, p], np.float32))
            e_val = np.array(e_val, np.bool_).T
            t_val = np.array(t_val, np.float32).T
        
        # Convert to tensors
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        e_train_tensor = torch.tensor(e_train, dtype=torch.long, device=self.device)
        t_train_tensor = torch.tensor(t_train, dtype=torch.float32, device=self.device)

        if self.valid_data:
            x_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            e_val_tensor = torch.tensor(e_val, dtype=torch.long, device=self.device)
            t_val_tensor = torch.tensor(t_val, dtype=torch.float32, device=self.device)

        if self.standardize:
            x_train_tensor = self._standardize_x(x_train_tensor)
            if self.valid_data:
                x_val_tensor = self._standardize_x(x_val_tensor)

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
                    validation_loss = self._get_loss(x_val_tensor, e_val_tensor, t_val_tensor)
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
        
        # Compute baseline hazards with training data
        for p in range(self.number_progressions):
            self.breslow[p].fit(self.predict(X_train)[:, p], e_train[:, p], t_train[:, p])

        logger.shutdown()
        
        logger.history["best_val_loss"] = best_validation_loss
        logger.history["best_params"] = best_params
        logger.history["best_params_idx"] = best_params_idx
        
        self.history = logger.history
        
        return self
    
    def predict(self, x):

        """
        Predict risk scores for the given data.
        """

        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            if self.standardize:
                x_tensor = self._standardize_x(x_tensor)
            risk = self.network(x_tensor)[:, :self.number_progressions].cpu().numpy()
            
        return risk
    
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
        risk = self.predict(X)

        self.survival_functions = []
        for p in range(self.number_progressions):
            survival_function = self.breslow[p].get_survival_function(risk[:, p])
            self.survival_functions.append(survival_function)

            if plot:
                figure, ax = self._plot_survival_hazard_functions(survival_function, index, "DeepSurv Multi-Task", dataset, "Survival", seed, p)
                plt.show()
        
        return self.survival_functions

    def predict_cumulative_hazard_function(self, X, index, dataset, seed, plot=False):
        
        """
        H(x,t) = H₀(t) × exp(βᵀx).
        """

        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(f"When using `predict_cumulative_hazard_function` with a model, the seed must be an integer. Value received: {seed}")
        
        risk = self.predict(X)
        
        self.cumulative_hazard_functions = []
        for p in range(self.number_progressions):
            cumulative_hazard_function = self.breslow[p].get_cumulative_hazard_function(risk[:, p])
            self.cumulative_hazard_functions.append(cumulative_hazard_function)

            if plot:
                figure, ax = self._plot_survival_hazard_functions(cumulative_hazard_function, index, "DeepSurv Multi-Task", dataset, "CumulativeRisk", seed, p)
                plt.show()

        return self.cumulative_hazard_functions

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

        self.shap_explainer = [None] * self.number_progressions
        for p in range(self.number_progressions):
            # Applying Explainer (model type)
            explainer_risk = shap.Explainer(self.predict, X, feature_names=feature_names, seed=seed)
            
            # Background (faster)
            X_background = X.copy()
            if background:
                X_background = pd.DataFrame(shap.kmeans(X, background).data, columns=feature_names)

            self.shap_explainer[p] = explainer_risk(X_background)

            if plot:
                figure, ax = BaseSurvival.plot_shap(self.shap_explainer[p], index, scaler, "DeepSurv Multi-Task", dataset, seed, p)
                plt.show()

        return self.shap_explainer