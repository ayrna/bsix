import logging
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
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")

class DeepSurv(BaseSurvival):

    """
    Deep Survival model.
    """

    def __init__(self, num_inputs, valid_data=None, hidden_layers=None, epochs=500, learn_rate=0.0, lr_decay=0.0, l1_reg=0.0, l2_reg=0.0, momentum=0.9, 
                 activation="relu", dropout=0.0, standardize=True, ties="cox", device=None, validation_frequency=10, patience=2000, 
                 improvement_threshold=0.99999, patience_increase=2, logger=None, verbose=True, random_state=None):
          
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

        self.random_state = random_state

        # Network (will be initialized in train())
        self.network = None

        # Optimizer (will be initialized in train())
        self.optimizer = None
    
    def _set_seeds(self):

        """
        Initialise random seeds for reproducibility.
        """

        if self.random_state is not None:
            seed = self.random_state
            
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

        M = risk.max().detach()
        risk = risk - M

        hazard_ratio = torch.exp(risk)

        if self.ties == "cox":
            log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0)) + M
        elif self.ties == "breslow":
            mask = t.view(-1, 1) <= t.view(1, -1)
            mask = mask.float()
            log_risk = torch.log(torch.sum(mask * hazard_ratio.view(1, -1), dim=1)) + M
        
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

        risk = self.network(x)
        cox_loss = self._negative_log_likelihood(risk, t, e)

        l1_loss = self._compute_l1_loss() if self.l1_reg > 0.0 else 0.0
        l2_loss = self._compute_l2_loss() if self.l2_reg > 0.0 else 0.0
        
        total_loss = cox_loss + (self.l1_reg * l1_loss) + (self.l2_reg * l2_loss)

        return total_loss
    
    def get_concordance_index(self, x, t, e, **kwargs):

        """
        Calculate concordance index (C-index) for model predictions.
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

        return (x - self.offset) / self.scale
    
    def fit(self, X_train, y_train, **kwargs):
        
        """
        Fit the model to the data.
        """
        
        # Set random seeds
        self._set_seeds()

        # Breslow estimator for baseline hazards
        self.breslow = BreslowEstimator()

        # Sort by time
        X_train, y_train = self._sort(X_train, y_train)

        if self.logger is None:
            logger = DeepSurvLogger("DeepSurv")
        
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
        e_train = np.array([evento for evento, _ in y_train], np.bool_)
        t_train = np.array([tiempo for _, tiempo in y_train], np.float64)

        if self.valid_data:
            X_valid = np.array(self.valid_data["x"], np.float64)
            e_valid = np.array(self.valid_data["e"], np.bool_)
            t_valid = np.array(self.valid_data["t"], np.float64)
        
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
            self.optimizer.step()
            
            logger.logValue("loss", loss.item(), epoch)
            
            # Calculate training C-index
            ci_train = self.get_concordance_index(X_train, t_train, e_train)
            logger.logValue("c-index", ci_train, epoch)
            
            # Validation
            patience = self.patience
            if self.valid_data and (epoch % self.validation_frequency == 0):
                self.network.eval()
                with torch.no_grad():
                    validation_loss = self._get_loss(x_valid_tensor, e_valid_tensor, t_valid_tensor)
                    logger.logValue("valid_loss", validation_loss.item(), epoch)
                
                ci_valid = self.get_concordance_index(X_valid, t_valid, e_valid)
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
        self.breslow.fit(self.predict(X_train), e_train, t_train)

        logger.shutdown()
        
        logger.history["best_valid_loss"] = best_validation_loss
        logger.history["best_params"] = best_params
        logger.history["best_params_idx"] = best_params_idx
        
        return logger.history
    
    def predict(self, x):

        """
        Predict risk scores for the given data.
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
    def predict_survival_function(self, X, estimator_name, dataset, seed):

        """ 
        S(x, t) = exp(-H(x, t)).
        """

        risk = self.predict(X)

        survival_function = self.breslow.get_survival_function(risk)
        self._plot_survival_hazard_functions(survival_function, estimator_name, dataset, seed, "Survival")

        return survival_function

    def predict_cumulative_hazard_function(self, X, estimator_name, dataset, seed):
        
        """
        H(x,t) = H₀(t) × exp(βᵀx).
        """

        risk = self.predict(X)
        
        get_cumulative_hazard_function = self.breslow.get_cumulative_hazard_function(risk)
        self._plot_survival_hazard_functions(get_cumulative_hazard_function, estimator_name, dataset, seed, "CumulativeRisk")
        
        return get_cumulative_hazard_function
    
    # ----------------------
    # XAI
    # ----------------------
    def calculate_xai(self, X, estimator_name, dataset, seed, feature_names, background=False):

        """
        Calculate XAI values.
        """

        logging.getLogger("xai").setLevel(logging.WARNING)

        # Applying Explainer (model type)
        explainer_risk = shap.Explainer(self.predict, X, feature_names=feature_names, seed=seed)
        
        # Background (faster)
        X_background = X.copy()
        if background:
            X_background = pd.DataFrame(shap.kmeans(X, background).data, columns=feature_names)

        self.shap_explainer = explainer_risk(X_background)

        BaseSurvival.plot_shap(self.shap_explainer, estimator_name, dataset, seed)