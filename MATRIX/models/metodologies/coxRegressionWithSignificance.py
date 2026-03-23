import logging
import numpy as np
import pandas as pd
import shap
import warnings

from ..base import BaseSurvival
from lifelines import CoxPHFitter
from sksurv.linear_model.coxph import BreslowEstimator

warnings.filterwarnings("ignore")

class CoxRegressionWithSignificance(BaseSurvival):
    def __init__(self, inputs, labels, penalizer=0.0, l1_ratio=0.0, formula=None):
        self.inputs = inputs
        self.labels = labels

        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.formula = formula
        self.model = None

        self.labels_covariables = ["event", "time"]

    def _significance_covariables(self):
        _model = CoxPHFitter(penalizer=0.01)

        dataframe = pd.concat([self._toDataframe(self.inputs), self._toDataframe(self.labels, self.labels_covariables)], axis=1)
        significance_covariables = [c for c in dataframe.columns.to_list() if c not in self.labels_covariables]
        
        _model.fit(dataframe[significance_covariables + self.labels_covariables], duration_col=self.labels_covariables[1], event_col=self.labels_covariables[0], show_progress=False)
        significance = _model._compute_p_values()
        
        while max(significance) >= 0.05 and len(significance_covariables) > 0:
            covariables_to_delete = significance_covariables[np.where(significance == max(significance))[0][0]]
            significance_covariables.remove(covariables_to_delete)
            
            _model.fit(dataframe[significance_covariables + self.labels_covariables], duration_col=self.labels_covariables[1], event_col=self.labels_covariables[0], show_progress=False)
            significance = _model._compute_p_values()
            
        self.significance_covariables = significance_covariables
        
        return self

    def _toDataframe(self, data, columns=None):
        if columns == None:
            dataframe = pd.DataFrame(data, columns=[str(l) for l in range(data.shape[1])])
        else:
            dataframe = pd.DataFrame(data, columns=columns)

        return dataframe

    def fit(self, X, y):
        # Breslow estimator for baseline hazards
        self.breslow = BreslowEstimator()

        # Sort by time
        X, y = self._sort(X, y)
                          
        self._significance_covariables()

        dataframe = pd.concat([self._toDataframe(X), self._toDataframe(y, self.labels_covariables)], axis=1)

        self.model = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.model.fit(dataframe[self.significance_covariables + self.labels_covariables], duration_col=self.labels_covariables[1], event_col=self.labels_covariables[0], show_progress=False)
        
        # Compute baseline hazards with training data
        self.breslow.fit(self.predict(X), y["event"], y["time"])

        self.coef_ = self.model.params_
        
        return self

    def predict(self, X):
        risk = self.model.predict_log_partial_hazard(self._toDataframe(X)[self.significance_covariables]).to_numpy()

        return risk
    
    def score(self, X, y):
        
        return None
    
    # ----------------------
    # Base Survival methods
    # ----------------------
    def predict_survival_function(self, X, estimator_name, dataset, seed):

        """ 
        S(x, t) = exp(-H(x, t)) 
        """

        risk = self.predict(X)
        
        survival_function = self.breslow.get_survival_function(risk)
        self._plot_survival_hazard_functions(survival_function, estimator_name, dataset, seed, "Survival")

        return survival_function

    def predict_cumulative_hazard_function(self, X, estimator_name, dataset, seed):
        
        """
        H(x,t) = H₀(t) × exp(βᵀx)
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
        Calculate XAI values
        """

        logging.getLogger('xai').setLevel(logging.WARNING)

        # Applying Explainer (model type)
        explainer_risk = shap.Explainer(self.predict, X, feature_names=feature_names, seed=seed)
        
        # Background (faster)
        X_background = X.copy()
        if background:
            X_background = pd.DataFrame(shap.kmeans(X, background).data, columns=feature_names)

        shap_explainer = explainer_risk(X_background)

        self._plot_xai(shap_explainer, estimator_name, dataset, seed)
