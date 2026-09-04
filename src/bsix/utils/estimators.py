import numpy as np

from .survival_metrics import scorerConcordanceIndex

from scipy import stats
from sklearn.metrics import make_scorer

CLASSIFIERS = [
    "BaseCoxRegression",
    "BaseCoxRegressionWithTimeVarying",
    "BaseDeepHit",
    "BaseSurvivalTree",
    "BaseRandomSurvivalForest",

    "AcceleratedFailureTime",

    "CoxRegression",
    "DeepHit",
    "DeepSurv",
    "RandomSurvForest",
    "SurvTree",
    "SurvivalTabPFN",

    "CoxRegressionWithTimeVarying",
    "DeepTimeVarying",
    
    "DeepMultiTask",
]

NETS = [
    "BaseDeepHit",
    "DeepHit",
    "DeepMultiTask",
    "DeepSurv",
    "DeepTimeVarying",
]

class SerializableUniform:
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale
        self._dist = stats.uniform(loc=loc, scale=scale)

    def rvs(self, random_state=None, size=None):
        return self._dist.rvs(random_state=random_state, size=size)

    def __str__(self):
        return f"uniform(loc={self.loc}, scale={self.scale})"
    
    def __repr__(self):
        return self.__str__()

class SerializableLogUniform:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self._dist = stats.loguniform(a=a, b=b)

    def rvs(self, random_state=None, size=None):
        return self._dist.rvs(random_state=random_state, size=size)

    def __str__(self):
        return f"loguniform(a={self.a}, b={self.b})"
    
    def __repr__(self):
        return self.__str__()
    
def get_estimator(estimator_name, inputs, labels, valid_data, seed, config=False, n_jobs=-1, n_iter=30):

    """
    Get estimator (search cv) based on name.
    """

    if estimator_name in CLASSIFIERS:
        from sklearn.experimental import enable_halving_search_cv
        from sklearn.model_selection import RandomizedSearchCV, HalvingRandomSearchCV

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        if estimator_name == "AcceleratedFailureTime":
            from ..models import AcceleratedFailureTime

            param_grid = [
                {
                    "type": ["LogLogisticAFT", "WeibullAFT"],
                    "penalizer": np.round(np.logspace(-1, 1, 3), 8),
                    "l1_ratio": np.round(np.linspace(0, 1, 5, endpoint=False), 8),
                }
            ]

            estimator = AcceleratedFailureTime()
        
        elif estimator_name == "CoxRegression":
            from ..models import CoxRegression

            param_grid = [
                {
                    "alpha": np.round(np.logspace(-3, -1, 3), 8),
                    "ties": ["efron", "breslow"],
                    "n_iter": [100, 200, 300, 400, 500],
                }
            ]

            estimator = CoxRegression()

        elif estimator_name == "DeepHit":
            from ..models import DeepHit
            
            rng = np.random.default_rng(seed=seed)
            param_grid = [
                {   
                    #"epochs": [250, 500],
                    "hidden_layers_shared": [[8], [16], [32], [16, 16], [32, 32]],
                    "hidden_layers_specific": [[8], [16], [32], [16, 16], [32, 32]],
                    "learn_rate": SerializableLogUniform(1e-5, 1e-1),
                    "lr_decay": SerializableLogUniform(1e-6, 1e-3),
                    "l1_reg_output": SerializableLogUniform(1e-5, 1e-1),
                    "l2_reg_hidden": SerializableLogUniform(1e-5, 1e-1),
                    "dropout": SerializableUniform(loc=0.0, scale=1.0),
                    "activation": ["relu", "selu", "tanh", "sigmoid"],
                    "alpha": SerializableUniform(loc=0.0, scale=1.0),
                    "beta": SerializableUniform(loc=0.0, scale=1.0),
                }
            ]

            estimator = DeepHit(inputs.shape[1], len(np.unique(labels["event"])) - 1, 100, time_threshold=(100 - 20), seed=seed)

        elif estimator_name == "DeepSurv":
            from ..models import DeepSurv
               
            param_grid = [
                {
                    #"epochs": [250, 500],
                    "hidden_layers": [[8], [16], [32], [16, 16], [32, 32]],
                    "learn_rate": SerializableLogUniform(1e-5, 1e-1),
                    "lr_decay": SerializableLogUniform(1e-6, 1e-3),
                    "l1_reg": SerializableLogUniform(1e-5, 1e-1),
                    "l2_reg": SerializableLogUniform(1e-5, 1e-1),
                    "dropout": SerializableUniform(loc=0.0, scale=1.0),
                    "activation": ["relu", "selu", "tanh", "sigmoid"],
                }
            ]

            estimator = DeepSurv(inputs.shape[1], seed=seed)

        elif estimator_name == "SurvivalTabPFN":
            from ..models import SurvivalTabPFN
               
            param_grid = {
                "n_estimators":[2, 4, 8],
            }

            estimator = SurvivalTabPFN(seed=seed)

        elif estimator_name == "RandomSurvForest":
            from ..models import RandomSurvForest
               
            param_grid = [
                {
                    "n_estimators": [100, 300, 500],
                    "max_depth": [3, 5, 7],
                    "min_samples_leaf": [2, 3, 5],
                    "min_samples_split": [2, 6, 10],
                }
            ]
            
            estimator = RandomSurvForest(seed=seed)

        elif estimator_name == "SurvTree":
            from ..models import SurvTree

            param_grid = [
                {
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 6, 10],
                    "min_samples_leaf": [2, 3, 5],
                }
            ]

            estimator = SurvTree(seed=seed)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        elif estimator_name == "CoxRegressionWithTimeVarying":
            from ..models import CoxRegressionWithTimeVarying

            param_grid = [
                {
                    "alpha": np.round(np.logspace(-1, 1, 3), 8),
                    "ties": ["efron", "breslow"],
                    "n_iter": [100, 200, 300],
                }
            ]

            estimator = CoxRegressionWithTimeVarying()

        elif estimator_name == "DeepTimeVarying":
            from ..models import DeepTimeVarying
               
            param_grid = [
                {
                    #"epochs": [250, 500],
                    "hidden_layers": [[8], [16], [32], [16, 16], [32, 32]],
                    "learn_rate": SerializableLogUniform(1e-5, 1e-1),
                    "lr_decay": SerializableLogUniform(1e-6, 1e-3),
                    "l1_reg": SerializableLogUniform(1e-5, 1e-1),
                    "l2_reg": SerializableLogUniform(1e-5, 1e-1),
                    "dropout": SerializableUniform(loc=0.0, scale=1.0),
                    "activation": ["relu", "selu", "tanh", "sigmoid"],
                }
            ]

            estimator = DeepTimeVarying(inputs.shape[1], seed=seed)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        elif estimator_name == "DeepMultiTask":
            from ..models import DeepMultiTask
            
            rng = np.random.default_rng(seed=seed)
            param_grid = [
                {
                    #"epochs": [250, 500],
                    "hidden_layers": [[8], [16], [32], [16, 16], [32, 32]],
                    "learn_rate": SerializableLogUniform(1e-5, 1e-1),
                    "lr_decay": SerializableLogUniform(1e-6, 1e-3),
                    "cox_reg": SerializableLogUniform(1e-1, 1e1),
                    "l1_reg": SerializableLogUniform(1e-5, 1e-1),
                    "l2_reg": SerializableLogUniform(1e-5, 1e-1),
                    "dropout": SerializableUniform(loc=0.0, scale=1.0),
                    "activation": ["relu", "selu", "tanh", "sigmoid"],
                    "coef_likelihood": np.round(rng.dirichlet(alpha=np.ones(1 if labels.ndim == 1 else labels.shape[1]), size=30), 8).tolist(),
                }
            ]

            estimator = DeepMultiTask(inputs.shape[1], seed=seed)

        elif estimator_name == "BaseCoxRegression":
            from ..models import BaseCoxRegression

            param_grid = [
                {
                    "alpha": np.round(np.logspace(-3, -1, 3), 8),
                    "ties": ["efron", "breslow"],
                    "n_iter": [100, 200, 300, 400, 500],
                }
            ]

            estimator = BaseCoxRegression()

        elif estimator_name == "BaseCoxRegressionWithTimeVarying":
            from ..models import BaseCoxRegressionWithTimeVarying

            param_grid = [
                {
                    "penalizer": np.round(np.logspace(-1, 1, 3), 8),
                    "l1_ratio": np.round(np.linspace(0, 1, 5, endpoint=False), 8),
                }
            ]

            estimator = BaseCoxRegressionWithTimeVarying()

        elif estimator_name == "BaseDeepHit":
            from ..models import BaseDeepHit
            
            rng = np.random.default_rng(seed=seed)
            param_grid = [
                {   
                    #"epochs": [50, 100],
                    "num_nodes": [[8], [16], [32], [16, 16], [32, 32]],
                    "learning_rate": SerializableLogUniform(1e-5, 1e-1),
                    "alpha": SerializableUniform(loc=0.0, scale=1.0),
                    "sigma": SerializableUniform(loc=0.0, scale=1.0),
                    "dropout": SerializableUniform(loc=0.0, scale=1.0),
                    "activation": ["relu", "selu", "tanh", "sigmoid"],
                }
            ]

            estimator = BaseDeepHit(100, time_threshold=(100 - 20), seed=seed)

        elif estimator_name == "BaseRandomSurvivalForest":
            from ..models import BaseRandomSurvivalForest

            param_grid = [
                {
                    "n_estimators": [100, 300, 500],
                    "max_depth": [3, 5, 7],
                    "min_samples_leaf": [2, 3, 5],
                    "min_samples_split": [2, 6, 10],
                }
            ]

            estimator = BaseRandomSurvivalForest(seed=seed)
            
        elif estimator_name == "BaseSurvivalTree":
            from ..models import BaseSurvivalTree

            param_grid = [
                {
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 6, 10],
                    "min_samples_leaf": [2, 3, 5],
                }
            ]

            estimator = BaseSurvivalTree(seed=seed)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        else:
            raise NotImplementedError(
                f"Estimator {estimator_name} not implemented in set_estimators function."
            )

        if config:
            return estimator, param_grid
        else:
            if len(param_grid) > 0:
                if estimator_name in NETS:
                    return HalvingRandomSearchCV(
                        estimator=estimator,
                        param_distributions=param_grid,
                        refit=True,
                        return_train_score=True,
                        n_candidates=n_iter,
                        factor=2,
                        resource='epochs',
                        min_resources='exhaust',
                        max_resources=1000,
                        n_jobs=n_jobs,
                        cv=valid_data,
                        scoring=make_scorer(scorerConcordanceIndex, greater_is_better=True),
                        error_score="raise",
                        random_state=seed,
                        verbose=10
                    )
                else:
                    return RandomizedSearchCV(
                        estimator=estimator,
                        param_distributions=param_grid,
                        refit=True,
                        return_train_score=True,
                        n_iter=n_iter,
                        n_jobs=n_jobs,
                        cv=valid_data,
                        scoring=make_scorer(scorerConcordanceIndex, greater_is_better=True),
                        error_score="raise",
                        random_state=seed,
                        verbose=10
                    )
            else:
                return estimator

    else:
        raise ValueError(f"Estimator {estimator_name} not recognised.")