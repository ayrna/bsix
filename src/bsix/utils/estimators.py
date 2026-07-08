import numpy as np

from .survival_metrics import scorerConcordanceIndex

from sklearn.metrics import make_scorer

CLASSIFIERS = [
    "BaseCoxRegression",
    "BaseCoxRegressionWithTimeVarying",
    "BaseDeepHit",
    "BaseRandomSurvivalForest",
    "BaseSurvivalTree",

    "AcceleratedFailureTime",
    "CoxRegression",
    "CoxRegressionWithTimeVarying",
    "DeepHitFFNN",
    "DeepMultiTaskFFNN",
    "DeepMultiTaskMultiLossFFNN",
    "DeepSurvFFNN",
    "DeepTimeVaryingFFNN",
    "RandomSurvForest",
    "SurvTree",
]

def get_estimator(estimator_name, inputs, labels, valid_data, seed, n_jobs=-1, n_iter=30):

    """
    Get estimator (search cv) based on name.
    """

    if estimator_name in CLASSIFIERS:
        from sklearn.model_selection import RandomizedSearchCV

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

        elif estimator_name == "DeepSurvFFNN":
            from ..models import DeepSurv
               
            param_grid = [
                {
                    "epochs":[250, 500],
                    "hidden_layers": [[4], [8], [16], [32]],
                    "learn_rate": np.round(np.logspace(-5, -3, 3), 8),
                    "lr_decay": np.round(np.logspace(-8, -6, 3), 8),
                    "l1_reg": np.round(np.logspace(-5, -3, 3), 8),
                    "l2_reg": np.round(np.logspace(-4, -2, 3), 8),
                    "dropout": np.round(np.linspace(0.25, 0.75, 3), 8),
                    "activation": ["relu", "selu", "tanh", "sigmoid"],
                }
            ]

            estimator = DeepSurv(inputs.shape[1], seed=seed)

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

        elif estimator_name == "DeepTimeVaryingFFNN":
            from ..models import DeepTimeVarying
               
            param_grid = [
                {
                    "epochs":[250, 500],
                    "hidden_layers": [[4], [8], [16], [32]],
                    "learn_rate": np.round(np.logspace(-5, -3, 3), 8),
                    "lr_decay": np.round(np.logspace(-8, -6, 3), 8),
                    "l1_reg": np.round(np.logspace(-5, -3, 3), 8),
                    "l2_reg": np.round(np.logspace(-4, -2, 3), 8),
                    "dropout": np.round(np.linspace(0.25, 0.75, 3), 8),
                    "activation": ["relu", "selu", "tanh", "sigmoid"],
                }
            ]

            estimator = DeepTimeVarying(inputs.shape[1], seed=seed)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        elif estimator_name == "DeepHitFFNN":
            from ..models import DeepHit
            
            rng = np.random.default_rng(seed=seed)
            param_grid = [
                {   
                    "epochs": [50, 100],
                    "hidden_layers_shared": [[4], [8], [16], [32]],
                    "hidden_layers_specific": [[4], [8], [16], [32]],
                    "learn_rate": np.round(np.logspace(-5, -3, 3), 8),
                    "lr_decay": np.round(np.logspace(-8, -6, 3), 8),
                    "l1_reg_output": np.round(np.logspace(-5, -3, 3), 8),
                    "l2_reg_hidden": np.round(np.logspace(-4, -2, 3), 8),
                    "dropout": np.round(np.linspace(0.25, 0.75, 3), 8),
                    "activation": ["relu", "selu", "tanh", "sigmoid"],
                    "alpha": np.round(np.linspace(0.1, 0.9, 5), 8),
                    "beta": np.round(np.linspace(0.1, 0.9, 5), 8),
                }
            ]

            estimator = DeepHit(inputs.shape[1], len(np.unique(labels["event"])) - 1, 100, time_threshold=(100 - 50), seed=seed)

        elif estimator_name == "DeepMultiTaskFFNN":
            from ..models import DeepMultiTask
            
            rng = np.random.default_rng(seed=seed)
            param_grid = [
                {
                    "epochs":[250, 500],
                    "hidden_layers": [[4], [8], [16], [32]],
                    "learn_rate": np.round(np.logspace(-5, -3, 3), 8),
                    "lr_decay": np.round(np.logspace(-8, -6, 3), 8),
                    "l1_reg": np.round(np.logspace(-5, -3, 3), 8),
                    "l2_reg": np.round(np.logspace(-4, -2, 3), 8),
                    "cox_reg": np.round(np.logspace(-3, 3, 7), 8),
                    "dropout": np.round(np.linspace(0.25, 0.75, 3), 8),
                    "activation": ["relu", "selu", "tanh"],
                    "coef_likelihood": np.round(rng.dirichlet(alpha=np.ones(1 if labels.ndim == 1 else labels.shape[1]), size=7), 8).tolist(),
                }
            ]

            estimator = DeepMultiTask(inputs.shape[1], seed=seed)

        elif estimator_name == "DeepMultiTaskMultiLossFFNN":
            from ..models import DeepMultiTaskMultiLoss
            
            rng = np.random.default_rng(seed=seed)
            param_grid = [
                {
                    "epochs":[250, 500],
                    "hidden_layers": [[4], [8], [16], [32]],
                    "learn_rate": np.round(np.logspace(-5, -3, 3), 8),
                    "lr_decay": np.round(np.logspace(-8, -6, 3), 8),
                    "l1_reg": np.round(np.logspace(-5, -3, 3), 8),
                    "l2_reg": np.round(np.logspace(-4, -2, 3), 8),
                    "cox_reg": np.round(np.logspace(-3, 3, 7), 8),
                    "bin_reg": np.round(np.logspace(-3, 3, 7), 8),
                    "dropout": np.round(np.linspace(0.25, 0.75, 3), 8),
                    "activation": ["relu", "selu", "tanh"],
                    "coef_likelihood": np.round(rng.dirichlet(alpha=np.ones(1 if labels.ndim == 1 else labels.shape[1]), size=7), 8).tolist(),
                    "coef_binary": np.round(rng.dirichlet(alpha=np.ones(1 if labels.ndim == 1 else labels.shape[1]), size=7), 8).tolist(),
                }
            ]

            estimator = DeepMultiTaskMultiLoss(inputs.shape[1], seed=seed)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

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
                    "epochs": [50, 100],
                    "num_nodes": [[4], [8], [16], [32]],
                    "learning_rate": np.round(np.logspace(-5, -3, 3), 8),
                    "alpha": np.round(np.linspace(0.1, 0.9, 5), 8),
                    "sigma": np.round(np.linspace(0.1, 0.9, 5), 8),
                    "dropout": np.round(np.linspace(0.25, 0.75, 3), 8),
                    "activation": ["relu", "selu", "tanh", "sigmoid"],
                }
            ]

            estimator = BaseDeepHit(100, time_threshold=(100 - 50), seed=seed)

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
        
        if len(param_grid) > 0:
            return RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
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