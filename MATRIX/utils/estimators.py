import numpy as np

from .survival_metrics import scorerConcordanceIndex
from sklearn.metrics import make_scorer

CLASSIFIERS = [
    "CoxRegression",
    "CoxRegressionWithTimeVarying",
    "DeepMultiTaskFFNN",
    "DeepSurvFFNN",
    "DeepTimeVaryingFFNN",
    "RandomSurvForest",
]

MULTITASKCLASSIFIERS = [
    "DeepMultiTaskFFNN",
]

TIMEVARYINGCLASSIFIERS = [
    "CoxRegressionWithTimeVarying",
    "DeepTimeVaryingFFNN",
]

def get_estimator(estimator_name, inputs, labels, valid_data, random_state, n_jobs=-1, n_iter=30):

    """
    Get estimator (search) based on name.
    """

    if estimator_name in CLASSIFIERS:
        from sklearn.model_selection import RandomizedSearchCV

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        if estimator_name == "CoxRegression":
            from ..models import CoxRegression

            param_grid = [
                {
                    "alpha": np.round(np.logspace(-1, 1, 3), 8),
                    "ties": ["efron", "breslow"],
                    "n_iter": [100, 200, 300],
                }
            ]

            estimator = CoxRegression()

        elif estimator_name == "RandomSurvForest":
            from ..models import RandomSurvForest
               
            param_grid = [
                {
                    "n_estimators": [100, 200, 300, 400, 500],
                    "max_depth": [3, 7, 15, None],
                    "min_samples_split": [2, 6, 10],
                }
            ]
            
            estimator = RandomSurvForest(random_state=random_state)

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
                    "activation": ["relu", "selu", "tanh"],
                }
            ]

            estimator = DeepSurv(inputs.shape[1], random_state=random_state)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        elif estimator_name == "CoxRegressionWithTimeVarying":
            from ..models import CoxRegressionWithTimeVarying

            param_grid = [
                {
                    "penalizer": np.round(np.logspace(-1, 1, 3), 8),
                    "l1_ratio": np.round(np.linspace(0, 1, 5), 8),
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
                    "activation": ["relu", "selu", "tanh"],
                }
            ]

            estimator = DeepTimeVarying(inputs.shape[1], random_state=random_state)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        elif estimator_name == "DeepMultiTaskFFNN":
            from ..models import DeepMultiTask
            
            rng = np.random.default_rng(seed=random_state)
            param_grid = [
                {
                    "epochs": [250, 500],
                    "hidden_layers": [[4], [8], [16], [8, 8], [16, 16], [8, 8, 8]],
                    "learn_rate": np.round(np.logspace(-7, -1, 7), 8),
                    "lr_decay": np.round(np.logspace(-7, -1, 7), 8),
                    "l1_reg": np.round(np.logspace(-3, 3, 7), 8),
                    "l2_reg": np.round(np.logspace(-3, 3, 7), 8),
                    "cox_reg": np.round(np.logspace(-3, 3, 7), 8),
                    "bin_reg": np.round(np.logspace(-3, 3, 7), 8),
                    "dropout": np.round(np.linspace(0, 1, 5), 8),
                    "activation": ["relu", "selu", "tanh"],
                    "coef_likelihood": np.round(rng.dirichlet(alpha=np.ones(1 if labels.ndim == 1 else labels.shape[1]), size=7), 8).tolist(),
                    "coef_binary": np.round(rng.dirichlet(alpha=np.ones(1 if labels.ndim == 1 else labels.shape[1]), size=7), 8).tolist(),
                    #"momentum": np.round(np.linspace(0.4, 0.9, 6), 5),
                    #"ties": ["cox", "breslow"],
                    #"batch_size": [128, 256, 512],
                }
            ]

            estimator = DeepMultiTask(inputs.shape[1], random_state=random_state)

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
                random_state=random_state,
                verbose=10
            )
        else:
            return estimator

    else:
        raise ValueError(f"Estimator {estimator_name} not recognised.")