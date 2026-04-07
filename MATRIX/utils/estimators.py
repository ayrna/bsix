import numpy as np

from .survival_metrics import concordanceIndex
from sklearn.metrics import make_scorer


CLASSIFIERS = [
    "CoxRegression",
    "CoxRegressionWithTimeVarying",
    "RandomSurvForest",
    "DeepSurvFFNN",
]

def get_estimator(estimator_name, inputs, valid_data, random_state, n_jobs=-1, n_iter=30, **kwargs):

    """
    Get estimator (search) based on name.
    """

    if estimator_name in CLASSIFIERS:
        from sklearn.model_selection import RandomizedSearchCV

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

        elif estimator_name == "CoxRegressionWithTimeVarying":
            from ..models import CoxRegressionWithTimeVarying

            param_grid = [
                {
                    "penalizer": np.round(np.logspace(-1, 1, 3), 8),
                    "l1_ratio": np.round(np.linspace(0, 1, 5), 8),
                }
            ]

            estimator = CoxRegressionWithTimeVarying()

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
                scoring=make_scorer(concordanceIndex, greater_is_better=True),
                error_score="raise",
                random_state=random_state,
                verbose=10,
                **kwargs,
            )
        else:
            return estimator

    else:
        raise ValueError(f"Estimator {estimator_name} not recognised.")

