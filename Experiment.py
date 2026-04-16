import argparse
import json
import numpy as np
import pandas as pd
import time

from MATRIX.models import BaseSurvival
from MATRIX.utils import get_data, get_estimator, get_metrics

from remayn.result import make_result
from remayn.result_set import ResultFolder

ESTIMATOR_TO_BLOCK = {
    "CoxRegression": "standard",
    "RandomSurvForest": "standard",
    "DeepSurvFFNN": "standard",
    "DeepMultiTaskFFNN": "multitask",
    "CoxRegressionWithTimeVarying": "time_varying",
    "DeepTimeVaryingFFNN": "time_varying",
}

def _set_global_seed(random_state):

    """
    Set global seeds and deterministic flags like benchmark runner.
    """

    import os
    import random
    import torch

    os.environ["PYTHONHASHSEED"] = str(random_state)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(random_state)
    np.random.seed(random_state)

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_config(estimator, estimator_name, dataset, rs):
    config = estimator.get_params().copy()
    config["estimator_name"] = estimator_name
    config["rs"] = rs
    config["dataset"] = dataset

    keys_to_remove = [
        "cv",
        "estimator",
        "estimator__device",
        "estimator__n_jobs",
        "estimator__val_dataset",
        "estimator__verbose",
        "n_jobs",
        "pre_dispatch",
        "verbose",
        "scoring",
    ]

    for key in keys_to_remove:
        if key in config:
            del config[key]

    return config

def _build_time_varying_dataframe(data_dir, dataset_name, random_state):
	
	"""
	Create the time-varying dataframe following Tutorial.ipynb steps.
	"""

	df = pd.read_csv(f"./{data_dir}/{dataset_name}")
	splits = BaseSurvival.dinamic_discretise(
		y=df[["time", "event"]],
		dataset=dataset_name.replace(".csv", ""),
		random_state=random_state,
	)

	df["identifier"] = df.index.values
	df = BaseSurvival.to_time_dependent(
		dataframe=df,
		splits=splits,
		identifier="identifier",
		time="time",
		event="event",
	)
	
	df = BaseSurvival.to_time_varying(
		dataframe=df,
		identifier="identifier",
		time="time",
		event="event",
	)
	
	return df

def _load_block_data(block_name, data_dir, dataset_name, test_size, validation_size, random_state):
	
    """
    Load data for one model block: standard, multitask or time-varying.
    """

    if block_name == "standard":
        return get_data(
            data_dir=data_dir,
            dataset_name=dataset_name,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
        )

    if block_name == "multitask":
        return get_data(
            data_dir=data_dir,
            dataset_name=dataset_name,
            test_size=test_size,
            validation_size=validation_size,
            to_multitask=True,
            random_state=random_state,
        )

    if block_name == "time_varying":
        df_time_varying = _build_time_varying_dataframe(data_dir, dataset_name, random_state)
        return get_data(
            df=df_time_varying,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
        )

    raise ValueError(f"Unknown block '{block_name}'")

def _get_validation(X_train, y_train, X_validation, y_validation):
	
    """
    Return validation split.
    """

    from sklearn.model_selection import PredefinedSplit

    X_train_val = np.concatenate([X_train, X_validation])
    y_train_val = np.concatenate([y_train, y_validation])

    validation_fold = np.concatenate(
        [
            -1 * np.ones(len(X_train), dtype=int),
            np.zeros(len(X_validation), dtype=int),
        ]
    )
    validation_split = PredefinedSplit(validation_fold)

    return X_train_val, y_train_val, validation_split

def load_and_run_experiment(
    data_dir,
    results_dir,
    dataset,
    test_size=0.2,
    validation_size=0.2,
    estimator_name="CoxRegression",
    random_state=0,
    n_jobs=-1,
    interactive=False, n_iter=30
):

    _set_global_seed(random_state)

    block_name = ESTIMATOR_TO_BLOCK[estimator_name]
    print(f"\n=== Running block: {block_name} ({dataset}, seed={random_state}) ===")

    X_train, y_train, X_validation, y_validation, X_test, y_test, feature_names, scaler = _load_block_data(
        block_name=block_name,
        data_dir=data_dir,
        dataset_name=dataset,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )
    
    print(f"\n-> Estimator: {estimator_name}")

    X_train_val, y_train_val, validation_split = _get_validation(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation
    )

    estimator = get_estimator(
        estimator_name=estimator_name,
        inputs=X_train_val,
        labels=y_train_val,
        valid_data=validation_split,
        random_state=random_state,
        n_jobs=n_jobs,
        n_iter=n_iter,
    )

    if not interactive:
        config = _get_config(estimator, estimator_name, dataset, random_state)
        results = ResultFolder(results_dir)

        if config in results:
            print("Experiment already run. Skipping...")
            return
    
    start = time.time()
    estimator.fit(X_train_val, y_train_val)
    total_time = time.time() - start

    estimator.best_estimator_.predict_survival_function(X_train_val, estimator_name, dataset, random_state)
    estimator.best_estimator_.predict_cumulative_hazard_function(X_train_val, estimator_name, dataset, random_state)
    estimator.best_estimator_.calculate_xai(X_train_val, estimator_name, dataset, random_state, feature_names, background=None)

    if y_train_val.ndim == 1:
        train_survival_pred = estimator.predict(X_train_val)
        test_survival_pred = estimator.predict(X_test)
        train_metrics = get_metrics([y_train_val, y_train_val], [train_survival_pred])
        test_metrics = get_metrics([y_train_val, y_test], [test_survival_pred])
    else:
        train_survival_pred, train_binary_pred = estimator.predict(X_train_val)
        test_survival_pred, test_binary_pred = estimator.predict(X_test)
        train_metrics = get_metrics([y_train_val, y_train_val], [train_survival_pred, train_binary_pred])
        test_metrics = get_metrics([y_train_val, y_test], [test_survival_pred, test_binary_pred])

    config = _get_config(estimator, estimator_name, dataset, random_state)
    best_params = estimator.best_params_ if hasattr(estimator, "best_params_") else {}
    
    if not interactive:
        result = make_result(
            base_path=results_dir,
            config=config,
            predictions=np.array([test_survival_pred], dtype=object) if y_train_val.ndim == 1 else np.array([test_survival_pred, test_binary_pred], dtype=object),
            targets=np.array([y_train_val, y_test], dtype=object),
            train_predictions=np.array([train_survival_pred], dtype=object) if y_train_val.ndim == 1 else np.array([train_survival_pred, train_binary_pred], dtype=object),
            train_targets=np.array([y_train_val, y_train_val]),
            time=total_time,
            best_params=best_params,
            best_model=estimator.best_estimator_
        )
        result.save()
    else:
        print("Train Metrics")
        print(json.dumps(train_metrics, indent=4))
        print(f"Test Metrics")
        print(json.dumps(test_metrics, indent=4))

        if hasattr(estimator, "best_params_"):
            print("Best Params")
            print(json.dumps(estimator.best_params_, indent=4))

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Runner unico MATRIX")
    parser.add_argument("--data_dir", default="MATRIX/datasets", help="Ruta de datasets")
    parser.add_argument("--results_dir", default="./results", help="Directorio de salida")
    parser.add_argument("--test_size", type=float, default=0.2, help="Tamano de test")
    parser.add_argument("--validation_size", type=float, default=0.2, help="Tamano de validacion")
    parser.add_argument("--dataset", default="colon.csv", help="Dataset para corrida unica")
    parser.add_argument("--estimator_name", default="CoxRegression", help="Estimador para corrida unica")
    parser.add_argument("--seed", type=int, default=0, help="Semilla para corrida unica")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Procesos paralelos")
    parser.add_argument("--n_iter", type=int, default=30, help="Iteraciones RandomizedSearchCV")

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    print("Running experiment with config:")
    print(json.dumps(vars(args), indent=4))

    load_and_run_experiment(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        dataset=args.dataset,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.seed,
        estimator_name=args.estimator_name,
        n_jobs=args.n_jobs,
        interactive=True,
        n_iter=args.n_iter,
    )


if __name__ == "__main__":
    main()