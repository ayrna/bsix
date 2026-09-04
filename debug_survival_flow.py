import os


def main():
    import json

    from execution.flows import run_survival_flow

    config = dict(
        data_dir="bsix.datasets",
        results_dir="./results",
        dataset="colon.csv",
        test_size=0.25,
        validation_size=0.2,
        seed=0,
        estimator_name="CoxRegression",
        n_iter=1,
        n_jobs=1,
        interactive=False,
    )

    print(f"Running experiment with config: ")
    print(json.dumps(config, indent=4))

    run_survival_flow(**config)

    import config as CONFIG
    from execution.configs import load_config
    from execution.results import collect_results

    CONFIG = load_config(config_path="./execution/config/default.py")

    collect_results(
            CONFIG.results_dir,
            output_path=CONFIG.prepared_results_dir,
            appendix=CONFIG.prepared_results_appendix,
            methods=CONFIG.collect_methods,
            datasets=CONFIG.collect_datasets,
            seeds=CONFIG.collect_seeds,
            config_columns_to_include=CONFIG.config_columns_to_include,
            best_params_columns_to_include=CONFIG.best_params_columns_to_include,
            include_train=CONFIG.collect_train,
            include_val=CONFIG.collect_val,
            skip_zip=CONFIG.skip_zip,
            n_jobs=1,
            external_cv=False,
        )

if __name__ == "__main__":
    main()
