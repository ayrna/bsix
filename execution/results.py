from bsix.utils import from_results_to_metrics
from typing import Optional, List, Union
from pathlib import Path
from remayn.result_set import ResultFolder
from datetime import datetime


def collect_results(
    results_path: Union[Path, str],
    output_path: Union[Path, str] = "prepared_results",
    appendix: str = "",
    datasets: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    config_columns_to_include: Optional[List[str]] = None,
    best_params_columns_to_include: Optional[List[str]] = None,
    include_train: bool = True,
    include_val: bool = False,
    skip_zip: bool = False,
    n_jobs: int = 1,
    external_cv: bool = True,
):

    if config_columns_to_include is None:
        config_columns_to_include = ["dataset", "estimator_name", "random_state",]

    if best_params_columns_to_include is None:
        best_params_columns_to_include = []

    def filter_fn(result):
        if datasets is not None and result.config["dataset"] not in datasets:
            return False

        if methods is not None and result.config["estimator_name"] not in methods:
            return False

        if seeds is not None and result.config["random_state"] not in seeds:
            return False

        if external_cv:
            # Get only final results
            if (
                "search_n_iter" not in result.config
                or result.config["search_n_iter"] is None
            ):
                if (
                    result.config["n_folds"] is not None
                    and result.config["n_folds"] > 1
                ):
                    return False
                if (
                    result.config["validation_size"] is not None
                    and result.config["validation_size"] > 0
                ):
                    return False

        return True

    results_path = Path(results_path)
    results = ResultFolder(results_path)

    df = results.create_dataframe(
        config_columns=config_columns_to_include,
        best_params_columns=best_params_columns_to_include,
        filter_fn=filter_fn,
        metrics_fn=from_results_to_metrics,
        include_train=include_train,
        include_val=include_val,
        config_columns_prefix="",
        n_jobs=n_jobs,
    )

    print(df)

    df = df.sort_values(by=["dataset", "estimator_name", "random_state"])

    if len(appendix) > 0 and not appendix.startswith("_"):
        appendix = f"_{appendix}"

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_wo_ext = (
        output_path / f'{datetime.now().strftime(r"%Y%m%d_%H%M%S")}{appendix}'
    )

    df.to_excel(f"{output_file_wo_ext}.xlsx", index=False)

    if not skip_zip:
        from shutil import make_archive

        make_archive(str(output_file_wo_ext), "zip", str(results_path))
