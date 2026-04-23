import numpy as np

from MATRIX.models import BaseSurvival
from MATRIX.utils import get_metrics

from collections import defaultdict
from remayn.report import create_excel_summary_report
from remayn.result_set import ResultFolder
from types import SimpleNamespace

def _filter_search(result, estimator_name, dataset, seed):

    """
    Filter function to find the result with the given estimator name, dataset and seed.
    """

    if estimator_name is not None and result.config.get('estimator_name') != estimator_name:
        return False

    if dataset is not None and result.config.get('dataset') != dataset:
        return False

    if seed is not None and result.config.get('random_state') != seed:
        return False

    return True

def _sort_results(results, estimator_name, dataset, seed):
    
    """
    Sort the results by the given estimator name, dataset and seed.
    """

    sort_fields = []
    if estimator_name is None:
        sort_fields.append('estimator_name')
    if dataset is None:
        sort_fields.append('dataset')
    if seed is None:
        sort_fields.append('random_state')

    if sort_fields:
        results = sorted(
            results,
            key=lambda result: tuple(result.config.get(field) for field in sort_fields)
        )

    return results

def get_results(estimator_name=None, dataset=None, seed=None):

    """
    Get the results for the given estimator name, dataset and seed.
    """

    rf = ResultFolder('./results')
    
    filtered_results = rf.filter(lambda result: _filter_search(result, estimator_name, dataset, seed))
    filtered_results = _sort_results(filtered_results, estimator_name, dataset, seed)

    results = []
    for result in filtered_results:
        result.get_data()
        results.append(result)

    return results

def _sort_dict(data_dict):

    """
    Rearrange the matrices in data_list and values_list so that their columns match the order of the first element in feature_names.
    """

    for identifier_name, data in data_dict.items():
        # Use the order of the first element in feature_names as a reference
        reference_order = list(data["feature_names"][0])
        
        _values_list = []
        _data_list = []
        
        # Iterate over values_list and its corresponding feautures
        for values_item, current_features in zip(data["values_list"], data["feature_names"]):
            
            # Search for each reference column within the current features
            current_features_list = list(current_features)
            _order = [current_features_list.index(fn) for fn in reference_order]
            
            # Implement the new order
            if values_item.ndim == 1:
                _values_list.append(values_item[_order])
            else:
                _values_list.append(values_item[:, _order])
        
        # If data_list exists, reorder it as well
        if "data_list" in data:
            for data_item, current_features in zip(data["data_list"], data["feature_names"]):
                current_features_list = list(current_features)
                _order = [current_features_list.index(fn) for fn in reference_order]
                _data_list.append(data_item[:, _order])
            data["data_list"] = _data_list
        
        # Save sorted data
        data["values_list"] = _values_list
        data["feature_names"] = reference_order

    return data_dict

def get_xai(estimator_name=None, dataset=None, seed=None):

    """
    Get the xai for the given estimator name, dataset and seed.
    """

    rf = ResultFolder('./results')
    
    filtered_results = rf.filter(lambda result: _filter_search(result, estimator_name, dataset, seed))

    dictionary_coefficients = defaultdict(lambda: {'values_list': [], 'feature_names': []})
    dictionary_shap = defaultdict(lambda: {'data_list': [], 'values_list': [], 'feature_names': []})
    for result in filtered_results:
        result.get_data()

        # Create an identifier name based on the estimator name and dataset
        identifier_name = f"{result.config['estimator_name']}_{result.config['dataset']}"

        # Accumulate data in the relevant dictionary (coefficients)
        if hasattr(result.data_.best_model, "coefficients"):
            dictionary_coefficients[identifier_name]['values_list'].append(np.array(list(result.data_.best_model.coefficients.values())))
            dictionary_coefficients[identifier_name]['feature_names'].append(list(result.data_.best_model.coefficients.keys()))
        
        # Store data in the relevant dictionary (shap)
        if hasattr(result.data_.best_model, "shap_explainer"):
            dictionary_shap[identifier_name]['data_list'].append(result.data_.best_model.shap_explainer.data)
            dictionary_shap[identifier_name]['values_list'].append(result.data_.best_model.shap_explainer.values)
            dictionary_shap[identifier_name]['feature_names'].append(result.data_.best_model.shap_explainer.feature_names)

    if dictionary_coefficients == {}:
        dictionary_coefficients = None
    else:
        dictionary_coefficients = _sort_dict(dictionary_coefficients)

    if dictionary_shap == {}:
        dictionary_shap = None
    else:
        dictionary_shap = _sort_dict(dictionary_shap)

    # Calculate average coefficients by dataset_estimator
    if dictionary_coefficients is not None:
        average_coefficients = {}
        for identifier_name, data in dictionary_coefficients.items():
            mean_coefficients = np.mean(data['values_list'], axis=0)
            average_coefficients[identifier_name] = dict(zip(data['feature_names'], mean_coefficients))

        # Draw coefficients values means of all seeds by dataset_estimator
        for identifier_name, coefficients in average_coefficients.items():
            dataset_name, estimator_name = identifier_name.split('_')
            BaseSurvival.plot_coefficients(coefficients, estimator_name, dataset_name, seed)

    # Create separate shap_explainer objects for each dataset_estimator
    if dictionary_shap is not None:
        shap_explainers = {}
        for identifier_name, data in dictionary_shap.items():
            shap_explainers[identifier_name] = SimpleNamespace(data=np.vstack(data['data_list']), values=np.vstack(data['values_list']), feature_names=data['feature_names'])

        # Draw shap values of all seeds by dataset_estimator
        for identifier_name, shap_explainer in shap_explainers.items():
            dataset_name, estimator_name = identifier_name.split('_')
            BaseSurvival.plot_shap(shap_explainer, estimator_name, dataset_name, seed)

def save_results(estimator_name=None, dataset=None, seed=None):

    """
    Save the results for the given estimator name, dataset and seed.
    """

    rf = ResultFolder('./results')
    
    filtered_results = rf.filter(lambda result: _filter_search(result, estimator_name, dataset, seed))

    # Define the columns from the config that we want to include in the dataframe
    config_colums = [
        "dataset",
        "estimator_name",
        "random_state"
    ]

    df = filtered_results.create_dataframe(
        config_columns=config_colums,
        metrics_fn=get_metrics,
        include_train=True,
        include_val=False,
        config_columns_prefix=""
    )

    # Columns that will be used to group the results and compute means
    groups_columns = ["dataset", "estimator_name"]

    create_excel_summary_report(df, 'report.xlsx', group_columns=groups_columns)