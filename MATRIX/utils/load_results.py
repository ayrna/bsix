import itertools
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
    Rearrange the matrices/arrays in 'data_list' and 'values_list' so that their columns 
    match a unified reference order of 'feature_names' across all items.
    """

    def _align_to_reference(item_list, feature_lists, reference_order):

        """
        Aligns features of each seed/item to the unified reference order.
        Missing features are padded with NaNs.
        """

        aligned_list = []
        for item, current_features in zip(item_list, feature_lists):
            current_features_list = list(current_features)
            
            # Ensure item is at least 1D to safely check its dimensions
            item_array = np.atleast_1d(item)
            is_2d = item_array.ndim >= 2
            
            _list = []
            for fn in reference_order:
                if fn in current_features_list:
                    idx = current_features_list.index(fn)
                    # Extract the column for 2D arrays, or the scalar for 1D arrays
                    _list.append(item_array[:, idx] if is_2d else item_array[idx])
                else:
                    # Pad with NaNs of the correct shape if the feature is missing
                    if is_2d:
                        _list.append(np.full(item_array.shape[0], np.nan))
                    else:
                        _list.append(np.nan)
            
            # Reconstruct the array with the aligned features
            if is_2d:
                aligned_list.append(np.column_stack(_list))
            else:
                aligned_list.append(np.array(_list))
                
        return aligned_list


    def _pad_and_stack_numeric(aligned_list):

        """
        Pads items with missing samples (rows) with NaNs to ensure homogeneous shapes 
        across all seeds, preventing NumPy conversion errors, and stacks them.
        """

        if not aligned_list:
            return np.array([])
            
        is_2d = aligned_list[0].ndim == 2
        
        if is_2d:
            # Find the maximum number of samples (rows) across all seeds
            max_samples = max(item.shape[0] for item in aligned_list)
            
            padded_list = []
            for item in aligned_list:
                missing_rows = max_samples - item.shape[0]
                
                if missing_rows > 0:
                    # Pad with NaNs at the bottom if the current seed has fewer samples
                    padding = np.full((missing_rows, item.shape[1]), np.nan)
                    item = np.vstack((item, padding))
                    
                padded_list.append(item)
            
            # Safely convert to a numpy array (homogeneous shape guaranteed)
            return np.squeeze(np.array(padded_list, dtype=float))
        else:
            # For 1D arrays, no row padding is needed
            return np.squeeze(np.array(aligned_list, dtype=float))


    # Dictionary processing
    for identifier_name, data in data_dict.items():
        
        # Create a unified, duplicate-free list of all features preserving order
        reference_order = list(dict.fromkeys(itertools.chain.from_iterable(data["feature_names"])))
        
        # Align values_list and safely stack into a NumPy array
        _values_aligned = _align_to_reference(data["values_list"], data["feature_names"], reference_order)
        data["values_list"] = _pad_and_stack_numeric(_values_aligned)
        
        # Align data_list if it exists
        if "data_list" in data:
            _data_aligned = _align_to_reference(data["data_list"], data["feature_names"], reference_order)
            data["data_list"] = _pad_and_stack_numeric(_data_aligned)
            
        # Update feature names to the reference order as a safe 1D array
        data["feature_names"] = np.atleast_1d(reference_order)

    return data_dict

def get_xai(estimator_name=None, dataset=None, seed=None, individual=None):

    """
    Get the xai for the given estimator name, dataset and seed.
    """

    rf = ResultFolder('./results')
    
    filtered_results = rf.filter(lambda result: _filter_search(result, estimator_name, dataset, seed))

    dictionary_coefficients = defaultdict(lambda: {'values_list': [], 'feature_names': []})
    dictionary_scaler = defaultdict(lambda: {'scaler': []})
    dictionary_shap = defaultdict(lambda: {'data_list': [], 'values_list': [], 'feature_names': []})
    
    for result in filtered_results:
        result.get_data()

        # Create an identifier name based on the estimator name and dataset
        identifier_name = f"{result.config['estimator_name']}_{result.config['dataset']}"

        # Accumulate data in the relevant dictionary (coefficients)
        if hasattr(result.data_.best_model, "coefficients"):
            dictionary_coefficients[identifier_name]['values_list'].append(list(result.data_.best_model.coefficients.values()))
            dictionary_coefficients[identifier_name]['feature_names'].append(list(result.data_.best_model.coefficients.keys()))

        # Store data in the relevant dictionary (shap)
        if hasattr(result.data_.best_model, "shap_explainer"):
            dictionary_shap[identifier_name]['data_list'].append(result.data_.best_model.shap_explainer.data)
            dictionary_shap[identifier_name]['values_list'].append(result.data_.best_model.shap_explainer.values)
            dictionary_shap[identifier_name]['feature_names'].append(result.data_.best_model.shap_explainer.feature_names)

            dictionary_shap[identifier_name]['data_list'] = list(dictionary_shap[identifier_name]['data_list'])
            dictionary_shap[identifier_name]['values_list'] = list(dictionary_shap[identifier_name]['values_list'])

        # Store scaler in the relevant dictionary (scaler)
        if hasattr(result.data_.best_model, "scaler_"):
            dictionary_scaler[identifier_name]['scaler'].append(result.data_.best_model.scaler_)

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
            if (data['values_list']).ndim > 1:
                mean_coefficients = np.nanmean(data['values_list'], axis=1)
            else:
                mean_coefficients = data['values_list']

            # Ensure at keast 1D in arrays
            data['feature_names'] = np.atleast_1d(data['feature_names'])
            mean_coefficients = np.atleast_1d(mean_coefficients)

            average_coefficients[identifier_name] = dict(zip(data['feature_names'], mean_coefficients))

        # Draw coefficients values means of all seeds by dataset_estimator
        for identifier_name, coefficients in average_coefficients.items():
            estimator_name, dataset_name = identifier_name.split('_')
            BaseSurvival.plot_coefficients(coefficients, estimator_name, dataset_name, seed)

    # Create separate shap_explainer objects for each dataset_estimator
    if dictionary_shap is not None:
        shap_explainers = {}
        for identifier_name, data in dictionary_shap.items():
            shap_explainers[identifier_name] = SimpleNamespace(data=data['data_list'], values=data['values_list'], feature_names=data['feature_names'])

        # Draw shap values of all seeds by dataset_estimator
        for identifier_name, shap_explainer in shap_explainers.items():
            estimator_name, dataset_name = identifier_name.split('_')
            BaseSurvival.plot_shap(shap_explainer, estimator_name, dataset_name, seed)
            if individual is not None:
                BaseSurvival.plot_individual_shap(shap_explainer, individual, dictionary_scaler[identifier_name]['scaler'][0], estimator_name, dataset_name, seed)

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