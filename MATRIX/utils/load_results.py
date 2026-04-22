from remayn.result_set import ResultFolder

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