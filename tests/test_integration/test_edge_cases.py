"""
Edge case and error handling tests (Category ERR).
Tests for robustness and error conditions following BSIX_TEST_SPECIFICATION.md
"""

import pytest
import numpy as np


class TestEmptyAndSingleData:
    """Edge cases for empty or minimal datasets (ERR-001, ERR-010)"""

    @pytest.mark.edge_case
    def test_empty_training_data_accelerated_failure_time(self):
        """ERR-001: Models handle empty training data"""
        from bsix.models import AcceleratedFailureTime
        
        X = np.array([]).reshape(0, 5).astype(np.float32)
        y = np.array([], dtype=[('event', bool), ('time', np.float32)])
        
        model = AcceleratedFailureTime()
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_empty_training_data_cox_regression(self):
        """ERR-001: Models handle empty training data"""
        from bsix.models import CoxRegression
        
        X = np.array([]).reshape(0, 5).astype(np.float32)
        y = np.array([], dtype=[('event', bool), ('time', np.float32)])
        
        model = CoxRegression()
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_empty_training_data_cox_regression_with_time_varying(self):
        """ERR-001: Models handle empty training data"""
        from bsix.models import CoxRegressionWithTimeVarying
        
        X = np.array([]).reshape(0, 5).astype(np.float32)
        y = np.array([], dtype=[('event', bool), ('time_start', np.float32), ('time_stop', np.float32)])
        
        model = CoxRegressionWithTimeVarying()
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_empty_training_data_deep_hit(self):
        """ERR-001: Models handle empty training data"""
        from bsix.models import DeepHit
        
        X = np.array([]).reshape(0, 5).astype(np.float32)
        y = np.array([], dtype=[('event', bool), ('time', np.float32)])
        
        model = DeepHit(X.shape[1], len(np.unique(y["event"])) - 1, 10)
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_empty_training_data_deep_multi_task(self):
        """ERR-001: Models handle empty training data"""
        from bsix.models import DeepMultiTask
        
        X = np.array([]).reshape(0, 5).astype(np.float32)
        y = np.array([], dtype=[('event', bool), ('time', np.float32)])
        
        model = DeepMultiTask(X.shape[1])
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_empty_training_data_deep_surv(self):
        """ERR-001: Models handle empty training data"""
        from bsix.models import DeepSurv
        
        X = np.array([]).reshape(0, 5).astype(np.float32)
        y = np.array([], dtype=[('event', bool), ('time', np.float32)])
        
        model = DeepSurv(X.shape[1])
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_empty_training_data_deep_time_varying(self):
        """ERR-001: Models handle empty training data"""
        from bsix.models import DeepTimeVarying
        
        X = np.array([]).reshape(0, 5).astype(np.float32)
        y = np.array([], dtype=[('event', bool), ('time', np.float32)])
        
        model = DeepTimeVarying(0)
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_empty_training_data_random_surv_forest(self):
        """ERR-001: Models handle empty training data"""
        from bsix.models import RandomSurvForest
        
        X = np.array([]).reshape(0, 5).astype(np.float32)
        y = np.array([], dtype=[('event', bool), ('time', np.float32)])
        
        model = RandomSurvForest(seed=42)
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_empty_training_data_survival_tab_pfn(self):
        """ERR-001: Models handle empty training data"""
        from bsix.models import SurvivalTabPFN
        
        X = np.array([]).reshape(0, 5).astype(np.float32)
        y = np.array([], dtype=[('event', bool), ('time', np.float32)])
        
        model = SurvivalTabPFN()
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_empty_training_data_surv_tree(self):
        """ERR-001: Models handle empty training data"""
        from bsix.models import SurvTree
        
        X = np.array([]).reshape(0, 5).astype(np.float32)
        y = np.array([], dtype=[('event', bool), ('time', np.float32)])
        
        model = SurvTree()
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)
    
    @pytest.mark.edge_case
    def test_single_sample_training_accelerated_failure_time(self, single_survival_data):
        """ERR-010: Single training sample edge case"""
        from bsix.models import AcceleratedFailureTime
        
        X, y = single_survival_data
        
        model = AcceleratedFailureTime()
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_single_sample_training_cox_regression(self, single_survival_data):
        """ERR-010: Single training sample edge case"""
        from bsix.models import CoxRegression
        
        X, y = single_survival_data
        
        model = CoxRegression()
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_single_sample_training_cox_regression_with_time_varying(self, single_survival_data_time_varying):
        """ERR-010: Single training sample edge case"""
        from bsix.models import CoxRegressionWithTimeVarying
        
        X, y = single_survival_data_time_varying
        
        model = CoxRegressionWithTimeVarying()
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_single_sample_training_deep_hit(self, single_survival_data):
        """ERR-010: Single training sample edge case"""
        from bsix.models import DeepHit
        
        X, y = single_survival_data
        
        model = DeepHit(X.shape[1], len(np.unique(y["event"])) - 1, 10)
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_single_sample_training_deep_multi_task(self, single_survival_data_multi_event):
        """ERR-010: Single training sample edge case"""
        from bsix.models import DeepMultiTask
        
        X, y = single_survival_data_multi_event

        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y.ndim == 1 else y.shape[1])), 8).tolist()
        
        model = DeepMultiTask(X.shape[1], coef_likelihood=coef_likelihood)
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_single_sample_training_deep_surv(self, single_survival_data):
        """ERR-010: Single training sample edge case"""
        from bsix.models import DeepSurv
        
        X, y = single_survival_data
        
        model = DeepSurv(X.shape[1])
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_single_sample_training_deep_time_varying(self, single_survival_data_time_varying):
        """ERR-010: Single training sample edge case"""
        from bsix.models import DeepTimeVarying
        
        X, y = single_survival_data_time_varying
        
        model = DeepTimeVarying(X.shape[1])
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_single_sample_training_random_surv_forest(self, single_survival_data):
        """ERR-010: Single training sample edge case"""
        from bsix.models import RandomSurvForest
        
        X, y = single_survival_data
        
        model = RandomSurvForest(seed=42)
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_single_sample_training_survival_tab_pfn(self, single_survival_data):
        """ERR-010: Single training sample edge case"""
        from bsix.models import SurvivalTabPFN
        
        X, y = single_survival_data
        
        model = SurvivalTabPFN()
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_single_sample_training_surv_tree(self, single_survival_data):
        """ERR-010: Single training sample edge case"""
        from bsix.models import SurvTree
        
        X, y = single_survival_data
        
        model = SurvTree()
        
        # Should raise appropriate error, not crash silently
        with pytest.raises((ValueError)):
            model.fit(X, y)


class TestSpecialValues:
    """Tests for special numeric values (ERR-002)"""
    
    @pytest.mark.edge_case
    def test_infinite_features_accelerated_failure_time(self, simple_survival_data):
        """ERR-002: Features with Inf values"""
        from bsix.models import AcceleratedFailureTime
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.inf
        
        model = AcceleratedFailureTime()
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_infinite_features_cox_regression(self, simple_survival_data):
        """ERR-002: Features with Inf values"""
        from bsix.models import CoxRegression
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.inf
        
        model = CoxRegression()
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_infinite_features_cox_regression_with_time_varying(self, simple_survival_data_timevarying):
        """ERR-002: Features with Inf values"""
        from bsix.models import CoxRegressionWithTimeVarying
        
        X, y = simple_survival_data_timevarying

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.inf
        
        model = CoxRegressionWithTimeVarying()
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_infinite_features_deep_hit(self, simple_survival_data):
        """ERR-002: Features with Inf values"""
        from bsix.models import DeepHit
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.inf
        
        model = DeepHit(X.shape[1], len(np.unique(y["event"])) - 1, 10)
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_infinite_features_deep_multi_task(self, simple_survival_data_multi_event):
        """ERR-002: Features with Inf values"""
        from bsix.models import DeepMultiTask
        
        X, y = simple_survival_data_multi_event

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.inf
        
        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y.ndim == 1 else y.shape[1])), 8).tolist()

        model = DeepMultiTask(X.shape[1], coef_likelihood=coef_likelihood)
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_infinite_features_deep_surv(self, simple_survival_data):
        """ERR-002: Features with Inf values"""
        from bsix.models import DeepSurv
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.inf
        
        model = DeepSurv(X.shape[1])
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_infinite_features_deep_time_varying(self, simple_survival_data_timevarying):
        """ERR-002: Features with Inf values"""
        from bsix.models import DeepTimeVarying
        
        X, y = simple_survival_data_timevarying

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.inf
        
        model = DeepTimeVarying(X.shape[1])
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_infinite_features_random_surv_forest(self, simple_survival_data):
        """ERR-002: Features with Inf values"""
        from bsix.models import RandomSurvForest
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)
        
        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.inf
        
        model = RandomSurvForest(seed=42)
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_infinite_features_survival_tab_pfn(self, simple_survival_data):
        """ERR-002: Features with Inf values"""
        from bsix.models import SurvivalTabPFN
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.inf
        
        model = SurvivalTabPFN()
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_infinite_features_surv_tree(self, simple_survival_data):
        """ERR-002: Features with Inf values"""
        from bsix.models import SurvTree
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.inf
        
        model = SurvTree()
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)
    
    @pytest.mark.edge_case
    def test_nan_features_accelerated_failure_time(self, simple_survival_data):
        """ERR-002: Features with NaN values"""
        from bsix.models import AcceleratedFailureTime
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.nan
        
        model = AcceleratedFailureTime()
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_nan_features_cox_regression(self, simple_survival_data):
        """ERR-002: Features with NaN values"""
        from bsix.models import CoxRegression
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.nan
        
        model = CoxRegression()
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_nan_features_cox_regression_with_time_varying(self, simple_survival_data_timevarying):
        """ERR-002: Features with NaN values"""
        from bsix.models import CoxRegressionWithTimeVarying
        
        X, y = simple_survival_data_timevarying

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.nan
        
        model = CoxRegressionWithTimeVarying()
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_nan_features_deep_hit(self, simple_survival_data):
        """ERR-002: Features with NaN values"""
        from bsix.models import DeepHit
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.nan
        
        model = DeepHit(X.shape[1], len(np.unique(y["event"])) - 1, 10)
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_nan_features_deep_multi_task(self, simple_survival_data_multi_event):
        """ERR-002: Features with NaN values"""
        from bsix.models import DeepMultiTask
        
        X, y = simple_survival_data_multi_event

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.nan
        
        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y.ndim == 1 else y.shape[1])), 8).tolist()

        model = DeepMultiTask(X.shape[1], coef_likelihood=coef_likelihood)
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_nan_features_deep_surv(self, simple_survival_data):
        """ERR-002: Features with NaN values"""
        from bsix.models import DeepSurv
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.nan
        
        model = DeepSurv(X.shape[1])
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_nan_features_deep_time_varying(self, simple_survival_data_timevarying):
        """ERR-002: Features with NaN values"""
        from bsix.models import DeepTimeVarying
        
        X, y = simple_survival_data_timevarying

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.nan
        
        model = DeepTimeVarying(X.shape[1])
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_nan_features_random_surv_forest(self, simple_survival_data):
        """ERR-002: Features with NaN values"""
        from bsix.models import RandomSurvForest
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)
        
        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.nan
        
        model = RandomSurvForest(seed=42)
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_nan_features_survival_tab_pfn(self, simple_survival_data):
        """ERR-002: Features with NaN values"""
        from bsix.models import SurvivalTabPFN
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.nan
        
        model = SurvivalTabPFN()
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_nan_features_surv_tree(self, simple_survival_data):
        """ERR-002: Features with NaN values"""
        from bsix.models import SurvTree
        
        X, y = simple_survival_data

        num = X.shape[0]
        features = X.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X[np.ix_(index_nan, col_idx)] = np.nan
        
        model = SurvTree()
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)
    
    
    @pytest.mark.edge_case
    def test_nan_predictions_accelerated_failure_time(self, simple_survival_data):
        """ERR-003: predict with NaN in input"""
        from bsix.models import AcceleratedFailureTime
        
        X_train, y_train = simple_survival_data
        
        model = AcceleratedFailureTime()
        model.fit(X_train, y_train)

        num = X_train.shape[0]
        features = X_train.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)

        # Predict with NaN
        X_test = X_train.copy()
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X_test[np.ix_(index_nan, col_idx)] = np.nan
        
        # Should handle
        with pytest.raises((ValueError)):
            pred = model.predict(X_test)

    @pytest.mark.edge_case
    def test_nan_predictions_cox_regression(self, simple_survival_data):
        """ERR-003: predict with NaN in input"""
        from bsix.models import CoxRegression
        
        X_train, y_train = simple_survival_data
        
        model = CoxRegression()
        model.fit(X_train, y_train)

        num = X_train.shape[0]
        features = X_train.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)

        # Predict with NaN
        X_test = X_train.copy()
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X_test[np.ix_(index_nan, col_idx)] = np.nan
        
        # Should handle
        with pytest.raises((ValueError)):
            pred = model.predict(X_test)

    @pytest.mark.edge_case
    def test_nan_predictions_cox_regression_with_time_varying(self, simple_survival_data_timevarying):
        """ERR-003: predict with NaN in input"""
        from bsix.models import CoxRegressionWithTimeVarying
        
        X_train, y_train = simple_survival_data_timevarying
        
        model = CoxRegressionWithTimeVarying()
        model.fit(X_train, y_train)

        num = X_train.shape[0]
        features = X_train.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)

        # Predict with NaN
        X_test = X_train.copy()
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X_test[np.ix_(index_nan, col_idx)] = np.nan
        
        # Should handle
        with pytest.raises((ValueError)):
            pred = model.predict(X_test)

    @pytest.mark.edge_case
    def test_nan_predictions_deep_hit(self, simple_survival_data):
        """ERR-003: predict with NaN in input"""
        from bsix.models import DeepHit
        
        X_train, y_train = simple_survival_data
        
        model = DeepHit(X_train.shape[1], len(np.unique(y_train["event"])) - 1, 10)
        model.fit(X_train, y_train)

        num = X_train.shape[0]
        features = X_train.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)

        # Predict with NaN
        X_test = X_train.copy()
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X_test[np.ix_(index_nan, col_idx)] = np.nan
        
        # Should handle
        with pytest.raises((ValueError)):
            pred = model.predict(X_test)

    @pytest.mark.edge_case
    def test_nan_predictions_deep_multi_task(self, simple_survival_data_multi_event):
        """ERR-003: predict with NaN in input"""
        from bsix.models import DeepMultiTask
        
        X_train, y_train = simple_survival_data_multi_event
        
        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y_train.ndim == 1 else y_train.shape[1])), 8).tolist()

        model = DeepMultiTask(X_train.shape[1], coef_likelihood=coef_likelihood)
        model.fit(X_train, y_train)

        num = X_train.shape[0]
        features = X_train.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)

        # Predict with NaN
        X_test = X_train.copy()
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X_test[np.ix_(index_nan, col_idx)] = np.nan
        
        # Should handle
        with pytest.raises((ValueError)):
            pred = model.predict(X_test)

    @pytest.mark.edge_case
    def test_nan_predictions_deep_surv(self, simple_survival_data):
        """ERR-003: predict with NaN in input"""
        from bsix.models import DeepSurv
        
        X_train, y_train = simple_survival_data
        
        model = DeepSurv(X_train.shape[1])
        model.fit(X_train, y_train)

        num = X_train.shape[0]
        features = X_train.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)

        # Predict with NaN
        X_test = X_train.copy()
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X_test[np.ix_(index_nan, col_idx)] = np.nan
        
        # Should handle
        with pytest.raises((ValueError)):
            pred = model.predict(X_test)

    @pytest.mark.edge_case
    def test_nan_predictions_deep_time_varying(self, simple_survival_data_timevarying):
        """ERR-003: predict with NaN in input"""
        from bsix.models import DeepTimeVarying
        
        X_train, y_train = simple_survival_data_timevarying
        
        model = DeepTimeVarying(X_train.shape[1])
        model.fit(X_train, y_train)

        num = X_train.shape[0]
        features = X_train.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)

        # Predict with NaN
        X_test = X_train.copy()
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X_test[np.ix_(index_nan, col_idx)] = np.nan
        
        # Should handle
        with pytest.raises((ValueError)):
            pred = model.predict(X_test)

    @pytest.mark.edge_case
    def test_nan_predictions_random_surv_forest(self, simple_survival_data):
        """ERR-003: predict with NaN in input"""
        from bsix.models import RandomSurvForest
        
        X_train, y_train = simple_survival_data
        
        model = RandomSurvForest(seed=42)
        model.fit(X_train, y_train)

        num = X_train.shape[0]
        features = X_train.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)

        # Predict with NaN
        X_test = X_train.copy()
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X_test[np.ix_(index_nan, col_idx)] = np.nan
        
        # Should handle
        with pytest.raises((ValueError)):
            pred = model.predict(X_test)

    @pytest.mark.edge_case
    def test_nan_predictions_survival_tab_pfn(self, simple_survival_data):
        """ERR-003: predict with NaN in input"""
        from bsix.models import SurvivalTabPFN
        
        X_train, y_train = simple_survival_data
        
        model = SurvivalTabPFN()
        model.fit(X_train, y_train)

        num = X_train.shape[0]
        features = X_train.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)

        # Predict with NaN
        X_test = X_train.copy()
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X_test[np.ix_(index_nan, col_idx)] = np.nan
        
        # Should handle
        with pytest.raises((ValueError)):
            pred = model.predict(X_test)

    @pytest.mark.edge_case
    def test_nan_predictions_surv_tree(self, simple_survival_data):
        """ERR-003: predict with NaN in input"""
        from bsix.models import SurvTree
        
        X_train, y_train = simple_survival_data
        
        model = SurvTree()
        model.fit(X_train, y_train)

        num = X_train.shape[0]
        features = X_train.shape[1]
        num_nan = int(num * 0.10)

        np.random.seed(42)

        # Predict with NaN
        X_test = X_train.copy()
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        col_idx = np.random.choice(features, size=3, replace=False)
        X_test[np.ix_(index_nan, col_idx)] = np.nan
        
        # Should handle
        with pytest.raises((ValueError)):
            pred = model.predict(X_test)


class TestDataTypeHandling:
    """Tests for data type mismatches (ERR-004)"""
    
    @pytest.mark.edge_case
    def test_wrong_y_dtype_accelerated_failure_time(self, wrong_dtype_survival_data):
        """ERR-004: y array with wrong dtype"""
        from bsix.models import AcceleratedFailureTime
        
        X, y = wrong_dtype_survival_data
        
        model = AcceleratedFailureTime()
        
        # Should handle or raise gracefully
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_wrong_y_dtype_cox_regression(self, wrong_dtype_survival_data):
        """ERR-004: y array with wrong dtype"""
        from bsix.models import CoxRegression
        
        X, y = wrong_dtype_survival_data
        
        model = CoxRegression()
        
        # Should handle or raise gracefully
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_wrong_y_dtype_cox_regression_with_time_varying(self, wrong_dtype_timevarying_survival_data):
        """ERR-004: y array with wrong dtype"""
        from bsix.models import CoxRegressionWithTimeVarying
        
        X, y = wrong_dtype_timevarying_survival_data
        
        model = CoxRegressionWithTimeVarying()
        
        # Should handle or raise gracefully
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_wrong_y_dtype_deep_hit(self, wrong_dtype_survival_data):
        """ERR-004: y array with wrong dtype"""
        from bsix.models import DeepHit
        
        X, y = wrong_dtype_survival_data
        
        model = DeepHit(X.shape[1], len(np.unique(y[:, 0])) - 1, 10)
        
        # Should handle or raise gracefully
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_wrong_y_dtype_deep_multi_task(self, wrong_dtype_multi_event_survival_data):
        """ERR-004: y array with wrong dtype"""
        from bsix.models import DeepMultiTask
        
        X, y = wrong_dtype_multi_event_survival_data
        
        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y.ndim == 1 else y.shape[1])), 8).tolist()
        
        model = DeepMultiTask(X.shape[1], coef_likelihood=coef_likelihood)
        
        # Should handle or raise gracefully
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_wrong_y_dtype_deep_surv(self, wrong_dtype_survival_data):
        """ERR-004: y array with wrong dtype"""
        from bsix.models import DeepSurv
        
        X, y = wrong_dtype_survival_data
        
        model = DeepSurv(X.shape[1])
        
        # Should handle or raise gracefully
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_wrong_y_dtype_deep_time_varying(self, wrong_dtype_timevarying_survival_data):
        """ERR-004: y array with wrong dtype"""
        from bsix.models import CoxRegression
        
        X, y = wrong_dtype_timevarying_survival_data
        
        model = CoxRegression()
        
        # Should handle or raise gracefully
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_wrong_y_dtype_random_surv_forest(self, wrong_dtype_survival_data):
        """ERR-004: y array with wrong dtype"""
        from bsix.models import RandomSurvForest
        
        X, y = wrong_dtype_survival_data
        
        model = RandomSurvForest(seed=42)
        
        # Should handle or raise gracefully
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_wrong_y_dtype_survival_tab_pfn(self, wrong_dtype_survival_data):
        """ERR-004: y array with wrong dtype"""
        from bsix.models import SurvivalTabPFN
        
        X, y = wrong_dtype_survival_data
        
        model = SurvivalTabPFN()
        
        # Should handle or raise gracefully
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_wrong_y_dtype_surv_tree(self, wrong_dtype_survival_data):
        """ERR-004: y array with wrong dtype"""
        from bsix.models import SurvTree
        
        X, y = wrong_dtype_survival_data
        
        model = SurvTree()
        
        # Should handle or raise gracefully
        with pytest.raises((ValueError)):
            model.fit(X, y)



class TestSeedHandling:
    """Tests for random seed edge cases (ERR-005)"""
    
    @pytest.mark.edge_case
    def test_nan_seed_deep_hit(self, simple_survival_data):
        """ERR-005: Nan seed value"""
        from bsix.models import DeepHit
        
        X, y = simple_survival_data
        
        model = DeepHit(X.shape[1], len(np.unique(y["event"])) - 1, 10, seed=None)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_nan_seed_deep_multi_task(self, simple_survival_data_multi_event):
        """ERR-005: Nan seed value"""
        from bsix.models import DeepMultiTask
        
        X, y = simple_survival_data_multi_event
        
        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y.ndim == 1 else y.shape[1])), 8).tolist()
        
        model = DeepMultiTask(X.shape[1], coef_likelihood=coef_likelihood, seed=None)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_nan_seed_deep_surv(self, simple_survival_data):
        """ERR-005: Nan seed value"""
        from bsix.models import DeepSurv
        
        X, y = simple_survival_data
        
        model = DeepSurv(X.shape[1], seed=None)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_nan_seed_deep_time_varying(self, simple_survival_data_timevarying):
        """ERR-005: Nan seed value"""
        from bsix.models import DeepTimeVarying
        
        X, y = simple_survival_data_timevarying
        
        model = DeepTimeVarying(X.shape[1], seed=None)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_nan_seed_random_surv_forest(self, simple_survival_data):
        """ERR-005: Nan seed value"""
        from bsix.models import RandomSurvForest
        
        X, y = simple_survival_data
        
        model = RandomSurvForest(seed=None)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_nan_seed_survival_tab_pfn(self, simple_survival_data):
        """ERR-005: Nan seed value"""
        from bsix.models import SurvivalTabPFN
        
        X, y = simple_survival_data

        model = SurvivalTabPFN(seed=None)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_nan_seed_surv_tree(self, simple_survival_data):
        """ERR-005: Nan seed value"""
        from bsix.models import SurvTree
        
        X, y = simple_survival_data

        model = SurvTree(seed=None)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_large_seed_deep_hit(self, simple_survival_data):
        """Large seed value"""
        from bsix.models import DeepHit
        
        X, y = simple_survival_data
        
        model = DeepHit(X.shape[1], len(np.unique(y["event"])) - 1, 10, seed=1e10)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_large_seed_deep_multi_task(self, simple_survival_data_multi_event):
        """Large seed value"""
        from bsix.models import DeepMultiTask
        
        X, y = simple_survival_data_multi_event
        
        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y.ndim == 1 else y.shape[1])), 8).tolist()
        
        model = DeepMultiTask(X.shape[1], coef_likelihood=coef_likelihood, seed=1e10)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_large_seed_deep_surv(self, simple_survival_data):
        """Large seed value"""
        from bsix.models import DeepSurv
        
        X, y = simple_survival_data
        
        model = DeepSurv(X.shape[1], seed=1e10)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_large_seed_deep_time_varying(self, simple_survival_data_timevarying):
        """Large seed value"""
        from bsix.models import DeepTimeVarying
        
        X, y = simple_survival_data_timevarying
        
        model = DeepTimeVarying(X.shape[1], seed=1e10)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_large_seed_random_surv_forest(self, simple_survival_data):
        """Large seed value"""
        from bsix.models import RandomSurvForest
        
        X, y = simple_survival_data
        
        model = RandomSurvForest(seed=1e10)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_large_seed_survival_tab_pfn(self, simple_survival_data):
        """Large seed value"""
        from bsix.models import SurvivalTabPFN
        
        X, y = simple_survival_data

        model = SurvivalTabPFN(seed=1e10)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None

    @pytest.mark.edge_case
    def test_large_seed_surv_tree(self, simple_survival_data):
        """Large seed value"""
        from bsix.models import SurvTree
        
        X, y = simple_survival_data

        model = SurvTree(seed=1e10)

        model.fit(X, y)
        
        # Should handle nan seed
        assert model is not None


class TestBoundaryTimes:
    """Tests for boundary values in survival times (ERR-008)""" 
    
    @pytest.mark.edge_case
    def test_invalid_times_accelerated_failure_time(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import AcceleratedFailureTime
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 0
        
        model = AcceleratedFailureTime()

        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_invalid_times_cox_regression(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import CoxRegression
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 0
        
        model = CoxRegression()

        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_invalid_times_cox_regression_with_time_varying(self, simple_survival_data_timevarying):
        """ERR-008: Invalid times"""
        from bsix.models import CoxRegressionWithTimeVarying
        
        X, y = simple_survival_data_timevarying
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time_stop"][index_nan] = 0
        
        model = CoxRegressionWithTimeVarying()

        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_invalid_times_deep_hit(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import DeepHit
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 0
        
        model = DeepHit(X.shape[1], len(np.unique(y["event"])) - 1, 10)

        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_invalid_times_deep_multi_task(self, simple_survival_data_multi_event):
        """ERR-008: Invalid times"""
        from bsix.models import DeepMultiTask
        
        X, y = simple_survival_data_multi_event
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 0
        
        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y.ndim == 1 else y.shape[1])), 8).tolist()

        model = DeepMultiTask(X.shape[1], coef_likelihood=coef_likelihood)

        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_invalid_times_deep_surv(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import DeepSurv
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 0
        
        model = DeepSurv(X.shape[1])

        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_invalid_times_deep_time_varying(self, simple_survival_data_timevarying):
        """ERR-008: Invalid times"""
        from bsix.models import DeepTimeVarying
        
        X, y = simple_survival_data_timevarying
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time_stop"][index_nan] = 0
        
        model = DeepTimeVarying(X.shape[1])

        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_invalid_times_random_surv_forest(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import RandomSurvForest
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 0
        
        model = RandomSurvForest(seed=42)

        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_invalid_times_survival_tab_pfn(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import SurvivalTabPFN
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 0
        
        model = SurvivalTabPFN()

        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.edge_case
    def test_invalid_times_surv_tree(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import SurvTree
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 0
        
        model = SurvTree()

        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)
    
    @pytest.mark.edge_case
    def test_large_times_accelerated_failure_time(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import AcceleratedFailureTime
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 1e10
        
        model = AcceleratedFailureTime()

        # Should handle
        model.fit(X, y)

    @pytest.mark.edge_case
    def test_large_times_cox_regression(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import CoxRegression
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 1e10
        
        model = CoxRegression()

        # Should handle
        model.fit(X, y)

    @pytest.mark.edge_case
    def test_large_times_cox_regression_with_time_varying(self, simple_survival_data_timevarying):
        """ERR-008: Invalid times"""
        from bsix.models import CoxRegressionWithTimeVarying
        
        X, y = simple_survival_data_timevarying
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time_stop"][index_nan] = 1e10
        
        model = CoxRegressionWithTimeVarying()

        # Should handle
        model.fit(X, y)

    @pytest.mark.edge_case
    def test_large_times_deep_hit(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import DeepHit
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 1e10
        
        model = DeepHit(X.shape[1], len(np.unique(y["event"])) - 1, 10)

        # Should handle
        model.fit(X, y)

    @pytest.mark.edge_case
    def test_large_times_deep_multi_task(self, simple_survival_data_multi_event):
        """ERR-008: Invalid times"""
        from bsix.models import DeepMultiTask
        
        X, y = simple_survival_data_multi_event
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 1e10
        
        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y.ndim == 1 else y.shape[1])), 8).tolist()

        model = DeepMultiTask(X.shape[1], coef_likelihood=coef_likelihood)

        # Should handle
        model.fit(X, y)

    @pytest.mark.edge_case
    def test_large_times_deep_surv(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import DeepSurv
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 1e10
        
        model = DeepSurv(X.shape[1])

        # Should handle
        model.fit(X, y)

    @pytest.mark.edge_case
    def test_large_times_deep_time_varying(self, simple_survival_data_timevarying):
        """ERR-008: Invalid times"""
        from bsix.models import DeepTimeVarying
        
        X, y = simple_survival_data_timevarying
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time_stop"][index_nan] = 1e10
        
        model = DeepTimeVarying(X.shape[1])

        # Should handle
        model.fit(X, y)

    @pytest.mark.edge_case
    def test_large_times_random_surv_forest(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import RandomSurvForest
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 1e10
        
        model = RandomSurvForest(seed=42)

        # Should handle
        model.fit(X, y)

    @pytest.mark.edge_case
    def test_large_times_survival_tab_pfn(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import SurvivalTabPFN
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 1e10
        
        model = SurvivalTabPFN()

        # Should handle
        model.fit(X, y)

    @pytest.mark.edge_case
    def test_large_times_surv_tree(self, simple_survival_data):
        """ERR-008: Invalid times"""
        from bsix.models import SurvTree
        
        X, y = simple_survival_data
        
        num = X.shape[0]
        num_nan = int(num * 0.10)

        np.random.seed(42)
        index_nan = np.random.choice(num, size=num_nan, replace=False)
        y["time"][index_nan] = 1e10
        
        model = SurvTree()

        # Should handle
        model.fit(X, y)


class TestEventCensoring:
    """Tests for extreme censoring scenarios (ERR-009)"""
    
    @pytest.mark.integration
    def test_pipeline_all_censored_accelerated_failure_time(self, survival_data_all_censored):
        """Pipeline with all-censored test set"""
        from bsix.models import AcceleratedFailureTime
       
        X, y = survival_data_all_censored
       
        model = AcceleratedFailureTime()

        with pytest.raises((ValueError)):
            model.fit(X, y)


    @pytest.mark.integration
    def test_pipeline_all_censored_cox_regression(self, survival_data_all_censored):
        """Pipeline with all-censored test set"""
        from bsix.models import CoxRegression
        
        X, y = survival_data_all_censored
        
        model = CoxRegression()
        
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.integration
    def test_pipeline_all_censored_cox_regression_with_time_varying(self, survival_data_timevarying_all_censored):
        """Pipeline with all-censored test set"""
        from bsix.models import CoxRegressionWithTimeVarying
        
        X, y = survival_data_timevarying_all_censored
        
        model = CoxRegressionWithTimeVarying()
        
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.integration
    def test_pipeline_all_censored_deep_hit(self, survival_data_all_censored):
        """Pipeline with all-censored test set"""
        from bsix.models import DeepHit
        
        X, y = survival_data_all_censored
        
        model = DeepHit(X.shape[1], len(np.unique(y["event"])) - 1, 10)
        
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.integration
    def test_pipeline_all_censored_deep_multi_task(self, survival_data_multi_event_all_censored):
        """Pipeline with all-censored test set"""
        from bsix.models import DeepMultiTask
        
        X, y = survival_data_multi_event_all_censored
        
        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y.ndim == 1 else y.shape[1])), 8).tolist()
        
        model = DeepMultiTask(X.shape[1], coef_likelihood=coef_likelihood)
        
        with pytest.raises((ValueError)):
            model.fit(X, y)


    @pytest.mark.integration
    def test_pipeline_all_censored_deep_surv(self, survival_data_all_censored):
        """Pipeline with all-censored test set"""
        from bsix.models import DeepSurv
       
        X, y = survival_data_all_censored
       
        model = DeepSurv(X.shape[1])

        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)


    @pytest.mark.integration
    def test_pipeline_all_censored_deep_time_varying(self, survival_data_timevarying_all_censored):
        """Pipeline with all-censored test set"""
        from bsix.models import DeepTimeVarying
        
        X, y = survival_data_timevarying_all_censored
        
        model = DeepTimeVarying(X.shape[1])
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.integration
    def test_pipeline_all_censored_random_surv_forest(self, survival_data_all_censored):
        """Pipeline with all-censored test set"""
        from bsix.models import RandomSurvForest
        
        X, y = survival_data_all_censored
        
        model = RandomSurvForest(seed=42)
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.integration
    def test_pipeline_all_censored_survival_tab_pfn(self, survival_data_all_censored):
        """Pipeline with all-censored test set"""
        from bsix.models import SurvivalTabPFN
        
        X, y = survival_data_all_censored
        
        model = SurvivalTabPFN()
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)

    @pytest.mark.integration
    def test_pipeline_all_censored_surv_tree(self, survival_data_all_censored):
        """Pipeline with all-censored test set"""
        from bsix.models import SurvTree
        
        X, y = survival_data_all_censored
        
        model = SurvTree()
        
        # Should handle
        with pytest.raises((ValueError)):
            model.fit(X, y)


class TestImportAndDependencies:
    """Tests for import errors and missing dependencies"""
    
    def test_all_core_imports(self):
        """All core modules can be imported"""

        import bsix.models.base
        import bsix.models.loggers
        import bsix.models.metodologies
        import bsix.models.nets
        import bsix.utils
        import bsix.utils.calculate_metrics
        import bsix.utils.classification_metrics
        import bsix.utils.estimators
        import bsix.utils.load_data
        import bsix.utils.load_results
        import bsix.utils.survival_metrics
        import bsix.utils.survival_utils
        import execution
        import execution.config
        import execution.flows
