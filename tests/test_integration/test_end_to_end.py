"""
Integration tests for end-to-end workflows.
Tests for complete BSIX pipelines following BSIX_TEST_SPECIFICATION.md
"""

import pytest
import numpy as np


class TestEndToEndAcceleratedFailureTime:
    """End-to-end tests for AcceleratedFailureTime pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_accelerated_failure_time(self, dataframe_mixed_types):
        """Full pipeline: data loading -> AcceleratedFailureTime -> metrics"""
        from bsix.utils.load_data import get_data
        from bsix.models import AcceleratedFailureTime
        from bsix.utils.calculate_metrics import compute_metrics
        
        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)
        
        # 2. Create model
        model = AcceleratedFailureTime()
        
        # 3. Train
        model.fit(X_train, y_train)
        
        # 4. Predict
        predictions_validation = model.predict(X_validation)
        predictions_test = model.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0
    
    @pytest.mark.integration
    def test_accelerated_failure_time_with_cross_validation(self, dataframe_mixed_types):
        """AcceleratedFailureTime with train/validation/test split"""
        from bsix.utils import get_data, get_estimator
        from bsix.utils.calculate_metrics import compute_metrics
        
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

        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)

        X_train_val, y_train_val, validation_split = _get_validation(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation
        )

        # 2. Create model
        estimator = get_estimator(
            estimator_name="AcceleratedFailureTime",
            inputs=X_train_val,
            labels=y_train_val,
            valid_data=validation_split,
            seed=42,
            n_jobs=5,
            n_iter=30,
        )
        
        # 3. Train
        estimator.fit(X_train_val, y_train_val)
        
        # 4. Predict
        predictions_validation = estimator.predict(X_validation)
        predictions_test = estimator.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0


class TestEndToEndCoxRegression:
    """End-to-end tests for CoxRegression pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_cox_regression(self, dataframe_mixed_types):
        """Full pipeline: data loading -> CoxRegression -> metrics"""
        from bsix.utils.load_data import get_data
        from bsix.models import CoxRegression
        from bsix.utils.calculate_metrics import compute_metrics
        
        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)
        
        # 2. Create model
        model = CoxRegression()
        
        # 3. Train
        model.fit(X_train, y_train)
        
        # 4. Predict
        predictions_validation = model.predict(X_validation)
        predictions_test = model.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0
    
    @pytest.mark.integration
    def test_cox_regression_with_cross_validation(self, dataframe_mixed_types):
        """CoxRegression with train/validation/test split"""
        from bsix.utils import get_data, get_estimator
        from bsix.utils.calculate_metrics import compute_metrics
        
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

        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)

        X_train_val, y_train_val, validation_split = _get_validation(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation
        )

        # 2. Create model
        estimator = get_estimator(
            estimator_name="CoxRegression",
            inputs=X_train_val,
            labels=y_train_val,
            valid_data=validation_split,
            seed=42,
            n_jobs=5,
            n_iter=30,
        )
        
        # 3. Train
        estimator.fit(X_train_val, y_train_val)
        
        # 4. Predict
        predictions_validation = estimator.predict(X_validation)
        predictions_test = estimator.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0


class TestEndToEndCoxRegressionWithTimeVarying:
    """End-to-end tests for CoxRegressionWithTimeVarying pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_cox_regression_with_time_varying(self, dataframe_time_varying):
        """Full pipeline: data loading -> CoxRegressionWithTimeVarying -> metrics"""
        from bsix.utils.load_data import get_data
        from bsix.models import CoxRegressionWithTimeVarying
        from bsix.utils.calculate_metrics import compute_metrics
        
        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_time_varying, seed=42)
        
        # 2. Create model
        model = CoxRegressionWithTimeVarying()
        
        # 3. Train
        model.fit(X_train, y_train)
        
        # 4. Predict
        predictions_validation = model.predict(X_validation)
        predictions_test = model.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0
    
    @pytest.mark.integration
    def test_cox_regression_with_time_varying_with_cross_validation(self, dataframe_time_varying):
        """CoxRegressionWithTimeVarying with train/validation/test split"""
        from bsix.utils import get_data, get_estimator
        from bsix.utils.calculate_metrics import compute_metrics
        
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

        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_time_varying, seed=42)

        X_train_val, y_train_val, validation_split = _get_validation(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation
        )

        # 2. Create model
        estimator = get_estimator(
            estimator_name="CoxRegressionWithTimeVarying",
            inputs=X_train_val,
            labels=y_train_val,
            valid_data=validation_split,
            seed=42,
            n_jobs=5,
            n_iter=30,
        )
        
        # 3. Train
        estimator.fit(X_train_val, y_train_val)
        
        # 4. Predict
        predictions_validation = estimator.predict(X_validation)
        predictions_test = estimator.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0

class TestEndToEndDeepHit:
    """End-to-end tests for DeepHit pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_deep_hit(self, dataframe_mixed_types):
        """Full pipeline: data loading -> DeepHit -> metrics"""
        from bsix.utils.load_data import get_data
        from bsix.models import DeepHit
        from bsix.utils.calculate_metrics import compute_metrics
        
        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)
        
        # 2. Create model
        model = DeepHit(X_train.shape[1], len(np.unique(y_train["event"])) - 1, 10)
        
        # 3. Train
        model.fit(X_train, y_train)
        
        # 4. Predict
        predictions_validation = model.predict(X_validation)
        predictions_test = model.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0
    
    @pytest.mark.integration
    def test_deep_hit_with_cross_validation(self, dataframe_mixed_types):
        """DeepHit with train/validation/test split"""
        from bsix.utils import get_data, get_estimator
        from bsix.utils.calculate_metrics import compute_metrics
        
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

        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)

        X_train_val, y_train_val, validation_split = _get_validation(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation
        )

        # 2. Create model
        estimator = get_estimator(
            estimator_name="DeepHit",
            inputs=X_train_val,
            labels=y_train_val,
            valid_data=validation_split,
            seed=42,
            n_jobs=5,
            n_iter=30,
        )
        
        # 3. Train
        estimator.fit(X_train_val, y_train_val)
        
        # 4. Predict
        predictions_validation = estimator.predict(X_validation)
        predictions_test = estimator.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0

class TestEndToEndDeepMultiTask:
    """End-to-end tests for DeepMultiTask pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_deep_multi_task(self, dataframe_multi_event):
        """Full pipeline: data loading -> DeepMultiTask -> metrics"""
        from bsix.utils.load_data import get_data
        from bsix.models import DeepMultiTask
        from bsix.utils.calculate_metrics import compute_metrics
        
        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_multi_event, seed=42, to_multitask=True)
        
        # 2. Create model
        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y_train.ndim == 1 else y_train.shape[1])), 8).tolist()

        model = DeepMultiTask(X_train.shape[1], coef_likelihood=coef_likelihood)
        
        # 3. Train
        model.fit(X_train, y_train)
        
        # 4. Predict
        predictions_validation = model.predict(X_validation)
        predictions_test = model.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation), y_train.shape[1])
        assert predictions_test.shape == (len(X_test), y_train.shape[1])
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0
    
    @pytest.mark.integration
    def test_deep_multi_task_with_cross_validation(self, dataframe_multi_event):
        """DeepMultiTask with train/validation/test split"""
        from bsix.utils import get_data, get_estimator
        from bsix.utils.calculate_metrics import compute_metrics
        
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

        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_multi_event, seed=42, to_multitask=True)

        X_train_val, y_train_val, validation_split = _get_validation(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation
        )

        # 2. Create model
        estimator = get_estimator(
            estimator_name="DeepMultiTask",
            inputs=X_train_val,
            labels=y_train_val,
            valid_data=validation_split,
            seed=42,
            n_jobs=5,
            n_iter=30,
        )
        
        # 3. Train
        estimator.fit(X_train_val, y_train_val)
        
        # 4. Predict
        predictions_validation = estimator.predict(X_validation)
        predictions_test = estimator.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation), y_train.shape[1])
        assert predictions_test.shape == (len(X_test), y_train.shape[1])
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0


class TestEndToEndDeepSurv:
    """End-to-end tests for DeepSurv pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_deep_surv(self, dataframe_mixed_types):
        """Full pipeline: data loading -> DeepSurv -> metrics"""
        from bsix.utils.load_data import get_data
        from bsix.models import DeepSurv
        from bsix.utils.calculate_metrics import compute_metrics
        
        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)
        
        # 2. Create model
        model = DeepSurv(X_train.shape[1])
        
        # 3. Train
        model.fit(X_train, y_train)
        
        # 4. Predict
        predictions_validation = model.predict(X_validation)
        predictions_test = model.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0
    
    @pytest.mark.integration
    def test_deep_surv_with_cross_validation(self, dataframe_mixed_types):
        """DeepSurv with train/validation/test split"""
        from bsix.utils import get_data, get_estimator
        from bsix.utils.calculate_metrics import compute_metrics
        
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

        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)

        X_train_val, y_train_val, validation_split = _get_validation(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation
        )

        # 2. Create model
        estimator = get_estimator(
            estimator_name="DeepSurv",
            inputs=X_train_val,
            labels=y_train_val,
            valid_data=validation_split,
            seed=42,
            n_jobs=5,
            n_iter=30,
        )
        
        # 3. Train
        estimator.fit(X_train_val, y_train_val)
        
        # 4. Predict
        predictions_validation = estimator.predict(X_validation)
        predictions_test = estimator.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0


class TestEndToEndDeepTimeVarying:
    """End-to-end tests for DeepTimeVarying pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_deep_time_varying(self, dataframe_time_varying):
        """Full pipeline: data loading -> DeepTimeVarying -> metrics"""
        from bsix.utils.load_data import get_data
        from bsix.models import DeepTimeVarying
        from bsix.utils.calculate_metrics import compute_metrics
        
        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_time_varying, seed=42)
        
        # 2. Create model
        model = DeepTimeVarying(X_train.shape[1])
        
        # 3. Train
        model.fit(X_train, y_train)
        
        # 4. Predict
        predictions_validation = model.predict(X_validation)
        predictions_test = model.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0
    
    @pytest.mark.integration
    def test_deep_time_varying_with_cross_validation(self, dataframe_time_varying):
        """DeepTimeVarying with train/validation/test split"""
        from bsix.utils import get_data, get_estimator
        from bsix.utils.calculate_metrics import compute_metrics
        
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

        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_time_varying, seed=42)

        X_train_val, y_train_val, validation_split = _get_validation(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation
        )

        # 2. Create model
        estimator = get_estimator(
            estimator_name="DeepTimeVarying",
            inputs=X_train_val,
            labels=y_train_val,
            valid_data=validation_split,
            seed=42,
            n_jobs=5,
            n_iter=30,
        )
        
        # 3. Train
        estimator.fit(X_train_val, y_train_val)
        
        # 4. Predict
        predictions_validation = estimator.predict(X_validation)
        predictions_test = estimator.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0


class TestEndToEndRandomSurvForest:
    """End-to-end tests for RandomSurvForest pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_random_surv_forest(self, dataframe_mixed_types):
        """Full pipeline: data loading -> RandomSurvForest -> metrics"""
        from bsix.utils.load_data import get_data
        from bsix.models import RandomSurvForest
        from bsix.utils.calculate_metrics import compute_metrics
        
        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)
        
        # 2. Create model
        model = RandomSurvForest(seed=42)
        
        # 3. Train
        model.fit(X_train, y_train)
        
        # 4. Predict
        predictions_validation = model.predict(X_validation)
        predictions_test = model.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0
    
    @pytest.mark.integration
    def test_random_surv_forest_with_cross_validation(self, dataframe_mixed_types):
        """RandomSurvForest with train/validation/test split"""
        from bsix.utils import get_data, get_estimator
        from bsix.utils.calculate_metrics import compute_metrics
        
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

        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)

        X_train_val, y_train_val, validation_split = _get_validation(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation
        )

        # 2. Create model
        estimator = get_estimator(
            estimator_name="RandomSurvForest",
            inputs=X_train_val,
            labels=y_train_val,
            valid_data=validation_split,
            seed=42,
            n_jobs=5,
            n_iter=30,
        )
        
        # 3. Train
        estimator.fit(X_train_val, y_train_val)
        
        # 4. Predict
        predictions_validation = estimator.predict(X_validation)
        predictions_test = estimator.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0

class TestEndToEndSurvivalTabPFN:
    """End-to-end tests for SurvivalTabPFN pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_survival_tab_pfn(self, dataframe_mixed_types):
        """Full pipeline: data loading -> SurvivalTabPFN -> metrics"""
        from bsix.utils.load_data import get_data
        from bsix.models import SurvivalTabPFN
        from bsix.utils.calculate_metrics import compute_metrics
        
        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)
        
        # 2. Create model
        model = SurvivalTabPFN()
        
        # 3. Train
        model.fit(X_train, y_train)
        
        # 4. Predict
        predictions_validation = model.predict(X_validation)
        predictions_test = model.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0
    
    @pytest.mark.integration
    def test_survival_tab_pfn_with_cross_validation(self, dataframe_mixed_types):
        """SurvivalTabPFN with train/validation/test split"""
        from bsix.utils import get_data, get_estimator
        from bsix.utils.calculate_metrics import compute_metrics
        
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

        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)

        X_train_val, y_train_val, validation_split = _get_validation(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation
        )

        # 2. Create model
        estimator = get_estimator(
            estimator_name="SurvivalTabPFN",
            inputs=X_train_val,
            labels=y_train_val,
            valid_data=validation_split,
            seed=42,
            n_jobs=5,
            n_iter=30,
        )
        
        # 3. Train
        estimator.fit(X_train_val, y_train_val)
        
        # 4. Predict
        predictions_validation = estimator.predict(X_validation)
        predictions_test = estimator.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0

class TestEndToEndSurvTree:
    """End-to-end tests for SurvTree pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_surv_tree(self, dataframe_mixed_types):
        """Full pipeline: data loading -> SurvTree -> metrics"""
        from bsix.utils.load_data import get_data
        from bsix.models import SurvTree
        from bsix.utils.calculate_metrics import compute_metrics
        
        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)
        
        # 2. Create model
        model = SurvTree()
        
        # 3. Train
        model.fit(X_train, y_train)
        
        # 4. Predict
        predictions_validation = model.predict(X_validation)
        predictions_test = model.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0
    
    @pytest.mark.integration
    def test_surv_tree_with_cross_validation(self, dataframe_mixed_types):
        """SurvTree with train/validation/test split"""
        from bsix.utils import get_data, get_estimator
        from bsix.utils.calculate_metrics import compute_metrics
        
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

        # 1. Load and prepare data
        X_train, y_train, X_validation, y_validation, X_test, y_test, *_ = get_data(df=dataframe_mixed_types, seed=42)

        X_train_val, y_train_val, validation_split = _get_validation(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation
        )

        # 2. Create model
        estimator = get_estimator(
            estimator_name="SurvTree",
            inputs=X_train_val,
            labels=y_train_val,
            valid_data=validation_split,
            seed=42,
            n_jobs=5,
            n_iter=30,
        )
        
        # 3. Train
        estimator.fit(X_train_val, y_train_val)
        
        # 4. Predict
        predictions_validation = estimator.predict(X_validation)
        predictions_test = estimator.predict(X_test)
        
        # 5. Evaluate metrics
        metrics_validation = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_validation,
            predictions=predictions_validation
        )
        
        metrics_test = compute_metrics(
            train_targets=y_train,
            evaluation_targets=y_test,
            predictions=predictions_test
        )
        
        # Validations
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert predictions_validation.shape == (len(X_validation),)
        assert predictions_test.shape == (len(X_test),)
        assert isinstance(metrics_validation, dict)
        assert len(metrics_validation) > 0
        assert isinstance(metrics_test, dict)
        assert len(metrics_test) > 0


class TestDataPreprocessingVariations:
    """End-to-end tests with different data preprocessing scenarios"""

    @pytest.mark.integration
    def test_pipeline_with_nan_handling_accelerated_failure_time(self, dataframe_with_nans):
        """Full pipeline with NaN handling"""
        from bsix.utils.load_data import get_data
        from bsix.models import AcceleratedFailureTime
       
        X_train, y_train, _, _, X_test, *_ = get_data(df=dataframe_with_nans, seed=42)
       
        # Should work after NaN elimination
        model = AcceleratedFailureTime()
        model.fit(X_train, y_train)
       
        pred = model.predict(X_test)

        assert np.all(np.isfinite(pred))

    @pytest.mark.integration
    def test_pipeline_with_nan_handling_cox_regression(self, dataframe_with_nans):
        """Full pipeline with NaN handling"""
        from bsix.utils.load_data import get_data
        from bsix.models import CoxRegression
       
        X_train, y_train, _, _, X_test, *_ = get_data(df=dataframe_with_nans, seed=42)
       
        # Should work after NaN elimination
        model = CoxRegression()
        model.fit(X_train, y_train)
       
        pred = model.predict(X_test)

        assert np.all(np.isfinite(pred))

    @pytest.mark.integration
    def test_pipeline_with_nan_handling_cox_regression_with_time_varying(self, dataframe_with_nans_time_varying):
        """Full pipeline with NaN handling"""
        from bsix.utils.load_data import get_data
        from bsix.models import CoxRegressionWithTimeVarying
       
        X_train, y_train, _, _, X_test, *_ = get_data(df=dataframe_with_nans_time_varying, seed=42)
       
        # Should work after NaN elimination
        model = CoxRegressionWithTimeVarying()
        model.fit(X_train, y_train)
       
        pred = model.predict(X_test)

        assert np.all(np.isfinite(pred))

    @pytest.mark.integration
    def test_pipeline_with_nan_handling_deep_hit(self, dataframe_with_nans):
        """Full pipeline with NaN handling"""
        from bsix.utils.load_data import get_data
        from bsix.models import DeepHit
       
        X_train, y_train, _, _, X_test, *_ = get_data(df=dataframe_with_nans, seed=42)
       
        # Should work after NaN elimination
        model = DeepHit(X_train.shape[1], len(np.unique(y_train["event"])) - 1, 10)
        model.fit(X_train, y_train)
       
        pred = model.predict(X_test)

        assert np.all(np.isfinite(pred))

    @pytest.mark.integration
    def test_pipeline_with_nan_handling_deep_multi_task(self, dataframe_with_nans_multi_event):
        """Full pipeline with NaN handling"""
        from bsix.utils.load_data import get_data
        from bsix.models import DeepMultiTask
       
        X_train, y_train, _, _, X_test, *_ = get_data(df=dataframe_with_nans_multi_event, seed=42)
       
        # Should work after NaN elimination
        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y_train.ndim == 1 else y_train.shape[1])), 8).tolist()

        model = DeepMultiTask(X_train.shape[1], coef_likelihood=coef_likelihood)
        model.fit(X_train, y_train)
       
        pred = model.predict(X_test)

        assert np.all(np.isfinite(pred))

    @pytest.mark.integration
    def test_pipeline_with_nan_handling_deep_surv(self, dataframe_with_nans):
        """Full pipeline with NaN handling"""
        from bsix.utils.load_data import get_data
        from bsix.models import DeepSurv
       
        X_train, y_train, _, _, X_test, *_ = get_data(df=dataframe_with_nans, seed=42)
       
        # Should work after NaN elimination
        model = DeepSurv(X_train.shape[1])
        model.fit(X_train, y_train)
       
        pred = model.predict(X_test)

        assert np.all(np.isfinite(pred))

    @pytest.mark.integration
    def test_pipeline_with_nan_handling_deep_time_varying(self, dataframe_with_nans_time_varying):
        """Full pipeline with NaN handling"""
        from bsix.utils.load_data import get_data
        from bsix.models import DeepTimeVarying
       
        X_train, y_train, _, _, X_test, *_ = get_data(df=dataframe_with_nans_time_varying, seed=42)
       
        # Should work after NaN elimination
        model = DeepTimeVarying(X_train.shape[1])
        model.fit(X_train, y_train)
       
        pred = model.predict(X_test)

        assert np.all(np.isfinite(pred))

    @pytest.mark.integration
    def test_pipeline_with_nan_handling_random_surv_forest(self, dataframe_with_nans):
        """Full pipeline with NaN handling"""
        from bsix.utils.load_data import get_data
        from bsix.models import RandomSurvForest
       
        X_train, y_train, _, _, X_test, *_ = get_data(df=dataframe_with_nans, seed=42)
       
        # Should work after NaN elimination
        model = RandomSurvForest(seed=42)
        model.fit(X_train, y_train)
       
        pred = model.predict(X_test)

        assert np.all(np.isfinite(pred))

    @pytest.mark.integration
    def test_pipeline_with_nan_handling_survival_tab_pfn(self, dataframe_with_nans):
        """Full pipeline with NaN handling"""
        from bsix.utils.load_data import get_data
        from bsix.models import SurvivalTabPFN
       
        X_train, y_train, _, _, X_test, *_ = get_data(df=dataframe_with_nans, seed=42)
       
        # Should work after NaN elimination
        model = SurvivalTabPFN()
        model.fit(X_train, y_train)
       
        pred = model.predict(X_test)

        assert np.all(np.isfinite(pred))

    @pytest.mark.integration
    def test_pipeline_with_nan_handling_surv_tree(self, dataframe_with_nans):
        """Full pipeline with NaN handling"""
        from bsix.utils.load_data import get_data
        from bsix.models import SurvTree
       
        X_train, y_train, _, _, X_test, *_ = get_data(df=dataframe_with_nans, seed=42)
       
        # Should work after NaN elimination
        model = SurvTree()
        model.fit(X_train, y_train)
       
        pred = model.predict(X_test)

        assert np.all(np.isfinite(pred))


class TestReproducibility:
    """End-to-end tests for reproducibility"""
   
    @pytest.mark.integration
    def test_deterministic_accelerated_failure_time(self, simple_survival_data):
        """AcceleratedFailureTime reproducibility with seed"""
        from bsix.models import AcceleratedFailureTime
       
        X, y = simple_survival_data
       
        # Run 1
        model1 = AcceleratedFailureTime()
        model1.fit(X, y)
        pred1 = model1.predict(X)
       
        # Run 2
        model2 = AcceleratedFailureTime()
        model2.fit(X, y)
        pred2 = model2.predict(X)
       
        # Should be close
        assert np.allclose(pred1, pred2, atol=1e-3)

    @pytest.mark.integration
    def test_deterministic_cox_regression(self, simple_survival_data):
        """CoxRegression reproducibility with seed"""
        from bsix.models import CoxRegression
       
        X, y = simple_survival_data
       
        # Run 1
        model1 = CoxRegression()
        model1.fit(X, y)
        pred1 = model1.predict(X)
       
        # Run 2
        model2 = CoxRegression()
        model2.fit(X, y)
        pred2 = model2.predict(X)
       
        # Should be close
        assert np.allclose(pred1, pred2, atol=1e-3)

    @pytest.mark.integration
    def test_deterministic_cox_regression_with_time_varying(self, simple_survival_data_timevarying):
        """CoxRegressionWithTimeVarying reproducibility with seed"""
        from bsix.models import CoxRegressionWithTimeVarying
       
        X, y = simple_survival_data_timevarying
       
        # Run 1
        model1 = CoxRegressionWithTimeVarying()
        model1.fit(X, y)
        pred1 = model1.predict(X)
       
        # Run 2
        model2 = CoxRegressionWithTimeVarying()
        model2.fit(X, y)
        pred2 = model2.predict(X)
       
        # Should be close
        assert np.allclose(pred1, pred2, atol=1e-3)

    @pytest.mark.integration
    def test_deterministic_deep_hit(self, simple_survival_data):
        """DeepHit reproducibility with seed"""
        from bsix.models import DeepHit
       
        X, y = simple_survival_data
       
        # Run 1
        model1 = DeepHit(X.shape[1], len(np.unique(y["event"])) - 1, 10, seed=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)
       
        # Run 2
        model2 = DeepHit(X.shape[1], len(np.unique(y["event"])) - 1, 10, seed=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)
       
        # Should be close
        assert np.allclose(pred1, pred2, atol=1e-3)

    @pytest.mark.integration
    def test_deterministic_deep_multi_task(self, simple_survival_data_multi_event):
        """DeepMultiTask reproducibility with seed"""
        from bsix.models import DeepMultiTask
       
        X, y = simple_survival_data_multi_event
        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y.ndim == 1 else y.shape[1])), 8).tolist()
       
        # Run 1
        model1 = DeepMultiTask(X.shape[1], coef_likelihood=coef_likelihood, seed=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)
       
        # Run 2
        model2 = DeepMultiTask(X.shape[1], coef_likelihood=coef_likelihood, seed=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)
       
        # Should be close
        assert np.allclose(pred1, pred2, atol=1e-3)

    @pytest.mark.integration
    def test_deterministic_deep_surv(self, simple_survival_data):
        """DeepSurv reproducibility with seed"""
        from bsix.models import DeepSurv
       
        X, y = simple_survival_data
       
        # Run 1
        model1 = DeepSurv(X.shape[1], seed=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)
       
        # Run 2
        model2 = DeepSurv(X.shape[1], seed=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)
       
        # Should be close
        assert np.allclose(pred1, pred2, atol=1e-3)

    @pytest.mark.integration
    def test_deterministic_deep_time_varying(self, simple_survival_data_timevarying):
        """DeepTimeVarying reproducibility with seed"""
        from bsix.models import DeepTimeVarying
       
        X, y = simple_survival_data_timevarying
       
        # Run 1
        model1 = DeepTimeVarying(X.shape[1], seed=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)
       
        # Run 2
        model2 = DeepTimeVarying(X.shape[1], seed=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)
       
        # Should be close
        assert np.allclose(pred1, pred2, atol=1e-3)

    @pytest.mark.integration
    def test_deterministic_random_surv_forest(self, simple_survival_data):
        """RandomSurvForest reproducibility with seed"""
        from bsix.models import RandomSurvForest
       
        X, y = simple_survival_data
       
        # Run 1
        model1 = RandomSurvForest(seed=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)
       
        # Run 2
        model2 = RandomSurvForest(seed=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)
       
        # Should be close
        assert np.allclose(pred1, pred2, atol=1e-3)

    @pytest.mark.integration
    def test_deterministic_survival_tab_pfn(self, simple_survival_data):
        """SurvivalTabPFN reproducibility with seed"""
        from bsix.models import SurvivalTabPFN
       
        X, y = simple_survival_data
       
        # Run 1
        model1 = SurvivalTabPFN(seed=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)
       
        # Run 2
        model2 = SurvivalTabPFN(seed=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)
       
        # Should be close
        assert np.allclose(pred1, pred2, atol=1e-3)

    @pytest.mark.integration
    def test_deterministic_surv_tree(self, simple_survival_data):
        """SurvTree reproducibility with seed"""
        from bsix.models import SurvTree
       
        X, y = simple_survival_data
       
        # Run 1
        model1 = SurvTree(seed=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)
       
        # Run 2
        model2 = SurvTree(seed=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)
       
        # Should be close
        assert np.allclose(pred1, pred2, atol=1e-3)

class TestFeatureImportance:
    """End-to-end tests for feature importance/SHAP"""
   
    @pytest.mark.integration
    def test_shap_explainability_accelerated_failure_time(self, dataframe_mixed_types):
        """Full pipeline with SHAP explanations"""
        from bsix.utils import get_data
        from bsix.models import AcceleratedFailureTime
       
        X_train, y_train, *_, train_idx, _, _, feature_names, scaler = get_data(df=dataframe_mixed_types, seed=42)
       
        model = AcceleratedFailureTime()
        model.fit(X_train, y_train)
       
        # Compute SHAP values
        model.calculate_xai(X_train, train_idx, scaler, "dataframe_mixed_types", seed=42, feature_names=feature_names)

    @pytest.mark.integration
    def test_shap_explainability_cox_regression(self, dataframe_mixed_types):
        """Full pipeline with SHAP explanations"""
        from bsix.utils import get_data
        from bsix.models import CoxRegression
       
        X_train, y_train, *_, train_idx, _, _, feature_names, scaler = get_data(df=dataframe_mixed_types, seed=42)
       
        model = CoxRegression()
        model.fit(X_train, y_train)
       
        # Compute SHAP values
        model.calculate_xai(X_train, train_idx, scaler, "dataframe_mixed_types", seed=42, feature_names=feature_names)

    @pytest.mark.integration
    def test_shap_explainability_cox_regression_with_time_varying(self, dataframe_time_varying):
        """Full pipeline with SHAP explanations"""
        from bsix.utils import get_data
        from bsix.models import CoxRegressionWithTimeVarying
       
        X_train, y_train, *_, train_idx, _, _, feature_names, scaler = get_data(df=dataframe_time_varying, seed=42)
       
        model = CoxRegressionWithTimeVarying()
        model.fit(X_train, y_train)
       
        # Compute SHAP values
        model.calculate_xai(X_train, train_idx, scaler, "dataframe_time_varying", seed=42, feature_names=feature_names)

    @pytest.mark.integration
    def test_shap_explainability_deep_hit(self, dataframe_mixed_types):
        """Full pipeline with SHAP explanations"""
        from bsix.utils import get_data
        from bsix.models import DeepHit
       
        X_train, y_train, *_, train_idx, _, _, feature_names, scaler = get_data(df=dataframe_mixed_types, seed=42)
       
        model = DeepHit(X_train.shape[1], len(np.unique(y_train["event"])) - 1, 10)
        model.fit(X_train, y_train)
       
        # Compute SHAP values
        model.calculate_xai(X_train, train_idx, scaler, "dataframe_mixed_types", seed=42, feature_names=feature_names)

    @pytest.mark.integration
    def test_shap_explainability_deep_multi_task(self, dataframe_multi_event):
        """Full pipeline with SHAP explanations"""
        from bsix.utils import get_data
        from bsix.models import DeepMultiTask
       
        X_train, y_train, *_, train_idx, _, _, feature_names, scaler = get_data(df=dataframe_multi_event, seed=42)

        rng = np.random.default_rng(42)
        coef_likelihood = np.round(rng.dirichlet(alpha=np.ones(1 if y_train.ndim == 1 else y_train.shape[1])), 8).tolist()
       
        model = DeepMultiTask(X_train.shape[1], coef_likelihood=coef_likelihood)
        model.fit(X_train, y_train)
       
        # Compute SHAP values
        model.calculate_xai(X_train, train_idx, scaler, "dataframe_multi_event", seed=42, feature_names=feature_names)

    @pytest.mark.integration
    def test_shap_explainability_deep_surv(self, dataframe_mixed_types):
        """Full pipeline with SHAP explanations"""
        from bsix.utils import get_data
        from bsix.models import DeepSurv
       
        X_train, y_train, *_, train_idx, _, _, feature_names, scaler = get_data(df=dataframe_mixed_types, seed=42)
       
        model = DeepSurv(X_train.shape[1])
        model.fit(X_train, y_train)
       
        # Compute SHAP values
        model.calculate_xai(X_train, train_idx, scaler, "dataframe_mixed_types", seed=42, feature_names=feature_names)

    @pytest.mark.integration
    def test_shap_explainability_deep_time_varying(self, dataframe_time_varying):
        """Full pipeline with SHAP explanations"""
        from bsix.utils import get_data
        from bsix.models import DeepTimeVarying
       
        X_train, y_train, *_, train_idx, _, _, feature_names, scaler = get_data(df=dataframe_time_varying, seed=42)
       
        model = DeepTimeVarying(X_train.shape[1])
        model.fit(X_train, y_train)
       
        # Compute SHAP values
        model.calculate_xai(X_train, train_idx, scaler, "dataframe_time_varying", seed=42, feature_names=feature_names)

    @pytest.mark.integration
    def test_shap_explainability_random_surv_forest(self, dataframe_mixed_types):
        """Full pipeline with SHAP explanations"""
        from bsix.utils import get_data
        from bsix.models import RandomSurvForest
       
        X_train, y_train, *_, train_idx, _, _, feature_names, scaler = get_data(df=dataframe_mixed_types, seed=42)
       
        model = RandomSurvForest(seed=42)
        model.fit(X_train, y_train)
       
        # Compute SHAP values
        model.calculate_xai(X_train, train_idx, scaler, "dataframe_mixed_types", seed=42, feature_names=feature_names)

    @pytest.mark.integration
    def test_shap_explainability_survival_tab_pfn(self, dataframe_mixed_types):
        """Full pipeline with SHAP explanations"""
        from bsix.utils import get_data
        from bsix.models import SurvivalTabPFN
       
        X_train, y_train, *_, train_idx, _, _, feature_names, scaler = get_data(df=dataframe_mixed_types, seed=42)
       
        model = SurvivalTabPFN()
        model.fit(X_train, y_train)
       
        # Compute SHAP values
        model.calculate_xai(X_train, train_idx, scaler, "dataframe_mixed_types", seed=42, feature_names=feature_names)

    @pytest.mark.integration
    def test_shap_explainability_surv_tree(self, dataframe_mixed_types):
        """Full pipeline with SHAP explanations"""
        from bsix.utils import get_data
        from bsix.models import SurvTree
       
        X_train, y_train, *_, train_idx, _, _, feature_names, scaler = get_data(df=dataframe_mixed_types, seed=42)
       
        model = SurvTree()
        model.fit(X_train, y_train)
       
        # Compute SHAP values
        model.calculate_xai(X_train, train_idx, scaler, "dataframe_mixed_types", seed=42, feature_names=feature_names)
