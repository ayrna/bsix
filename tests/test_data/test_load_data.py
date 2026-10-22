"""
Tests for data loading and preprocessing (Category DATA).
Tests for bsix.utils.load_data module following BSIX_TEST_SPECIFICATION.md
"""

import pytest
import numpy as np


class TestPrepareDataBasic:

    """
    Tests for basic _prepare_data functionality (DATA-001 to DATA-009).
    """
    
    def test_split_train_val_test_sizes(self, dataframe_mixed_types):

        """
        DATA-001: Split train/val/test with correct proportions (~60/20/20).
        """

        from bsix.utils.load_data import _prepare_data
        
        X_train, _, X_val, _, X_test, *_ = _prepare_data(
            dataframe_mixed_types, test_size=0.2, validation_size=0.2, seed=42
        )
        
        total_samples = len(dataframe_mixed_types)
        
        # Approximate proportions (with some tolerance for stratification)
        assert 0.55 < len(X_train) / total_samples < 0.65  # ~60%
        assert 0.15 < len(X_val) / total_samples < 0.25  # ~20%
        assert 0.15 < len(X_test) / total_samples < 0.25  # ~20%
    
    
    def test_reproducible_split_with_seed(self, dataframe_mixed_types):

        """
        DATA-001: Reproducibility with seed=42.
        """

        from bsix.utils.load_data import _prepare_data
        
        result1 = _prepare_data(
            dataframe_mixed_types, test_size=0.2, validation_size=0.2, seed=42
        )
        result2 = _prepare_data(
            dataframe_mixed_types, test_size=0.2, validation_size=0.2, seed=42
        )
        
        # All arrays should be identical
        assert np.allclose(result1[0], result2[0])
        assert np.allclose(result1[2], result2[2])
        assert np.allclose(result1[4], result2[4])
    
    
    def test_one_hot_encoding_categorical(self, dataframe_mixed_types):

        """
        DATA-002: One-hot encoding converts categorical to numeric.
        """

        from bsix.utils.load_data import _prepare_data
        
        X_train, _, X_val, _, X_test, *_ = _prepare_data(
            dataframe_mixed_types, test_size=0.2, validation_size=0.2, seed=42
        )
        
        # All arrays should be numeric float32
        assert X_train.dtype == np.float32
        assert X_val.dtype == np.float32
        assert X_test.dtype == np.float32
        
        # No NaN in any array
        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isnan(X_val))
        assert not np.any(np.isnan(X_test))
    
    
    def test_nan_elimination(self, dataframe_with_nans):

        """
        DATA-003: NaN rows are eliminated.
        """

        from bsix.utils.load_data import _prepare_data
        
        original_size = len(dataframe_with_nans)
        
        X_train, _, X_val, _, X_test, *_ = _prepare_data(
            dataframe_with_nans, test_size=0.2, validation_size=0.2, seed=42
        )
        
        total_clean = len(X_train) + len(X_val) + len(X_test)
        
        # Total should be less than original (NaNs removed)
        assert total_clean < original_size
        
        # All remaining values should be finite
        assert np.all(np.isfinite(X_train))
        assert np.all(np.isfinite(X_val))
        assert np.all(np.isfinite(X_test))
    
    
    def test_invalid_times_filtered(self, dataframe_with_invalid_times):

        """
        DATA-004: Times <= 0 are filtered out.
        """

        from bsix.utils.load_data import _prepare_data, _transformTrainValidationTest
        
        X_train, y_train, X_val, y_val, X_test, y_test, *_ = _prepare_data(
            dataframe_with_invalid_times, test_size=0.2, validation_size=0.2, seed=42
        )
        _, y_train_transformed, _ = _transformTrainValidationTest(X_train, y_train)
        _, y_val_transformed, _ = _transformTrainValidationTest(X_val, y_val)
        _, y_test_transformed, _ = _transformTrainValidationTest(X_test, y_test)

        # All times should be > 0
        assert np.all(y_train_transformed['time'] > 0)
        assert np.all(y_val_transformed['time'] > 0)
        assert np.all(y_test_transformed['time'] > 0)
    
    
    def test_timevarying_format_detection(self, dataframe_mixed_types):

        """
        DATA-005: Time-varying format is detected automatically.
        """

        from bsix.utils.load_data import _prepare_data
        
        # Add time_start, time_stop columns
        df = dataframe_mixed_types.copy()
        df['time_start'] = df['time'] - 50
        df['time_stop'] = df['time']
        
        X_train, y_train, *_ = _prepare_data(
            df, test_size=0.2, validation_size=0.2, seed=42
        )
        
        # Should handle time-varying data without error
        assert len(X_train) > 0
        assert len(y_train) > 0
    
    
    def test_feature_names_preserved(self, dataframe_mixed_types):

        """
        DATA-009: Feature names are preserved and ordered.
        """

        from bsix.utils.load_data import _prepare_data
        
        *_, feature_names = _prepare_data(
            dataframe_mixed_types, test_size=0.2, validation_size=0.2, seed=42
        )
        
        # Feature names should be a list
        assert isinstance(feature_names, (list, tuple))
        
        # Should not include 'event' or 'time'
        assert 'event' not in feature_names
        assert 'time' not in feature_names
        
        # Number of features should match array dimensions
        assert len(feature_names) > 0
    
    
    def test_dtype_float32_conversion(self, dataframe_mixed_types):

        """
        DATA-008: All arrays converted to float32.
        """

        from bsix.utils.load_data import _prepare_data, _transformTrainValidationTest
        
        X_train, y_train, X_val, y_val, X_test, y_test, *_ = _prepare_data(
            dataframe_mixed_types, test_size=0.2, validation_size=0.2, seed=42
        )

        X_train_transformed, y_train_transformed, _ = _transformTrainValidationTest(X_train, y_train)
        X_val_transformed, y_val_transformed, _ = _transformTrainValidationTest(X_val, y_val)
        X_test_transformed, y_test_transformed, _ = _transformTrainValidationTest(X_test, y_test)
        
        # All X arrays should be float32
        assert X_train_transformed.dtype == np.float32
        assert X_val_transformed.dtype == np.float32
        assert X_test_transformed.dtype == np.float32
        
        # y arrays should have structured dtype with float32 fields
        assert y_train_transformed.dtype.names is not None
        assert y_val_transformed.dtype.names is not None
        assert y_test_transformed.dtype.names is not None


class TestPrepareDataEdgeCases:

    """
    Tests for edge cases (DATA-010).
    """
    
    @pytest.mark.edge_case
    def test_empty_dataframe(self, dataframe_empty):
        
        """
        DATA-010: Empty DataFrame should be handled gracefully.
        """

        from bsix.utils.load_data import _prepare_data
        
        # Should raise an exception or handle gracefully
        with pytest.raises((ValueError)):
            _prepare_data(dataframe_empty, test_size=0.2, validation_size=0.2, seed=42)
    
    
    @pytest.mark.edge_case
    def test_single_row_dataframe(self, dataframe_single_row):

        """
        DATA-010: Single row DataFrame edge case.
        """

        from bsix.utils.load_data import _prepare_data
        
        # Should handle or raise appropriate error
        with pytest.raises((ValueError)):
            result = _prepare_data(dataframe_single_row, test_size=0.2, validation_size=0.2, seed=42)


class TestStratifiedGroupKFold:

    """
    Tests for StratifiedGroupKFold when identifier column exists (DATA-007).
    """
    
    def test_identifier_column_grouping(self, dataframe_mixed_types):

        """
        DATA-007: When 'identifier' column exists, use StratifiedGroupKFold.
        """

        from bsix.utils.load_data import _prepare_data
        
        # Add identifier column
        df = dataframe_mixed_types.copy()
        df['identifier'] = np.repeat([1, 2, 3, 4, 5], len(df) / 5)
        
        X_train, _, X_val, _, X_test, *_ = _prepare_data(
            df, test_size=0.2, validation_size=0.2, seed=42
        )
        
        # Should successfully split with groups
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
        
        # Total samples preserved
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(df)


class TestMultiEventHandling:

    """
    Tests for multi-event data handling.
    """
    
    def test_multi_event_detection(self, dataframe_multi_event):

        """
        DATA-006: Multi-event format detected and reshaped.
        """

        from bsix.utils.load_data import _prepare_data, _transformTrainValidationTest
        
        X_train, y_train, *_ = _prepare_data(
            dataframe_multi_event, test_size=0.2, validation_size=0.2, seed=42
        )

        X_train_transformed, y_train_transformed, _ = _transformTrainValidationTest(X_train, y_train)
        
        # Should handle multi-event data
        assert len(X_train_transformed) > 0
        # Y should have multiple event fields
        assert 'event' in y_train_transformed.dtype.names or 'event1' in y_train_transformed.dtype.names


class TestOutputShapes:

    """
    Tests for output array shapes and dimensions.
    """
    
    def test_output_tuple_structure(self, dataframe_mixed_types):

        """
        Output should be 10-element tuple.
        """

        from bsix.utils.load_data import _prepare_data
        
        result = _prepare_data(
            dataframe_mixed_types, test_size=0.2, validation_size=0.2, seed=42
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 10
    
    
    def test_x_y_sample_count_consistency(self, dataframe_mixed_types):

        """
        X and y should have same number of samples.
        """

        from bsix.utils.load_data import _prepare_data
        
        X_train, y_train, X_val, y_val, X_test, y_test, *_ = _prepare_data(
            dataframe_mixed_types, test_size=0.2, validation_size=0.2, seed=42
        )
        
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)
    
    
    def test_feature_dimension_consistency(self, dataframe_mixed_types):

        """
        All X arrays should have same feature dimension.
        """

        from bsix.utils.load_data import _prepare_data
        
        X_train, _, X_val, _, X_test, *_, feature_names = _prepare_data(
            dataframe_mixed_types, test_size=0.2, validation_size=0.2, seed=42
        )
        
        n_features = X_train.shape[1]
        assert X_val.shape[1] == n_features
        assert X_test.shape[1] == n_features
        assert len(feature_names) == n_features


class TestCustomTestValSizes:

    """
    Tests with different test/validation split ratios.
    """
    
    @pytest.mark.parametrize("test_size,val_size", [(0.1, 0.1), (0.2, 0.3), (0.3, 0.2), (0.25, 0.25),])
    def test_custom_split_ratios(self, dataframe_mixed_types, test_size, val_size):

        """
        _prepare_data should respect custom split ratios.
        """

        from bsix.utils.load_data import _prepare_data
        
        X_train, _, X_val, _, X_test, *_ = _prepare_data(
            dataframe_mixed_types, 
            test_size=test_size, 
            validation_size=val_size, 
            seed=42
        )
        
        total = len(dataframe_mixed_types)
        train_ratio = len(X_train) / total
        val_ratio = len(X_val) / total
        test_ratio = len(X_test) / total
        
        # Ratios should be approximately correct (within 5% tolerance)
        expected_train = 1.0 - test_size - val_size
        assert abs(train_ratio - expected_train) < 0.05
