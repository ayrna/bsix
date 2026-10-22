"""
Tests for survival utilities (Category: DATA utilities).
Tests for bsix.utils.survival_utils functions.
"""

import pytest
import numpy as np


class TestGetTau:

    """
    Tests for getTau function (METR-008).
    """
    
    def test_get_tau_basic(self, survival_data):

        """
        METR-008: getTau returns valid maximum time.
        """

        from bsix.utils.survival_utils import getTau
        
        X, y = survival_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        # Mock predictions
        predictions = np.random.randn(len(X_test))
        
        tau = getTau(y_train, y_test)
        
        # tau should be a reasonable time value
        assert isinstance(tau, (float, np.floating))
        assert tau > 0 and tau <= max(y_test['time'].max(), y_train['time'].max())


class TestGetTimes:

    """
    Tests for getTimes function (METR-009).
    """
    
    def test_get_times_grid(self, survival_data):

        """
        METR-009: getTimes returns equidistant points.
        """

        from bsix.utils.survival_utils import getTimes
        
        X, y = survival_data
        
        num_division = 5
        times = getTimes(y, num_division=num_division)
        
        # Should return equidistant points
        assert isinstance(times, np.ndarray)
        assert len(times) == num_division
        
        # Times should be within data range
        assert times[0] >= y['time'].min()
        assert times[-1] <= y['time'].max()
    
    
    def test_get_times_sorted(self, survival_data):

        """
        getTimes should return sorted array.
        """

        from bsix.utils.survival_utils import getTimes
        
        X, y = survival_data
        times = getTimes(y, num_division=10)
        
        # Should be sorted
        assert np.all(times[:-1] <= times[1:])
    
    
    @pytest.mark.parametrize("num_division", [2, 5, 10, 20, 50])
    def test_get_times_different_divisions(self, survival_data, num_division):

        """
        getTimes works with different division counts.
        """

        from bsix.utils.survival_utils import getTimes
        
        X, y = survival_data
        times = getTimes(y, num_division=num_division)
        
        # Should return correct number of points
        assert len(times) == num_division


class TestSurvivalUtilsRobustness:
    
    """
    Tests for robustness and edge cases.
    """
    
    def test_get_tau_with_all_censored(self, survival_data_all_censored):

        """
        getTau handles all-censored data gracefully.
        """

        from bsix.utils.survival_utils import getTau
        
        X, y = survival_data_all_censored
        X_train, X_test = X[:40], X[40:]
        y_train, y_test = y[:40], y[40:]
        predictions = np.random.randn(len(X_test))
        
        # Should not crash
        tau = getTau(y_train, y_test)
        assert tau > 0
    
    
    def test_get_times_single_sample(self):

        """
        getTimes handles single sample edge case.
        """

        from bsix.utils.survival_utils import getTimes
        
        y = np.zeros(1, dtype=[('event', bool), ('time', np.float32)])
        y['time'] = 100.0
        
        # Should handle gracefully
        times = getTimes(y, num_division=5)
        assert len(times) >= 1
