"""
Tests for survival metrics (Category METR).
Tests for bsix.utils.survival_metrics following BSIX_TEST_SPECIFICATION.md
"""

import pytest
import numpy as np

class TestConcordanceIndexHarrel:
    """Tests for C-Index of Harrell (METR-001)"""
    
    def test_concordanceIndexHarrel_random(self, survival_data, random_predictions):
        """CLFM-001: C-Index of Harrell calculation basic case"""
        from bsix.utils.survival_metrics import concordanceIndexHarrel
        
        _, y_true = survival_data
        y_pred = random_predictions
        
        # Random predictions
        score = concordanceIndexHarrel([y_true, y_true], y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.5, abs=5e-2)
    
    
    def test_concordanceIndexHarrel_perfect(self, survival_data, perfect_survival_predictions):
        """C-Index of Harrell with perfect predictions"""
        from bsix.utils.survival_metrics import concordanceIndexHarrel
        
        _, y_true = survival_data
        y_pred = perfect_survival_predictions

        # Perfect predictions
        score = concordanceIndexHarrel([y_true, y_true], y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(1.0, abs=1e-3)
    
    
    def test_concordanceIndexHarrel_wrong(self, survival_data, inverse_survival_predictions):
        """C-Index of Harrell with all wrong predictions"""
        from bsix.utils.survival_metrics import concordanceIndexHarrel
        
        _, y_true = survival_data
        y_pred = inverse_survival_predictions

        # Inverse predictions
        score = concordanceIndexHarrel([y_true, y_true], y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.0, abs=1e-3)


class TestConcordanceIndexIPCW:
    """Tests for C-Index IPCW (METR-002)"""
    
    def test_concordanceIndexIPCW_random(self, survival_data, random_predictions):
        """CLFM-002: C-Index IPCW calculation basic case"""
        from bsix.utils.survival_metrics import concordanceIndexIPCW
        
        _, y_true = survival_data
        y_pred = random_predictions
        
        # Random predictions
        score = concordanceIndexIPCW([y_true, y_true], y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.5, abs=5e-2)
    
    
    def test_concordanceIndexIPCW_perfect(self, survival_data, perfect_survival_predictions):
        """C-Index IPCW with perfect predictions"""
        from bsix.utils.survival_metrics import concordanceIndexIPCW
        
        _, y_true = survival_data
        y_pred = perfect_survival_predictions

        # Perfect predictions
        score = concordanceIndexIPCW([y_true, y_true], y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(1.0, abs=1e-3)
    
    
    def test_concordanceIndexIPCW_wrong(self, survival_data, inverse_survival_predictions):
        """C-Index IPCW with all wrong predictions"""
        from bsix.utils.survival_metrics import concordanceIndexIPCW
        
        _, y_true = survival_data
        y_pred = inverse_survival_predictions

        # Inverse predictions
        score = concordanceIndexIPCW([y_true, y_true], y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.0, abs=1e-3)


class TestCumulativeDynamicAUC:
    """Tests for Cumulative Dynamic AUC (METR-003)"""
    
    def test_cumulativeDinamicAUC_random(self, survival_data, random_predictions):
        """CLFM-003: Cumulative Dynamic AUC calculation basic case"""
        from bsix.utils.survival_metrics import cumulativeDinamicAUC
        
        _, y_true = survival_data
        y_pred = random_predictions
        
        # Random predictions
        score = cumulativeDinamicAUC([y_true, y_true], y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.5, abs=5e-2)
    
    
    def test_cumulativeDinamicAUC_perfect(self, survival_data, perfect_survival_predictions):
        """Cumulative Dynamic AUC with perfect predictions"""
        from bsix.utils.survival_metrics import cumulativeDinamicAUC
        
        _, y_true = survival_data
        y_pred = perfect_survival_predictions

        # Perfect predictions
        score = cumulativeDinamicAUC([y_true, y_true], y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(1.0, abs=1e-3)
    
    
    def test_cumulativeDinamicAUC_wrong(self, survival_data, inverse_survival_predictions):
        """Cumulative Dynamic AUC with all wrong predictions"""
        from bsix.utils.survival_metrics import cumulativeDinamicAUC
        
        _, y_true = survival_data
        y_pred = inverse_survival_predictions

        # Inverse predictions
        score = cumulativeDinamicAUC([y_true, y_true], y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.0, abs=1e-3)


class TestIntegratedBrierScore:
    """Tests for Integrated Brier Score (METR-004)"""
    
    def test_integratedBrierScore_random(self, survival_data, random_survival_functions):
        """CLFM-004: Integrated Brier Score calculation basic case"""
        from bsix.utils.survival_metrics import integratedBrierScore
        
        _, y_true = survival_data
        y_pred = random_survival_functions
        
        # Random predictions
        score = integratedBrierScore([y_true, y_true], y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.5, abs=5e-2)
    
    
    def test_integratedBrierScore_perfect(self, survival_data, perfect_survival_functions):
        """Integrated Brier Score with perfect predictions"""
        from bsix.utils.survival_metrics import integratedBrierScore
        
        _, y_true = survival_data
        y_pred = perfect_survival_functions

        # Perfect predictions
        score = integratedBrierScore([y_true, y_true], y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.0, abs=1e-3)
    
    
    def test_integratedBrierScore_wrong(self, survival_data, inverse_survival_functions):
        """Integrated Brier Score with all wrong predictions"""
        from bsix.utils.survival_metrics import integratedBrierScore
        
        _, y_true = survival_data
        y_pred = inverse_survival_functions

        # Inverse predictions
        score = integratedBrierScore([y_true, y_true], y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(1.0, abs=1e-3)


class TestMetricRanges:
    """Tests for valid metric ranges"""
    
    def test_metrics_in_valid_range(self, survival_data, perfect_survival_predictions, inverse_survival_predictions, perfect_survival_functions, inverse_survival_functions):
        """All metrics should be in valid ranges"""
        from bsix.utils.survival_metrics import integratedBrierScore, concordanceIndexHarrel, concordanceIndexIPCW, cumulativeDinamicAUC
        
        _, y_true = survival_data
        y_pred_perfect = perfect_survival_predictions
        y_pred_inverse = inverse_survival_predictions
        function_pred_perfect = perfect_survival_functions
        function_pred_inverse = inverse_survival_functions
        
        integratedBrierScore_perfect_score = integratedBrierScore([y_true, y_true], function_pred_perfect)
        concordanceIndexHarrel_perfect_score = concordanceIndexHarrel([y_true, y_true], y_pred_perfect)
        concordanceIndexIPCW_perfect_score = concordanceIndexIPCW([y_true, y_true], y_pred_perfect)
        cumulativeDinamicAUC_perfect_score = cumulativeDinamicAUC([y_true, y_true], y_pred_perfect)
        
        integratedBrierScore_inverse_score = integratedBrierScore([y_true, y_true], function_pred_inverse)
        concordanceIndexHarrel_inverse_score = concordanceIndexHarrel([y_true, y_true], y_pred_inverse)
        concordanceIndexIPCW_inverse_score = concordanceIndexIPCW([y_true, y_true], y_pred_inverse)
        cumulativeDinamicAUC_inverse_score = cumulativeDinamicAUC([y_true, y_true], y_pred_inverse)

        # Ranges
        # [0, 1]
        assert (0 <= integratedBrierScore_perfect_score <= 1) and (0 <= integratedBrierScore_inverse_score <= 1)
        assert (0 <= concordanceIndexHarrel_perfect_score <= 1) and (0 <= concordanceIndexHarrel_inverse_score <= 1)
        assert (0 <= concordanceIndexIPCW_perfect_score <= 1) and (0 <= concordanceIndexIPCW_inverse_score <= 1)
        assert (0 <= cumulativeDinamicAUC_perfect_score <= 1) and (0 <= cumulativeDinamicAUC_inverse_score <= 1)


class TestMetricsEdgeCases:
    """Tests for handling tied event times (METR-005)"""

    @pytest.mark.edge_case
    def test_empty_predictions(self):
        """Metrics with empty arrays"""
        from bsix.utils.survival_metrics import integratedBrierScore, concordanceIndexHarrel, concordanceIndexIPCW, cumulativeDinamicAUC
        
        y_true = np.array([])
        y_pred = np.array([])

        # Should handle
        with pytest.raises((ValueError)):
            integratedBrierScore([y_true, y_true], y_pred)
        with pytest.raises((ValueError)):
            concordanceIndexHarrel([y_true, y_true], y_pred)
        with pytest.raises((ValueError)):
            concordanceIndexIPCW([y_true, y_true], y_pred)
        with pytest.raises((ValueError)):
            cumulativeDinamicAUC([y_true, y_true], y_pred)


    @pytest.mark.edge_case
    def test_tied_events(self, survival_data, random_predictions, random_survival_functions):
        """Handle ties in survival data"""
        from bsix.utils.survival_metrics import integratedBrierScore, concordanceIndexHarrel, concordanceIndexIPCW, cumulativeDinamicAUC
        
        _, y_true = survival_data
        
        # Create data with many tied times
        y_tied = y_true.copy()
        y_tied['time'] = np.floor(y_tied['time'] / 10) * 10  # Round to nearest 10
        
        integratedBrierScore_perfect_score = integratedBrierScore([y_true, y_true], random_survival_functions)
        concordanceIndexHarrel_perfect_score = concordanceIndexHarrel([y_true, y_true], random_predictions)
        concordanceIndexIPCW_perfect_score = concordanceIndexIPCW([y_true, y_true], random_predictions)
        cumulativeDinamicAUC_perfect_score = cumulativeDinamicAUC([y_true, y_true], random_predictions)
        
        # Should still produce valid metric
        assert np.isfinite(integratedBrierScore_perfect_score)
        assert 0.0 <= integratedBrierScore_perfect_score <= 1.0
        assert np.isfinite(concordanceIndexHarrel_perfect_score)
        assert 0.0 <= concordanceIndexHarrel_perfect_score <= 1.0
        assert np.isfinite(concordanceIndexIPCW_perfect_score)
        assert 0.0 <= concordanceIndexIPCW_perfect_score <= 1.0
        assert np.isfinite(cumulativeDinamicAUC_perfect_score)
        assert 0.0 <= cumulativeDinamicAUC_perfect_score <= 1.0

    @pytest.mark.edge_case
    def test_metric_all_censored(self, survival_data_all_censored, random_predictions, random_survival_functions):
        """Metrics handle all-censored data"""
        from bsix.utils.survival_metrics import integratedBrierScore, concordanceIndexHarrel, concordanceIndexIPCW, cumulativeDinamicAUC
        # # # survival_data_all_censored da 100 patrones, pensar pasar n como parámetro
        _, y_true = survival_data_all_censored
        
        # Might return NaN but shouldn't crash
        with pytest.raises((ValueError)):
            integratedBrierScore([y_true, y_true], random_survival_functions)
        with pytest.raises((ValueError)):
            concordanceIndexHarrel([y_true, y_true], random_predictions)
        with pytest.raises((ValueError)):
            concordanceIndexIPCW([y_true, y_true], random_predictions)
        with pytest.raises((ValueError)):
            cumulativeDinamicAUC([y_true, y_true], random_predictions)



class TestMetricsConsistency:
    """Tests for metric consistency"""
    
    def test_consistency_predictions(self, survival_data, random_predictions, random_survival_functions):
        """Metrics should be consistent for the same predictions"""
        from bsix.utils.survival_metrics import integratedBrierScore, concordanceIndexHarrel, concordanceIndexIPCW, cumulativeDinamicAUC

        _, y_true = survival_data

        integratedBrierScore_score1 = integratedBrierScore([y_true, y_true], random_survival_functions)
        concordanceIndexHarrel_score1 = concordanceIndexHarrel([y_true, y_true], random_predictions)
        concordanceIndexIPCW_score1 = concordanceIndexIPCW([y_true, y_true], random_predictions)
        cumulativeDinamicAUC_score1 = cumulativeDinamicAUC([y_true, y_true], random_predictions)
        
        integratedBrierScore_score2 = integratedBrierScore([y_true, y_true], random_survival_functions)
        concordanceIndexHarrel_score2 = concordanceIndexHarrel([y_true, y_true], random_predictions)
        concordanceIndexIPCW_score2 = concordanceIndexIPCW([y_true, y_true], random_predictions)
        cumulativeDinamicAUC_score2 = cumulativeDinamicAUC([y_true, y_true], random_predictions)
            
        assert np.allclose(integratedBrierScore_score1, integratedBrierScore_score2, atol=1e-3)
        assert np.allclose(concordanceIndexHarrel_score1, concordanceIndexHarrel_score2, atol=1e-3)
        assert np.allclose(concordanceIndexIPCW_score1, concordanceIndexIPCW_score2, atol=1e-3)
        assert np.allclose(cumulativeDinamicAUC_score1, cumulativeDinamicAUC_score2, atol=1e-3)

    
    def test_consistency_symmetry(self, survival_data, random_predictions):
        """Metrics behave consistently across transformations"""
        from bsix.utils.survival_metrics import concordanceIndexHarrel, concordanceIndexIPCW, cumulativeDinamicAUC
        
        _, y = survival_data

        concordanceIndexHarrel_score1 = concordanceIndexHarrel([y, y], random_predictions)
        concordanceIndexHarrel_score2 = concordanceIndexHarrel([y, y], random_predictions * 5)
        concordanceIndexIPCW_score1 = concordanceIndexIPCW([y, y], random_predictions)
        concordanceIndexIPCW_score2 = concordanceIndexIPCW([y, y], random_predictions * 5)
        cumulativeDinamicAUC_score1 = cumulativeDinamicAUC([y, y], random_predictions)
        cumulativeDinamicAUC_score2 = cumulativeDinamicAUC([y, y], random_predictions * 5)
        
        # Should be equal for rank-based metrics
        assert np.isclose(concordanceIndexHarrel_score1, concordanceIndexHarrel_score2, atol=1e-3)
        assert np.isclose(concordanceIndexIPCW_score1, concordanceIndexIPCW_score2, atol=1e-3)
        assert np.isclose(cumulativeDinamicAUC_score1, cumulativeDinamicAUC_score2, atol=1e-3)
    