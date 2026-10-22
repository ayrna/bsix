"""
Tests for metric computation and aggregation (Category CALC).
Tests for bsix.utils.calculate_metrics following BSIX_TEST_SPECIFICATION.md
"""

import pytest
import numpy as np

class TestMetricsDict:
    """Tests for metrics dictionary structure (CALC-005 to CALC-008)"""
    
    def test_format_predictions_array(self, risk_scores):
        """CALC-005: format_predictions normalizes numpy array"""
        from bsix.utils import format_predictions
        
        pred = risk_scores
        formatted = format_predictions(pred)
        
        # Should return dict
        assert isinstance(formatted, dict)
        assert 'survival' in formatted
    
    
    def test_format_predictions_tuple(self, risk_scores, binary_predictions):
        """CALC-006: format_predictions handles (survival, binary) tuples"""
        from bsix.utils import format_predictions
        
        pred_survival = risk_scores
        pred_binary = binary_predictions
        predictions = (pred_survival, pred_binary)
        
        formatted = format_predictions(predictions)
        
        # Should separate into survival and binary
        assert isinstance(formatted, dict)
        assert 'survival' in formatted
        assert 'binary' in formatted


class TestComputeMetricsBasic:
    """Tests for compute_metrics function (CALC-001 to CALC-003)"""
    
    def test_compute_metrics_without_survival_function(self, survival_data, risk_scores):
        """CALC-001: compute_metrics without S(t)"""
        from bsix.utils import compute_metrics
        
        X, y = survival_data
        pred = risk_scores
        
        metrics = compute_metrics(
            train_targets=y,
            evaluation_targets=y,
            predictions=pred,
            survival_function=None
        )
        
        # Should return dict with multiple metrics
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
    
    
    def test_compute_metrics_with_survival_function(self, survival_data, risk_scores, survival_functions):
        """CALC-002: compute_metrics with S(t) should include Brier Score"""
        from bsix.utils import compute_metrics
        
        X, y = survival_data
        pred = risk_scores
        
        metrics = compute_metrics(
            train_targets=y,
            evaluation_targets=y,
            predictions=pred,
            survival_function=survival_functions
        )
        
        # Should include Brier Score
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        assert 'Brier Score' in metrics
    
    
    def test_compute_all_metrics(self, survival_data, risk_scores, binary_predictions, survival_functions):
        """CALC-003: test_compute_all_metrics returns all metrics"""
        from bsix.utils import compute_metrics
        
        X, y = survival_data
        pred_survival = risk_scores
        pred_binary = binary_predictions

        pred = (pred_survival, pred_binary)

        metrics = compute_metrics(
            train_targets=y,
            evaluation_targets=y,
            predictions=pred,
            survival_function=survival_functions
        )
        
        # Should contain survival-specific metrics
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        assert 'C-Index Harrel' in metrics
        assert 'C-Index IPCW' in metrics
        assert 'Cumulative Dinamic AUC' in metrics
        assert 'Brier Score' in metrics
        assert 'MAE' in metrics
        assert 'AMAE' in metrics
        assert 'MS' in metrics
        assert 'CCR' in metrics
        assert 'RECALL0' in metrics
        assert 'RECALL1' in metrics


class TestMultiEventMetrics:
    """Tests for multi-event metric handling (CALC-007)"""

    def test_multi_event_metrics_prefixes(self, survival_data_multi_event, risk_scores, binary_predictions, survival_functions):
        """CALC-007: Multi-event metrics have event prefixes (P0, P1, etc)"""
        from bsix.utils import compute_metrics
        
        X, y = survival_data_multi_event
        pred_survival = np.array([risk_scores, risk_scores]).T # (n, multi_event)
        pred_binary = np.array([binary_predictions, binary_predictions]).T # (n, multi_event)
        survival_functions = np.array([survival_functions, survival_functions])

        pred = (pred_survival, pred_binary)
        
        metrics = compute_metrics(
            train_targets=y,
            evaluation_targets=y,
            predictions=pred,
            survival_function=survival_functions
        )
        
        # Should contain survival-specific metrics
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        for k in range(2):
            assert f'P{k}  C-Index Harrel' in metrics
            assert f'P{k}  C-Index IPCW' in metrics
            assert f'P{k}  Cumulative Dinamic AUC' in metrics
            assert f'P{k}  Brier Score' in metrics
            assert f'P{k}  MAE' in metrics
            assert f'P{k}  AMAE' in metrics
            assert f'P{k}  MS' in metrics
            assert f'P{k}  CCR' in metrics
            assert f'P{k}  RECALL0' in metrics
            assert f'P{k}  RECALL1' in metrics


class TestMetricsFinite:
    """Tests that metrics are finite numbers"""
    
    def test_all_metrics_finite(self, survival_data, risk_scores, binary_predictions, survival_functions):
        """All computed metrics should be finite"""
        from bsix.utils import compute_metrics
        
        X, y = survival_data
        pred_survival = risk_scores
        pred_binary = binary_predictions

        pred = (pred_survival, pred_binary)

        metrics = compute_metrics(
            train_targets=y,
            evaluation_targets=y,
            predictions=pred,
            survival_function=survival_functions
        )
        
        # All values should be finite
        for key, value in metrics.items():
            assert np.isfinite(value)


class TestMetricsEdgeCases:
    """Tests for edge cases in metric computation"""

    @pytest.mark.edge_case
    def test_metric_division_by_zero_handling(self):
        #ERR-008: Metrics handle division by zero gracefully
        from bsix.utils import compute_metrics
        
        with pytest.raises((ValueError)):
            metrics = compute_metrics(
                train_targets=np.array([], dtype=np.float32),
                evaluation_targets=np.array([], dtype=np.float32),
                predictions=np.array([], dtype=np.float32),
                survival_function=None
            )

    @pytest.mark.edge_case
    def test_metrics_single_test_sample(self, single_survival_data, single_risk_score, single_binary_prediction, single_survival_function):
        """compute_metrics with single test sample"""
        from bsix.utils import compute_metrics
        
        # Single test sample
        X, y = single_survival_data
        pred_survival = single_risk_score
        pred_binary = single_binary_prediction
        
        pred = (pred_survival, pred_binary)
        with pytest.raises((ValueError)):
            metrics = compute_metrics(
                train_targets=y,
                evaluation_targets=y,
                predictions=pred,
                survival_function=single_survival_function
            )
    
    
    @pytest.mark.edge_case
    def test_metrics_all_censored_test(self, survival_data, risk_scores, binary_predictions, survival_functions):
        """compute_metrics with all-censored test set"""
        from bsix.utils import compute_metrics
        
        X, y = survival_data
        pred_survival = risk_scores
        pred_binary = binary_predictions
        
        # Make all events censored
        y['event'] = False
        
        pred = (pred_survival, pred_binary)

        with pytest.raises((ValueError)):
            metrics = compute_metrics(
                train_targets=y,
                evaluation_targets=y,
                predictions=pred,
                survival_function=survival_functions
            )


class TestNormalizedMetrics:
    """Tests for metric normalization and scaling"""
    
    def test_metrics_scale_consistency(self, survival_data, risk_scores, binary_predictions, survival_functions):
        """Metrics should be consistent across input scales"""
        from bsix.utils import compute_metrics
        
        X, y = survival_data
        pred1_survival = risk_scores

        # Scaled predictions
        pred2_survival = pred1_survival * 5
        
        metrics1 = compute_metrics(
            train_targets=y,
            evaluation_targets=y,
            predictions=pred1_survival
        )
        
        metrics2 = compute_metrics(
            train_targets=y,
            evaluation_targets=y,
            predictions=pred2_survival
        )

        names = ['C-Index Harrel', 'C-Index IPCW', 'Brier Score', 'Cumulative Dinamic', 'AUC', 'MAE', 'AMAE', 'CCR', 'MS', 'RECALL0', 'RECALL1']
        
        assert isinstance(metrics1, dict)
        assert isinstance(metrics2, dict)
        assert set(metrics1.keys()) == set(metrics2.keys())

        for key in metrics1.keys():
            if any(base in key for base in names):
                assert metrics1[key] == pytest.approx(metrics2[key], abs=1e-3), f"Metric failed: {key}"
