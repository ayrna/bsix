"""
Tests for binary classification metrics (Category CLFM).
Tests for bsix.utils.classification_metrics following BSIX_TEST_SPECIFICATION.md
"""

import pytest
import numpy as np


class TestCCR:
    """Tests for CCR (score) metric (CLFM-001)"""
    
    def test_ccr_random(self, survival_data, random_predictions):
        """CLFM-001: CCR calculation basic case"""
        from bsix.utils.classification_metrics import ccr
        
        _, y_true = survival_data
        y_pred = random_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
        
        # Random predictions
        score = ccr(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.5, abs=5e-2)
    
    
    def test_ccr_perfect(self, survival_data, perfect_classification_predictions):
        """CCR with perfect predictions"""
        from bsix.utils.classification_metrics import ccr
        
        _, y_true = survival_data
        y_pred = perfect_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Perfect predictions
        score = ccr(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(1.0, abs=1e-3)
    
    
    def test_ccr_wrong(self, survival_data, inverse_classification_predictions):
        """CCR with all wrong predictions"""
        from bsix.utils.classification_metrics import ccr
        
        _, y_true = survival_data
        y_pred = inverse_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Inverse predictions
        score = ccr(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.0, abs=1e-3)


class TestAMAE:
    """Tests for AMAE metric (CLFM-002)"""
    
    def test_amae_random(self, survival_data, random_predictions):
        """CLFM-002: AMAE calculation basic case"""
        from bsix.utils.classification_metrics import amae
        
        _, y_true = survival_data
        y_pred = random_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
        
        # Random predictions
        error = amae(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(0.5, abs=5e-2)
    
    
    def test_amae_perfect(self, survival_data, perfect_classification_predictions):
        """AMAE with perfect predictions"""
        from bsix.utils.classification_metrics import amae
        
        _, y_true = survival_data
        y_pred = perfect_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Perfect predictions
        error = amae(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(0.0, abs=1e-3)
    
    
    def test_amae_wrong(self, survival_data, inverse_classification_predictions):
        """AMAE with all wrong predictions"""
        from bsix.utils.classification_metrics import amae
        
        _, y_true = survival_data
        y_pred = inverse_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Inverse predictions
        error = amae(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(1.0, abs=1e-3)


class TestGM:
    """Tests for GM metric (CLFM-003)"""
    
    def test_gm_random(self, survival_data, random_predictions):
        """CLFM-003: GM calculation basic case"""
        from bsix.utils.classification_metrics import gm
        
        _, y_true = survival_data
        y_pred = random_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
        
        # Random predictions
        error = gm(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(0.5, abs=5e-2)
    
    
    def test_gm_perfect(self, survival_data, perfect_classification_predictions):
        """GM with perfect predictions"""
        from bsix.utils.classification_metrics import gm
        
        _, y_true = survival_data
        y_pred = perfect_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Perfect predictions
        error = gm(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(1.0, abs=1e-3)
    
    
    def test_gm_wrong(self, survival_data, inverse_classification_predictions):
        """GM with all wrong predictions"""
        from bsix.utils.classification_metrics import gm
        
        _, y_true = survival_data
        y_pred = inverse_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Inverse predictions
        error = gm(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(0.0, abs=1e-3)


class TestMAE:
    """Tests for MAE metric (CLFM-004)"""
    
    def test_mae_random(self, survival_data, random_predictions):
        """CLFM-004: MAE calculation basic case"""
        from bsix.utils.classification_metrics import mae
        
        _, y_true = survival_data
        y_pred = random_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
        
        # Random predictions
        error = mae(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(0.5, abs=5e-2)
    
    
    def test_mae_perfect(self, survival_data, perfect_classification_predictions):
        """MAE with perfect predictions"""
        from bsix.utils.classification_metrics import mae
        
        _, y_true = survival_data
        y_pred = perfect_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Perfect predictions
        error = mae(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(0.0, abs=1e-3)
    
    
    def test_mae_wrong(self, survival_data, inverse_classification_predictions):
        """MAE with all wrong predictions"""
        from bsix.utils.classification_metrics import mae
        
        _, y_true = survival_data
        y_pred = inverse_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Inverse predictions
        error = mae(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(1.0, abs=1e-3)


class TestMMAE:
    """Tests for MMAE metric (CLFM-005)"""
    
    def test_mmae_random(self, survival_data, random_predictions):
        """CLFM-005: MMAE calculation basic case"""
        from bsix.utils.classification_metrics import mmae
        
        _, y_true = survival_data
        y_pred = random_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
        
        # Random predictions
        error = mmae(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(0.5, abs=5e-2)
    
    
    def test_mmae_perfect(self, survival_data, perfect_classification_predictions):
        """MMAE with perfect predictions"""
        from bsix.utils.classification_metrics import mmae
        
        _, y_true = survival_data
        y_pred = perfect_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Perfect predictions
        error = mmae(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(0.0, abs=1e-3)
    
    
    def test_mmae_wrong(self, survival_data, inverse_classification_predictions):
        """MMAE with all wrong predictions"""
        from bsix.utils.classification_metrics import mmae
        
        _, y_true = survival_data
        y_pred = inverse_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Inverse predictions
        error = mmae(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(1.0, abs=1e-3)


class TestRECALL:
    """Tests for recall/sensitivity metric (CLFM-006)"""
    
    def test_recall_random(self, survival_data, random_predictions):
        """CLFM-006: RECALL calculation"""
        from bsix.utils.classification_metrics import recall
        
        _, y_true = survival_data
        y_pred = random_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
        
        # Random predictions
        score = recall(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.5, abs=5e-2)
    
    
    def test_recall_perfect(self, survival_data, perfect_classification_predictions):
        """RECALL with perfect predictions"""
        from bsix.utils.classification_metrics import recall
        
        _, y_true = survival_data
        y_pred = perfect_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Perfect predictions
        score = recall(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(1.0, abs=1e-3)
    
    
    def test_recall_wrong(self, survival_data, inverse_classification_predictions):
        """RECALL with all wrong predictions"""
        from bsix.utils.classification_metrics import recall
        
        _, y_true = survival_data
        y_pred = inverse_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Inverse predictions
        score = recall(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.0, abs=1e-3)


class TestMS:
    """Tests for MS metric (CLFM-007)"""
    
    def test_ms_random(self, survival_data, random_predictions):
        """CLFM-007: MS calculation"""
        from bsix.utils.classification_metrics import ms
        
        _, y_true = survival_data
        y_pred = random_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
        
        # Random predictions
        score = ms(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.5, abs=5e-2)
    
    
    def test_ms_perfect(self, survival_data, perfect_classification_predictions):
        """MS with perfect predictions"""
        from bsix.utils.classification_metrics import ms
        
        _, y_true = survival_data
        y_pred = perfect_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Perfect predictions
        score = ms(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(1.0, abs=1e-3)
    
    
    def test_ms_wrong(self, survival_data, inverse_classification_predictions):
        """MS with all wrong predictions"""
        from bsix.utils.classification_metrics import ms
        
        _, y_true = survival_data
        y_pred = inverse_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Inverse predictions
        score = ms(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.0, abs=1e-3)


class TestMZE:
    """Tests for MZE metric (CLFM-008)"""
    
    def test_mze_random(self, survival_data, random_predictions):
        """CLFM-008: MZE calculation basic case"""
        from bsix.utils.classification_metrics import mze
        
        _, y_true = survival_data
        y_pred = random_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
        
        # Random predictions
        error = mze(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(0.5, abs=5e-2)
    
    
    def test_mze_perfect(self, survival_data, perfect_classification_predictions):
        """MZE with perfect predictions"""
        from bsix.utils.classification_metrics import mze
        
        _, y_true = survival_data
        y_pred = perfect_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Perfect predictions
        error = mze(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(0.0, abs=1e-3)
    
    
    def test_mze_wrong(self, survival_data, inverse_classification_predictions):
        """MZE with all wrong predictions"""
        from bsix.utils.classification_metrics import mze
        
        _, y_true = survival_data
        y_pred = inverse_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Inverse predictions
        error = mze(y_true, y_pred)
        
        assert isinstance(error, (float, np.floating))
        assert error == pytest.approx(1.0, abs=1e-3)


class TestTKENDALL:
    """Tests for TKENDALL metric (CLFM-009)"""
    
    def test_tkendall_random(self, survival_data, random_predictions):
        """CLFM-009: TKENDALL calculation"""
        from bsix.utils.classification_metrics import tkendall
        
        _, y_true = survival_data
        y_pred = random_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
        
        # Random predictions
        score = tkendall(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.0, abs=5e-2)
    
    
    def test_tkendall_perfect(self, survival_data, perfect_classification_predictions):
        """TKENDALL with perfect predictions"""
        from bsix.utils.classification_metrics import tkendall
        
        _, y_true = survival_data
        y_pred = perfect_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Perfect predictions
        score = tkendall(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(1.0, abs=1e-3)
    
    
    def test_tkendall_wrong(self, survival_data, inverse_classification_predictions):
        """TKENDALL with all wrong predictions"""
        from bsix.utils.classification_metrics import tkendall
        
        _, y_true = survival_data
        y_pred = inverse_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Inverse predictions
        score = tkendall(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(-1.0, abs=1e-3)


class TestWKAPPA:
    """Tests for WKAPPA metric (CLFM-010)"""
    
    def test_wkappa_random(self, survival_data, random_predictions):
        """CLFM-010: WKAPPA calculation"""
        from bsix.utils.classification_metrics import wkappa
        
        _, y_true = survival_data
        y_pred = random_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
        
        # Random predictions
        score = wkappa(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.0, abs=5e-2)
    
    
    def test_wkappa_perfect(self, survival_data, perfect_classification_predictions):
        """WKAPPA with perfect predictions"""
        from bsix.utils.classification_metrics import wkappa
        
        _, y_true = survival_data
        y_pred = perfect_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Perfect predictions
        score = wkappa(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(1.0, abs=1e-3)
    
    
    def test_wkappa_wrong(self, survival_data, inverse_classification_predictions):
        """WKAPPA with all wrong predictions"""
        from bsix.utils.classification_metrics import wkappa
        
        _, y_true = survival_data
        y_pred = inverse_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Inverse predictions
        score = wkappa(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(-1.0, abs=1e-3)


class TestSPEARMAN:
    """Tests for SPEARMAN metric (CLFM-010)"""
    
    def test_spearman_random(self, survival_data, random_predictions):
        """CLFM-010: SPEARMAN calculation"""
        from bsix.utils.classification_metrics import spearman
        
        _, y_true = survival_data
        y_pred = random_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)
        
        # Random predictions
        score = spearman(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(0.0, abs=5e-2)
    
    
    def test_spearman_perfect(self, survival_data, perfect_classification_predictions):
        """SPEARMAN with perfect predictions"""
        from bsix.utils.classification_metrics import spearman
        
        _, y_true = survival_data
        y_pred = perfect_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Perfect predictions
        score = spearman(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(1.0, abs=1e-3)
    
    
    def test_spearman_wrong(self, survival_data, inverse_classification_predictions):
        """SPEARMAN with all wrong predictions"""
        from bsix.utils.classification_metrics import spearman
        
        _, y_true = survival_data
        y_pred = inverse_classification_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        # Inverse predictions
        score = spearman(y_true, y_pred)
        
        assert isinstance(score, (float, np.floating))
        assert score == pytest.approx(-1.0, abs=1e-3)


class TestMetricRanges:
    """Tests for valid metric ranges"""
    
    def test_metrics_in_valid_range(self, survival_data, perfect_classification_predictions, inverse_classification_predictions):
        """All metrics should be in valid ranges"""
        from bsix.utils.classification_metrics import ccr, amae, gm, mae, mmae, recall, ms, mze, tkendall, wkappa, spearman
        
        _, y_true = survival_data
        y_pred_perfect = perfect_classification_predictions
        y_pred_inverse = inverse_classification_predictions

        y_true = y_true["event"]
        
        ccr_perfect_score = ccr(y_true, y_pred_perfect)
        amae_perfect_score = amae(y_true, y_pred_perfect)
        gm_perfect_score = gm(y_true, y_pred_perfect)
        mae_perfect_score = mae(y_true, y_pred_perfect)
        mmae_perfect_score = mmae(y_true, y_pred_perfect)
        recall_perfect_score = recall(y_true, y_pred_perfect)
        ms_perfect_score = ms(y_true, y_pred_perfect)
        mze_perfect_score = mze(y_true, y_pred_perfect)
        tkendall_perfect_score = tkendall(y_true, y_pred_perfect)
        wkappa_perfect_score = wkappa(y_true, y_pred_perfect)
        spearman_perfect_score = spearman(y_true, y_pred_perfect)
        
        ccr_inverse_score = ccr(y_true, y_pred_inverse)
        amae_inverse_score = amae(y_true, y_pred_inverse)
        gm_inverse_score = gm(y_true, y_pred_inverse)
        mae_inverse_score = mae(y_true, y_pred_inverse)
        mmae_inverse_score = mmae(y_true, y_pred_inverse)
        recall_inverse_score = recall(y_true, y_pred_inverse)
        ms_inverse_score = ms(y_true, y_pred_inverse)
        mze_inverse_score = mze(y_true, y_pred_inverse)
        tkendall_inverse_score = tkendall(y_true, y_pred_inverse)
        wkappa_inverse_score = wkappa(y_true, y_pred_inverse)
        spearman_inverse_score = spearman(y_true, y_pred_inverse)
        
        # Ranges
        # [0, 1]
        assert (0 <= ccr_perfect_score <= 1) and (0 <= ccr_inverse_score <= 1)
        assert (0 <= gm_perfect_score <= 1) and (0 <= gm_inverse_score <= 1)
        assert (0 <= recall_perfect_score <= 1) and (0 <= recall_inverse_score <= 1)
        assert (0 <= ms_perfect_score <= 1) and (0 <= ms_inverse_score <= 1)
        assert (0 <= mze_perfect_score <= 1) and (0 <= mze_inverse_score <= 1)
        
        # [-1, 1]
        assert (-1 <= tkendall_perfect_score <= 1) and (-1 <= tkendall_inverse_score <= 1)
        assert (-1 <= wkappa_perfect_score <= 1) and (-1 <= wkappa_inverse_score <= 1)
        assert (-1 <= spearman_perfect_score <= 1) and (-1 <= spearman_inverse_score <= 1)

        # [0, infinito)
        assert (0 <= amae_perfect_score) and (0 <= amae_inverse_score)
        assert (0 <= mae_perfect_score) and (0 <= mae_inverse_score)
        assert (0 <= mmae_perfect_score) and (0 <= mmae_inverse_score)
        

class TestMetricsEdgeCases:
    """Tests for edge cases"""
    
    @pytest.mark.edge_case
    def test_empty_predictions(self):
        """Metrics with empty arrays"""
        from bsix.utils.classification_metrics import ccr, amae, gm, mae, mmae, recall, ms, mze, tkendall, wkappa, spearman
        
        y_true = np.array([])
        y_pred = np.array([])

        # Should handle
        with pytest.raises((ValueError)):
            ccr(y_true, y_pred)
        with pytest.raises((ValueError)):
            amae(y_true, y_pred)
        with pytest.raises((ValueError)):
            gm(y_true, y_pred)
        with pytest.raises((ValueError)):
            mae(y_true, y_pred)
        with pytest.raises((ValueError)):
            mmae(y_true, y_pred)
        with pytest.raises((ValueError)):
            recall(y_true, y_pred)
        with pytest.raises((ValueError)):
            ms(y_true, y_pred)
        with pytest.raises((ValueError)):
            mze(y_true, y_pred)
        with pytest.raises((ValueError)):
            tkendall(y_true, y_pred)
        with pytest.raises((ValueError)):
            wkappa(y_true, y_pred)
        with pytest.raises((ValueError)):
            spearman(y_true, y_pred)


class TestMetricsConsistency:
    """Tests for metric consistency"""
    
    def test_consistency_predictions(self, survival_data, random_predictions):
        """Metrics should be consistent for the same predictions"""
        from bsix.utils.classification_metrics import ccr, amae, gm, mae, mmae, recall, ms, mze, tkendall, wkappa, spearman

        _, y_true = survival_data
        y_pred = random_predictions

        y_true = y_true["event"]
        y_pred = np.where(y_pred >= 0.5, 1.0, 0.0)

        ccr_score1 = ccr(y_true, y_pred)
        amae_score1 = amae(y_true, y_pred)
        gm_score1 = gm(y_true, y_pred)
        mae_score1 = mae(y_true, y_pred)
        mmae_score1 = mmae(y_true, y_pred)
        recall_score1 = recall(y_true, y_pred)
        ms_score1 = ms(y_true, y_pred)
        mze_score1 = mze(y_true, y_pred)
        tkendall_score1 = tkendall(y_true, y_pred)
        wkappa_score1 = wkappa(y_true, y_pred)
        spearman_score1 = spearman(y_true, y_pred)
        
        ccr_score2 = ccr(y_true, y_pred)
        amae_score2 = amae(y_true, y_pred)
        gm_score2 = gm(y_true, y_pred)
        mae_score2 = mae(y_true, y_pred)
        mmae_score2 = mmae(y_true, y_pred)
        recall_score2 = recall(y_true, y_pred)
        ms_score2 = ms(y_true, y_pred)
        mze_score2 = mze(y_true, y_pred)
        tkendall_score2 = tkendall(y_true, y_pred)
        wkappa_score2 = wkappa(y_true, y_pred)
        spearman_score2 = spearman(y_true, y_pred)
            
        assert np.allclose(ccr_score1, ccr_score2, atol=1e-3)
        assert np.allclose(amae_score1, amae_score2, atol=1e-3)
        assert np.allclose(gm_score1, gm_score2, atol=1e-3)
        assert np.allclose(mae_score1, mae_score2, atol=1e-3)
        assert np.allclose(mmae_score1, mmae_score2, atol=1e-3)
        assert np.allclose(recall_score1, recall_score2, atol=1e-3)
        assert np.allclose(ms_score1, ms_score2, atol=1e-3)
        assert np.allclose(mze_score1, mze_score2, atol=1e-3)
        assert np.allclose(tkendall_score1, tkendall_score2, atol=1e-3)
        assert np.allclose(wkappa_score1, wkappa_score2, atol=1e-3)
        assert np.allclose(spearman_score1, spearman_score2, atol=1e-3)
