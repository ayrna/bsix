"""
Tests for loading and managing results (Category RES).
Tests for bsix.utils.load_results following BSIX_TEST_SPECIFICATION.md
"""

import pytest


class TestLoadResultsBasic:

    """
    Tests for get_results function (RES-001 to RES-006)
    """
    
    def test_load_all_results(self):

        """
        RES-001: get_results loads all results from folder
        """

        from bsix.utils.load_results import get_results
        
        results = get_results(result_folder="./tests/_results_")
        
        assert len(results) > 0

        for result in results:
            assert hasattr(result, "config")
            assert hasattr(result, "data_")
    
    
    def test_filter_by_estimator(self):

        """
        RES-002: get_results filters by estimator_name
        """

        from bsix.utils.load_results import get_results
        
        results = get_results(
            result_folder="./tests/_results_",
            estimator_name="DeepSurv"
        )

        assert len(results) > 0
        
        for result in results:
            assert hasattr(result, "config")
            assert hasattr(result, "data_")
    
    
    def test_filter_by_dataset(self):
        """RES-003: get_results filters by dataset"""
        from bsix.utils.load_results import get_results
        
        results = get_results(
            result_folder="./tests/_results_",
            dataset="breastcancer.arff"
        )

        assert len(results) > 0
        
        for result in results:
            assert hasattr(result, "config")
            assert hasattr(result, "data_")
    
    
    def test_filter_by_seed(self):
        """RES-004: get_results filters by seed"""
        from bsix.utils.load_results import get_results
        
        results = get_results(
            result_folder="./tests/_results_",
            seed=3
        )

        assert len(results) > 0
        
        for result in results:
            assert hasattr(result, "config")
            assert hasattr(result, "data_")
    
    
    def test_multiple_filters(self):
        """RES-005: Multiple filters applied simultaneously"""
        from bsix.utils.load_results import get_results
        
        results = get_results(
            result_folder="./tests/_results_",
            estimator_name="DeepSurv",
            dataset="breastcancer.arff",
            seed=3
        )

        assert len(results) > 0
        
        for result in results:
            assert hasattr(result, "config")
            assert hasattr(result, "data_")
    
    
    def test_no_matching_results(self):
        """RES-006: Returns empty list when no matches"""
        from bsix.utils.load_results import get_results
        
        results = get_results(
            result_folder="./tests/_results_",
            estimator_name="InvalidEstimator",
            dataset="InvalidDataset",
            seed="InvalidSeed"
        )

        assert len(results) == 0
            

class TestResultObject:
    """Tests for Result object properties (RES-007, RES-008)"""
    
    def test_result_config_access(self):
        """RES-007: Result.config is accessible and has estimator_name"""
        from bsix.utils.load_results import get_results
        
        result = get_results(result_folder="./tests/_results_")[0]
        
        assert "estimator_name" in result.config
        assert "dataset" in result.config
        assert "random_state" in result.config
    
    
    def test_result_data_access(self):
        """RES-008: Result.data is accessible"""
        from bsix.utils.load_results import get_results
        
        result = get_results(result_folder="./tests/_results_")[0]
        
        assert hasattr(result.data_, "targets")
        assert hasattr(result.data_, "predictions")
        assert hasattr(result.data_, "train_targets")
        assert hasattr(result.data_, "train_predictions")
        assert hasattr(result.data_, "best_model")
        assert hasattr(result.data_, "best_params")


class TestResultsEdgeCases:
    """Tests for edge cases"""
    
    @pytest.mark.edge_case
    def test_results_empty_folder(self):
        """get_results with empty folder"""
        from bsix.utils.load_results import get_results
        
        results = get_results(result_folder="InvalidFolderPath")

        assert len(results) == 0


class TestReportGeneration:
    """Tests for report generation"""
    
    def test_generate_report(self):
        """Excel report can be generated from results"""
        from bsix.utils.load_results import get_results, save_results
        
        results = get_results(result_folder="./tests/_results_")

        assert len(results) > 0
        print(results[0].data_.targets)
        save_results(result_folder="./tests/_results_", destination_path="./tests/_results_/report.xlsx")

        # Check if the report file was created
        import os
        assert os.path.exists("./tests/_results_/report.xlsx")
        os.remove("./tests/_results_/report.xlsx")
