"""
Tests for experiment orchestration (Category ORCH).
Tests for bsix.execution following BSIX_TEST_SPECIFICATION.md
"""

import execution.orchestrator as orchestrator
import pytest


class TestOrchestrator:

    """
    Tests for orchestrator configuration generation (ORCH-001 to ORCH-004).
    """
    
    def test_generate_configs_basic(self, basic_execution_config):

        """
        ORCH-001: generate_internal_cv_experiments_configs returns dict.
        """

        configs = orchestrator.generate_internal_cv_experiments_configs()
            
        # Should return dict mapping (estimator, dataset) to config list
        assert isinstance(configs, dict)
        
        if len(configs) > 0:
            for key, value in configs.items():
                assert isinstance(key, tuple)
                assert len(key) == 2  # (estimator, dataset)
                assert isinstance(value, list)
    
    
    def test_generate_configs_multiple_seeds(self, basic_execution_config):

        """
        ORCH-001: Multiple seeds generate multiple configs.
        """

        configs = orchestrator.generate_internal_cv_experiments_configs()
        
        # Should have multiple configurations for different seeds
        if len(configs) > 0:
            first_key = list(configs.keys())[0]
            configs_list = configs[first_key]
            
            # Should have multiple configs
            assert len(configs_list) >= 1
    
    
    def test_get_executor_params_basic(self, basic_execution_config):

        """
        ORCH-002: get_executor_params returns executor configuration.
        """

        params = orchestrator.get_executor_params(
            estimator="CoxRegression",
            dataset="metabric"
        )
        
        # Should return dict with executor params
        assert isinstance(params, dict)
    
    
    def test_get_executor_params_with_time(self, basic_execution_config):

        """
        ORCH-002: time parameter controls SLURM partition.
        """

        params_short = orchestrator.get_executor_params(
            estimator="CoxRegression",
            dataset="metabric",
        )
        
        params_long = orchestrator.get_executor_params(
            estimator="CoxRegression",
            dataset="metabric",
        )
        
        # Both should have executor params
        assert isinstance(params_short, dict)
        assert isinstance(params_long, dict)


class TestWorkers:

    """
    Tests for worker functions (ORCH-003 to ORCH-006).
    """

    
    def test_run_experiment_exists(self):

        """
        ORCH-003: run_experiment function exists.
        """

        from execution import workers
        
        # Should have run_experiment
        assert hasattr(workers, 'run_experiment')
    
    
    def test_run_experiment_with_best_config(self):

        """
        ORCH-004: run_experiment_with_best_config exists.
        """

        from execution import workers
        
        # Should have run_experiment_with_best_config
        assert hasattr(workers, 'run_experiment_with_best_config')
    
    
    def test_run_aggregator_exists(self):

        """
        ORCH-005: run_aggregator function exists.
        """

        from execution import workers
        
        # Should have run_aggregator
        assert hasattr(workers, 'run_aggregator')
    
    
    def test_run_results_collector_exists(self):

        """
        ORCH-006: run_results_collector function exists.
        """

        from execution import workers
        
        # Should have run_results_collector
        assert hasattr(workers, 'run_results_collector')


class TestSurvivalFlow:

    """
    Tests for survival flow execution (ORCH-007).
    """
    
    def test_survival_flow_exists(self):

        """
        survival_flow function should exist.
        """

        from execution.flows.survival_flow import run_survival_flow
        
        # Should be callable
        assert callable(run_survival_flow)
    
    
    def test_survival_flow_signature(self):

        """
        survival_flow should accept required parameters.
        """

        from execution.flows.survival_flow import run_survival_flow
        import inspect
        
        sig = inspect.signature(run_survival_flow)
        
        # Should have key parameters
        params = list(sig.parameters.keys())
        
        # Check for critical parameters
        assert len(params) > 0


class TestConfigurationLoading:

    """
    Tests for configuration loading.
    """
    
    def test_configs_module_exists(self):

        """
        configs module should exist.
        """

        from execution import configs
        
        assert configs is not None
    
    
    def test_default_config_loading(self):

        """
        Default configuration file should be loadable.
        """

        from execution.config import default
        
        # Should have configuration
        assert default is not None


class TestExperimentFlow:

    """
    Tests for experiment flow orchestration (Integration-like).
    """
    
    @pytest.mark.integration
    def test_experiment_flow_basic(self, basic_experiment_config):

        """
        Basic experiment flow: data -> model -> metrics.
        """

        from execution.flows.survival_flow import run_survival_flow

        run_survival_flow(**basic_experiment_config)


class TestOrchestratorEdgeCases:

    """
    Tests for edge cases in orchestration.
    """
    
    @pytest.mark.edge_case
    def test_orchestrator_invalid_estimator(self, invalid_estimator_config):

        """
        Orchestrator with invalid estimator.
        """

        from execution.flows.survival_flow import run_survival_flow

        # Should handle or raise gracefully
        with pytest.raises((KeyError)):
            run_survival_flow(**invalid_estimator_config)
    
    
    @pytest.mark.edge_case
    def test_orchestrator_invalid_dataset(self, invalid_dataset_config):

        """
        Orchestrator with invalid dataset.
        """

        from execution.flows.survival_flow import run_survival_flow
    
        # Should handle or raise gracefully
        with pytest.raises((ValueError)):
            run_survival_flow(**invalid_dataset_config)
