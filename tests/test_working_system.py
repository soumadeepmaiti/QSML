import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qsml.pricers.bs import BlackScholesPricer
from qsml.hedging.simulation import (
    HedgingSimulator, HedgingConfig, TransactionCosts,
    HedgeType, RebalanceFrequency, PathSimulator
)
from qsml.hedging.analysis import HedgingAnalyzer, generate_hedging_report


class TestPricingModels:
    """Test cases for pricing models."""
    
    def test_black_scholes_pricer(self):
        """Test Black-Scholes pricing functionality."""
        pricer = BlackScholesPricer()
        
        # Test basic pricing
        price = pricer.price(
            S=100.0, K=100.0, r=0.05, q=0.02, T=1.0, sigma=0.2, option_type='call'
        )
        
        assert isinstance(price, float)
        assert price > 0
        
        # Test put-call parity
        call_price = pricer.price(
            S=100.0, K=100.0, r=0.05, q=0.02, T=1.0, sigma=0.2, option_type='call'
        )
        put_price = pricer.price(
            S=100.0, K=100.0, r=0.05, q=0.02, T=1.0, sigma=0.2, option_type='put'
        )
        
        # Put-call parity: C - P = S*e^(-q*T) - K*e^(-r*T)
        expected_diff = 100.0 * np.exp(-0.02 * 1.0) - 100.0 * np.exp(-0.05 * 1.0)
        actual_diff = call_price - put_price
        
        assert abs(actual_diff - expected_diff) < 0.01
    
    def test_black_scholes_greeks(self):
        """Test Black-Scholes Greeks calculation."""
        pricer = BlackScholesPricer()
        
        greeks = pricer.all_greeks(
            S=100.0, K=100.0, r=0.05, q=0.02, T=1.0, sigma=0.2, option_type='call'
        )
        
        expected_greeks = ['price', 'delta', 'gamma', 'vega', 'theta', 'rho']
        for greek in expected_greeks:
            assert greek in greeks
            assert isinstance(greeks[greek], float)
        
        # Delta should be between 0 and 1 for call
        assert 0 <= greeks['delta'] <= 1
        
        # Gamma should be positive
        assert greeks['gamma'] > 0
        
        # Vega should be positive
        assert greeks['vega'] > 0


class TestHedgingSystem:
    """Test cases for hedging system."""
    
    def test_hedging_config(self):
        """Test hedging configuration."""
        config = HedgingConfig(
            hedge_type=HedgeType.DELTA,
            rebalance_frequency=RebalanceFrequency.DAILY,
            transaction_costs=TransactionCosts()
        )
        
        assert config.hedge_type == HedgeType.DELTA
        assert config.rebalance_frequency == RebalanceFrequency.DAILY
        assert config.delta_threshold == 0.1  # default value
    
    def test_path_simulator(self):
        """Test path simulation."""
        config = HedgingConfig()
        simulator = PathSimulator(config)
        
        # Test GBM path simulation
        S0 = 100.0
        r = 0.05
        q = 0.02
        T = 1.0
        time_steps = np.linspace(0, T, 253)
        
        paths = simulator.simulate_stock_paths(S0, r, q, T, time_steps)
        
        assert paths.shape[1] == len(time_steps)
        assert np.all(paths > 0)  # Stock prices should be positive
    
    def test_hedging_analyzer(self):
        """Test hedging analyzer functionality."""
        analyzer = HedgingAnalyzer()
        
        # Create mock simulation results
        mock_results = {
            'path_results': [
                {
                    'path_id': 0,
                    'net_pnl': 5.0,
                    'n_trades': 10,
                    'total_costs': 1.0,
                    'positions': [
                        {'net_delta': 0.5, 'portfolio_value': 100.0},
                        {'net_delta': 0.3, 'portfolio_value': 102.0}
                    ]
                },
                {
                    'path_id': 1,
                    'net_pnl': -2.0,
                    'n_trades': 8,
                    'total_costs': 0.8,
                    'positions': [
                        {'net_delta': 0.4, 'portfolio_value': 99.0},
                        {'net_delta': 0.2, 'portfolio_value': 101.0}
                    ]
                }
            ],
            'config': {
                'hedge_type': 'delta'
            },
            'n_successful_paths': 2
        }
        
        # Test P&L analysis
        pnl_analysis = analyzer.analyze_pnl_distribution(mock_results)
        
        assert 'pnl_statistics' in pnl_analysis
        assert 'risk_metrics' in pnl_analysis
        assert 'performance_metrics' in pnl_analysis
        
        # Test trading analysis
        trading_analysis = analyzer.analyze_trading_activity(mock_results)
        
        assert 'cost_statistics' in trading_analysis
        
        # Test hedging effectiveness
        effectiveness = analyzer.analyze_hedging_effectiveness(mock_results)
        
        assert isinstance(effectiveness, dict)


class TestDataIntegrity:
    """Test data integrity and file structure."""
    
    def test_project_structure(self):
        """Test that required project files exist."""
        project_root = Path(__file__).parent.parent
        
        # Check key directories
        assert (project_root / 'src' / 'qsml').exists()
        assert (project_root / 'notebooks').exists()
        assert (project_root / 'tests').exists()
        
        # Check key source files
        assert (project_root / 'src' / 'qsml' / 'pricers' / 'bs.py').exists()
        assert (project_root / 'src' / 'qsml' / 'hedging' / 'simulation.py').exists()
        assert (project_root / 'src' / 'qsml' / 'hedging' / 'analysis.py').exists()
        
        # Check configuration files
        assert (project_root / 'pyproject.toml').exists()
        assert (project_root / 'requirements.txt').exists()
    
    def test_import_structure(self):
        """Test that main modules can be imported."""
        # These imports should work without errors
        from qsml.pricers.bs import BlackScholesPricer
        from qsml.hedging.simulation import HedgingConfig, HedgeType
        from qsml.hedging.analysis import HedgingAnalyzer
        
        # Test instantiation
        pricer = BlackScholesPricer()
        analyzer = HedgingAnalyzer()
        
        assert pricer is not None
        assert analyzer is not None


if __name__ == "__main__":
    pytest.main([__file__])