#!/usr/bin/env python3
"""
Simple test runner for the hedging simulation system.
Tests the core functionality without complex imports.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_basic_imports():
    """Test that we can import the basic hedging modules."""
    try:
        from qsml.hedging.simulation import HedgeType, RebalanceFrequency, TransactionCosts
        from qsml.hedging.analysis import HedgingAnalyzer
        print("✓ Basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_transaction_costs():
    """Test transaction costs calculation."""
    try:
        from qsml.hedging.simulation import TransactionCosts
        
        costs = TransactionCosts(
            fixed_cost=1.0,
            proportional_cost=0.001,
            bid_ask_spread=0.01,
            market_impact_coeff=0.0001
        )
        
        cost = costs.calculate_cost(
            quantity=100,
            price=50.0,
            trade_type='buy'
        )
        
        assert cost > 0, "Cost should be positive"
        assert isinstance(cost, float), "Cost should be a float"
        print("✓ Transaction costs test passed")
        return True
    except Exception as e:
        print(f"✗ Transaction costs test failed: {e}")
        return False

def test_hedging_analyzer():
    """Test hedging analyzer with sample data."""
    try:
        from qsml.hedging.analysis import HedgingAnalyzer
        
        analyzer = HedgingAnalyzer()
        
        # Create sample results
        sample_results = {
            'path_results': [
                {'net_pnl': 10.0, 'n_trades': 5, 'total_costs': 2.0},
                {'net_pnl': -5.0, 'n_trades': 8, 'total_costs': 3.0},
                {'net_pnl': 15.0, 'n_trades': 6, 'total_costs': 2.5},
                {'net_pnl': 0.0, 'n_trades': 7, 'total_costs': 2.8},
                {'net_pnl': -10.0, 'n_trades': 9, 'total_costs': 3.5}
            ]
        }
        
        analysis = analyzer.analyze_pnl_distribution(sample_results)
        
        assert 'pnl_statistics' in analysis, "Should have P&L statistics"
        assert 'risk_metrics' in analysis, "Should have risk metrics"
        assert 'performance_metrics' in analysis, "Should have performance metrics"
        
        pnl_stats = analysis['pnl_statistics']
        assert 'mean' in pnl_stats, "Should have mean"
        assert 'std' in pnl_stats, "Should have std"
        
        print("✓ Hedging analyzer test passed")
        return True
    except Exception as e:
        print(f"✗ Hedging analyzer test failed: {e}")
        return False

def test_path_simulator():
    """Test stock price path simulation."""
    try:
        from qsml.hedging.simulation import PathSimulator
        
        simulator = PathSimulator()
        
        path = simulator.simulate_path(
            S0=100.0,
            T=1.0,
            n_steps=252,
            r=0.05,
            sigma=0.2,
            random_seed=42
        )
        
        assert len(path) == 253, f"Path should have 253 points, got {len(path)}"
        assert path[0] == 100.0, f"Initial price should be 100, got {path[0]}"
        assert all(p > 0 for p in path), "All prices should be positive"
        
        print("✓ Path simulator test passed")
        return True
    except Exception as e:
        print(f"✗ Path simulator test failed: {e}")
        return False

def test_strategy_comparison():
    """Test strategy comparison functionality."""
    try:
        from qsml.hedging.analysis import HedgingAnalyzer
        
        analyzer = HedgingAnalyzer()
        
        # Create sample strategy results
        strategy_results = {
            'strategy_a': {
                'path_results': [
                    {'net_pnl': 10.0, 'n_trades': 5, 'total_costs': 2.0},
                    {'net_pnl': 5.0, 'n_trades': 6, 'total_costs': 2.5}
                ]
            },
            'strategy_b': {
                'path_results': [
                    {'net_pnl': 8.0, 'n_trades': 4, 'total_costs': 1.5},
                    {'net_pnl': 12.0, 'n_trades': 5, 'total_costs': 2.0}
                ]
            }
        }
        
        comparison_df = analyzer.compare_strategies(strategy_results)
        
        assert isinstance(comparison_df, pd.DataFrame), "Should return DataFrame"
        assert 'strategy' in comparison_df.columns, "Should have strategy column"
        assert len(comparison_df) == 2, f"Should have 2 strategies, got {len(comparison_df)}"
        
        print("✓ Strategy comparison test passed")
        return True
    except Exception as e:
        print(f"✗ Strategy comparison test failed: {e}")
        return False

def test_numerical_stability():
    """Test numerical stability of calculations."""
    try:
        from qsml.hedging.analysis import HedgingAnalyzer
        
        analyzer = HedgingAnalyzer()
        
        # Test with extreme values
        extreme_results = {
            'path_results': [
                {'net_pnl': 1e6, 'n_trades': 1000, 'total_costs': 1e3},
                {'net_pnl': -1e6, 'n_trades': 500, 'total_costs': 5e2},
                {'net_pnl': 0.001, 'n_trades': 1, 'total_costs': 0.1}
            ]
        }
        
        analysis = analyzer.analyze_pnl_distribution(extreme_results)
        
        # Check that we get finite results
        assert np.isfinite(analysis['pnl_statistics']['mean']), "Mean should be finite"
        assert np.isfinite(analysis['pnl_statistics']['std']), "Std should be finite"
        
        print("✓ Numerical stability test passed")
        return True
    except Exception as e:
        print(f"✗ Numerical stability test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running QSML Hedging System Tests")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_transaction_costs,
        test_hedging_analyzer,
        test_path_simulator,
        test_strategy_comparison,
        test_numerical_stability
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())