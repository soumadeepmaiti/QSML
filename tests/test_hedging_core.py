#!/usr/bin/env python3
"""
Minimal test to verify the hedging simulation system works.
Tests only the hedging components that we just created.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_hedging_simulation_imports():
    """Test that the hedging simulation module imports correctly."""
    try:
        # Test direct import of hedging components
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'qsml', 'hedging'))
        
        from qsml.hedging import simulation
        from qsml.hedging import analysis
        
        # Test that key classes exist
        assert hasattr(simulation, 'HedgeType')
        assert hasattr(simulation, 'RebalanceFrequency') 
        assert hasattr(simulation, 'TransactionCosts')
        assert hasattr(simulation, 'HedgingConfig')
        assert hasattr(simulation, 'PathSimulator')
        assert hasattr(analysis, 'HedgingAnalyzer')
        
        print("✓ Hedging simulation imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_transaction_costs_direct():
    """Test transaction costs calculation directly."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'qsml', 'hedging'))
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

def test_path_simulator_direct():
    """Test stock path simulation directly."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'qsml', 'hedging'))
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

def test_hedging_analyzer_direct():
    """Test hedging analyzer directly."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'qsml', 'hedging'))
        from qsml.hedging.analysis import HedgingAnalyzer
        
        analyzer = HedgingAnalyzer()
        
        # Create sample results
        sample_results = {
            'path_results': [
                {'net_pnl': 10.0, 'n_trades': 5, 'total_costs': 2.0},
                {'net_pnl': -5.0, 'n_trades': 8, 'total_costs': 3.0},
                {'net_pnl': 15.0, 'n_trades': 6, 'total_costs': 2.5}
            ]
        }
        
        analysis = analyzer.analyze_pnl_distribution(sample_results)
        
        assert 'pnl_statistics' in analysis, "Should have P&L statistics"
        assert 'risk_metrics' in analysis, "Should have risk metrics"
        
        pnl_stats = analysis['pnl_statistics']
        assert 'mean' in pnl_stats, "Should have mean"
        assert 'std' in pnl_stats, "Should have std"
        
        print("✓ Hedging analyzer test passed")
        return True
    except Exception as e:
        print(f"✗ Hedging analyzer test failed: {e}")
        return False

def test_hedging_config_creation():
    """Test hedging configuration creation."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'qsml', 'hedging'))
        from qsml.hedging.simulation import HedgingConfig, HedgeType, RebalanceFrequency
        
        config = HedgingConfig(
            hedge_type=HedgeType.DELTA,
            rebalance_frequency=RebalanceFrequency.DAILY,
            delta_threshold=0.1,
            initial_stock_price=100.0,
            option_params={
                'K': 100.0,
                'T': 1.0,
                'r': 0.05,
                'option_type': 'call'
            }
        )
        
        assert config.hedge_type == HedgeType.DELTA
        assert config.rebalance_frequency == RebalanceFrequency.DAILY
        assert config.delta_threshold == 0.1
        assert config.option_params['K'] == 100.0
        
        print("✓ Hedging config test passed")
        return True
    except Exception as e:
        print(f"✗ Hedging config test failed: {e}")
        return False

def test_strategy_comparison_direct():
    """Test strategy comparison functionality directly."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'qsml', 'hedging'))
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

def main():
    """Run all tests."""
    print("Running QSML Hedging System Core Tests")
    print("=" * 50)
    
    tests = [
        test_hedging_simulation_imports,
        test_transaction_costs_direct,
        test_path_simulator_direct,
        test_hedging_analyzer_direct,
        test_hedging_config_creation,
        test_strategy_comparison_direct
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
        print("🎉 All core hedging tests passed!")
        print("\nThe hedging simulation and analysis system is working correctly!")
        print("Key components verified:")
        print("- Transaction cost modeling")
        print("- Monte Carlo path simulation")
        print("- Hedging configuration")
        print("- P&L analysis and risk metrics")
        print("- Strategy comparison framework")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())