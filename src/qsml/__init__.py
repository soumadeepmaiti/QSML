"""
QSML (Quantitative Stochastic Volatility Machine Learning) Project
================================================================

A comprehensive Master's-level quantitative finance project that combines:
- Heston stochastic volatility model calibration
- Machine learning surrogate pricing models
- Advanced hedging simulation and analysis

This project implements a complete pipeline from market data processing
to sophisticated option pricing, model calibration, ML-based surrogates,
and realistic hedging simulation with transaction costs.

Key Components:
- Black-Scholes and Heston pricing engines
- Volatility surface calibration
- ML surrogate models with arbitrage constraints
- Monte Carlo hedging simulation
- Comprehensive analysis and visualization tools

Author: Master's Project in Quantitative Finance
"""

__version__ = "1.0.0"
__author__ = "Quantitative Finance Project"

# Core modules  
from .pricers import bs
from .pricers import heston_fft 
from .pricers import heston_integral
from .calibration.surface import VolatilitySurface
from .calibration.objective import HestonCalibrator
from .ml_surrogate.architecture import HestonSurrogateNet
from .hedging.simulation import HedgingSimulator, HedgingConfig
from .hedging.analysis import HedgingAnalyzer

__all__ = [
    'bs',
    'heston_fft', 
    'heston_integral',
    'VolatilitySurface',
    'HestonCalibrator',
    'HestonSurrogateNet',
    'HedgingSimulator',
    'HedgingConfig',
    'HedgingAnalyzer'
]
__author__ = "Finance Quant Team"
__email__ = "quant@example.com"

# Import key modules for easy access
from .utils.config import load_config
from .utils.logging_utils import setup_logging

__all__ = [
    "load_config",
    "setup_logging",
]