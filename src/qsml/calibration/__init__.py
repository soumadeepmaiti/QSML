"""
Heston model calibration framework.

This package provides comprehensive tools for calibrating the Heston stochastic 
volatility model to market option prices:

- Surface data structures for managing implied volatility data
- Objective functions and optimization algorithms  
- Parameter constraints and validation
- Multi-method calibration with quality assessment
"""

from .surface import (
    SmileSlice,
    VolatilitySurface,
    build_surface_from_dataframe
)

from .objective import (
    HestonCalibrator,
    CalibrationConfig,
    CalibrationResult,
    ObjectiveType,
    CalibrationMethod
)

from .engine import (
    HestonParameters,
    CalibrationConstraints,
    HestonCalibrationEngine,
    create_calibration_report
)

__all__ = [
    # Surface data structures
    'SmileSlice',
    'VolatilitySurface', 
    'build_surface_from_dataframe',
    
    # Calibration core
    'HestonCalibrator',
    'CalibrationConfig',
    'CalibrationResult',
    'ObjectiveType',
    'CalibrationMethod',
    
    # High-level engine
    'HestonParameters',
    'CalibrationConstraints', 
    'HestonCalibrationEngine',
    'create_calibration_report'
]