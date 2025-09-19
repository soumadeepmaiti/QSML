import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from scipy.optimize import Bounds
from .objective import (
    HestonCalibrator, 
    CalibrationConfig, 
    CalibrationResult,
    ObjectiveType,
    CalibrationMethod
)
from .surface import VolatilitySurface, SmileSlice

logger = logging.getLogger(__name__)


@dataclass
class HestonParameters:
    """Container for Heston model parameters with validation."""
    
    kappa: float  # Mean reversion speed
    theta: float  # Long-term variance
    sigma: float  # Vol of vol
    rho: float    # Correlation
    v0: float     # Initial variance
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """Validate parameter constraints."""
        issues = []
        
        if self.kappa <= 0:
            issues.append("kappa must be positive")
        
        if self.theta <= 0:
            issues.append("theta must be positive")
        
        if self.sigma <= 0:
            issues.append("sigma must be positive")
        
        if not -1 < self.rho < 1:
            issues.append("rho must be in (-1, 1)")
        
        if self.v0 <= 0:
            issues.append("v0 must be positive")
        
        # Feller condition
        if 2 * self.kappa * self.theta < self.sigma**2:
            issues.append("Feller condition violated: 2*kappa*theta < sigma^2")
        
        if issues:
            logger.warning(f"Parameter validation issues: {issues}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'kappa': self.kappa,
            'theta': self.theta,
            'sigma': self.sigma,
            'rho': self.rho,
            'v0': self.v0
        }
    
    @classmethod
    def from_dict(cls, params: Dict[str, float]) -> 'HestonParameters':
        """Create from dictionary."""
        return cls(
            kappa=params['kappa'],
            theta=params['theta'],
            sigma=params['sigma'],
            rho=params['rho'],
            v0=params['v0']
        )
    
    @classmethod
    def default_params(cls) -> 'HestonParameters':
        """Default parameter set."""
        return cls(
            kappa=2.0,
            theta=0.04,
            sigma=0.3,
            rho=-0.5,
            v0=0.04
        )
    
    def __str__(self) -> str:
        """String representation."""
        return (f"HestonParameters(kappa={self.kappa:.4f}, theta={self.theta:.4f}, "
                f"sigma={self.sigma:.4f}, rho={self.rho:.4f}, v0={self.v0:.4f})")


@dataclass
class CalibrationConstraints:
    """Constraints for Heston calibration."""
    
    # Parameter bounds
    kappa_bounds: Tuple[float, float] = (0.01, 20.0)
    theta_bounds: Tuple[float, float] = (0.001, 1.0)
    sigma_bounds: Tuple[float, float] = (0.01, 2.0)
    rho_bounds: Tuple[float, float] = (-0.99, 0.99)
    v0_bounds: Tuple[float, float] = (0.001, 1.0)
    
    # Model constraints
    enforce_feller: bool = True
    min_total_variance: float = 0.0001  # Minimum integrated variance
    max_total_variance: float = 10.0    # Maximum integrated variance
    
    # Market data constraints
    min_moneyness: float = 0.7
    max_moneyness: float = 1.3
    min_time_to_expiry: float = 1/365  # 1 day
    max_time_to_expiry: float = 2.0    # 2 years
    min_option_price: float = 0.001    # Minimum option value
    
    def to_scipy_bounds(self) -> Bounds:
        """Convert to scipy Bounds object."""
        return Bounds(
            lb=np.array([
                self.kappa_bounds[0],
                self.theta_bounds[0],
                self.sigma_bounds[0],
                self.rho_bounds[0],
                self.v0_bounds[0]
            ]),
            ub=np.array([
                self.kappa_bounds[1],
                self.theta_bounds[1],
                self.sigma_bounds[1],
                self.rho_bounds[1],
                self.v0_bounds[1]
            ])
        )
    
    def validate_parameters(self, params: HestonParameters) -> List[str]:
        """Validate parameters against constraints."""
        violations = []
        
        if not (self.kappa_bounds[0] <= params.kappa <= self.kappa_bounds[1]):
            violations.append(f"kappa {params.kappa} outside bounds {self.kappa_bounds}")
        
        if not (self.theta_bounds[0] <= params.theta <= self.theta_bounds[1]):
            violations.append(f"theta {params.theta} outside bounds {self.theta_bounds}")
        
        if not (self.sigma_bounds[0] <= params.sigma <= self.sigma_bounds[1]):
            violations.append(f"sigma {params.sigma} outside bounds {self.sigma_bounds}")
        
        if not (self.rho_bounds[0] <= params.rho <= self.rho_bounds[1]):
            violations.append(f"rho {params.rho} outside bounds {self.rho_bounds}")
        
        if not (self.v0_bounds[0] <= params.v0 <= self.v0_bounds[1]):
            violations.append(f"v0 {params.v0} outside bounds {self.v0_bounds}")
        
        if self.enforce_feller and 2 * params.kappa * params.theta < params.sigma**2:
            violations.append("Feller condition violated")
        
        return violations


class HestonCalibrationEngine:
    """
    Main calibration engine for Heston models.
    
    Provides high-level interface for calibrating Heston parameters
    to market volatility surfaces with various optimization strategies.
    """
    
    def __init__(
        self,
        constraints: Optional[CalibrationConstraints] = None,
        config: Optional[CalibrationConfig] = None
    ):
        """Initialize calibration engine."""
        self.constraints = constraints or CalibrationConstraints()
        self.config = config or CalibrationConfig()
        
        # Update config with constraints
        self.config.kappa_bounds = self.constraints.kappa_bounds
        self.config.theta_bounds = self.constraints.theta_bounds
        self.config.sigma_bounds = self.constraints.sigma_bounds
        self.config.rho_bounds = self.constraints.rho_bounds
        self.config.v0_bounds = self.constraints.v0_bounds
        self.config.enforce_feller = self.constraints.enforce_feller
        
        self.calibrator = HestonCalibrator(self.config)
        self.logger = logger
    
    def calibrate_single_expiry(
        self,
        slice_obj: SmileSlice,
        initial_guess: Optional[Union[HestonParameters, Dict[str, float]]] = None,
        method: Optional[CalibrationMethod] = None
    ) -> CalibrationResult:
        """
        Calibrate to single expiry slice.
        
        Args:
            slice_obj: Market data for single expiry
            initial_guess: Initial parameter guess
            method: Optimization method override
            
        Returns:
            CalibrationResult with optimized parameters
        """
        # Set method if provided
        if method is not None:
            original_method = self.config.method
            self.config.method = method
            self.calibrator.config.method = method
        
        # Convert initial guess if needed
        if isinstance(initial_guess, HestonParameters):
            initial_guess = initial_guess.to_dict()
        
        # Run calibration
        result = self.calibrator.calibrate_slice(slice_obj, initial_guess)
        
        # Restore original method
        if method is not None:
            self.config.method = original_method
            self.calibrator.config.method = original_method
        
        return result
    
    def calibrate_surface_sequential(
        self,
        surface: VolatilitySurface,
        initial_guess: Optional[Union[HestonParameters, Dict[str, float]]] = None
    ) -> Dict[str, CalibrationResult]:
        """
        Calibrate each expiry sequentially, using previous results as initial guess.
        
        Args:
            surface: Complete volatility surface
            initial_guess: Initial parameter guess for first expiry
            
        Returns:
            Dictionary of calibration results by expiry
        """
        # Convert initial guess if needed
        if isinstance(initial_guess, HestonParameters):
            initial_guess = initial_guess.to_dict()
        
        return self.calibrator.calibrate_surface(
            surface=surface,
            joint_calibration=False,
            initial_guess=initial_guess
        )
    
    def calibrate_surface_joint(
        self,
        surface: VolatilitySurface,
        initial_guess: Optional[Union[HestonParameters, Dict[str, float]]] = None
    ) -> CalibrationResult:
        """
        Joint calibration across all expiries.
        
        Args:
            surface: Complete volatility surface
            initial_guess: Initial parameter guess
            
        Returns:
            Single calibration result for all expiries
        """
        # Convert initial guess if needed
        if isinstance(initial_guess, HestonParameters):
            initial_guess = initial_guess.to_dict()
        
        return self.calibrator.calibrate_surface(
            surface=surface,
            joint_calibration=True,
            initial_guess=initial_guess
        )
    
    def calibrate_with_multiple_methods(
        self,
        surface: VolatilitySurface,
        methods: Optional[List[CalibrationMethod]] = None,
        initial_guess: Optional[Union[HestonParameters, Dict[str, float]]] = None
    ) -> Dict[str, CalibrationResult]:
        """
        Try multiple optimization methods and return best result.
        
        Args:
            surface: Volatility surface
            methods: List of methods to try
            initial_guess: Initial parameter guess
            
        Returns:
            Dictionary of results by method name
        """
        if methods is None:
            methods = [
                CalibrationMethod.L_BFGS_B,
                CalibrationMethod.SLSQP,
                CalibrationMethod.DIFFERENTIAL_EVOLUTION
            ]
        
        # Convert initial guess if needed
        if isinstance(initial_guess, HestonParameters):
            initial_guess = initial_guess.to_dict()
        
        results = {}
        
        for method in methods:
            self.logger.info(f"Trying calibration method: {method.value}")
            
            try:
                if len(surface.slices) == 1:
                    # Single expiry
                    slice_obj = list(surface.slices.values())[0]
                    result = self.calibrate_single_expiry(slice_obj, initial_guess, method)
                else:
                    # Multiple expiries - use joint calibration
                    result = self.calibrate_surface_joint(surface, initial_guess)
                
                results[method.value] = result
                
            except Exception as e:
                self.logger.warning(f"Method {method.value} failed: {e}")
                # Create failed result
                results[method.value] = CalibrationResult(
                    params={param: np.nan for param in ['kappa', 'theta', 'sigma', 'rho', 'v0']},
                    success=False,
                    n_iterations=0,
                    objective_value=np.inf,
                    optimization_message=str(e),
                    price_rmse=np.nan,
                    feller_condition=False,
                    parameter_bounds_active={}
                )
        
        return results
    
    def get_best_calibration(
        self,
        results: Dict[str, CalibrationResult],
        prefer_feller: bool = True
    ) -> Tuple[str, CalibrationResult]:
        """
        Select best calibration result from multiple attempts.
        
        Args:
            results: Dictionary of calibration results
            prefer_feller: Prefer results that satisfy Feller condition
            
        Returns:
            Tuple of (method_name, best_result)
        """
        valid_results = {k: v for k, v in results.items() if v.success}
        
        if not valid_results:
            # Return least bad result
            best_key = min(results.keys(), key=lambda k: results[k].objective_value)
            return best_key, results[best_key]
        
        if prefer_feller:
            # First try to find results that satisfy Feller condition
            feller_results = {k: v for k, v in valid_results.items() if v.feller_condition}
            if feller_results:
                best_key = min(feller_results.keys(), key=lambda k: feller_results[k].objective_value)
                return best_key, feller_results[best_key]
        
        # Fall back to best objective value
        best_key = min(valid_results.keys(), key=lambda k: valid_results[k].objective_value)
        return best_key, valid_results[best_key]
    
    def estimate_initial_guess(self, surface: VolatilitySurface) -> HestonParameters:
        """
        Estimate reasonable initial parameter guess from market data.
        
        Args:
            surface: Volatility surface
            
        Returns:
            HestonParameters with initial guess
        """
        # Get ATM volatilities across expiries
        atm_vols = []
        times = []
        
        for slice_obj in surface.slices.values():
            atm_vol = slice_obj.get_atm_vol()
            if not np.isnan(atm_vol):
                atm_vols.append(atm_vol)
                times.append(slice_obj.time_to_expiry)
        
        if not atm_vols:
            # Fallback to default
            return HestonParameters.default_params()
        
        # Use average ATM vol as theta estimate
        avg_atm_vol = np.mean(atm_vols)
        theta = avg_atm_vol**2
        
        # Estimate v0 from shortest expiry
        if times:
            min_time_idx = np.argmin(times)
            v0 = atm_vols[min_time_idx]**2
        else:
            v0 = theta
        
        # Conservative estimates for other parameters
        kappa = 2.0  # Moderate mean reversion
        sigma = 0.3  # Moderate vol of vol
        rho = -0.5   # Typical negative correlation
        
        return HestonParameters(
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            v0=v0
        )
    
    def validate_calibration_quality(
        self,
        result: CalibrationResult,
        surface: VolatilitySurface
    ) -> Dict[str, any]:
        """
        Assess quality of calibration result.
        
        Args:
            result: Calibration result to validate
            surface: Original market data
            
        Returns:
            Dictionary with quality metrics
        """
        quality = {
            'success': result.success,
            'feller_condition': result.feller_condition,
            'price_rmse': result.price_rmse,
            'max_price_error': result.max_price_error,
            'n_options_used': result.n_options
        }
        
        # Parameter reasonableness checks
        if result.success and not np.isnan(result.price_rmse):
            params = HestonParameters.from_dict(result.params)
            
            quality.update({
                'parameters_reasonable': all([
                    0.1 <= params.kappa <= 10.0,      # Reasonable mean reversion
                    0.001 <= params.theta <= 0.5,     # Reasonable long-term var
                    0.01 <= params.sigma <= 1.0,      # Reasonable vol of vol
                    -0.95 <= params.rho <= 0.95,      # Reasonable correlation
                    0.001 <= params.v0 <= 0.5         # Reasonable initial var
                ]),
                'relative_price_error': result.price_rmse / np.mean(result.market_prices) if result.market_prices is not None else np.nan
            })
            
            # Check bounds activity
            active_bounds = sum(result.parameter_bounds_active.values()) if result.parameter_bounds_active else 0
            quality['bounds_active'] = active_bounds
            quality['likely_constrained'] = active_bounds >= 2
        
        return quality


def create_calibration_report(
    results: Dict[str, CalibrationResult],
    surface: VolatilitySurface,
    engine: HestonCalibrationEngine
) -> Dict[str, any]:
    """
    Generate comprehensive calibration report.
    
    Args:
        results: Calibration results from different methods
        surface: Original volatility surface
        engine: Calibration engine used
        
    Returns:
        Detailed report dictionary
    """
    best_method, best_result = engine.get_best_calibration(results)
    
    report = {
        'summary': {
            'best_method': best_method,
            'success': best_result.success,
            'objective_value': best_result.objective_value,
            'price_rmse': best_result.price_rmse,
            'n_options': best_result.n_options,
            'expiries_used': best_result.expiries_used
        },
        'best_parameters': best_result.params if best_result.success else None,
        'quality_assessment': engine.validate_calibration_quality(best_result, surface),
        'method_comparison': {
            method: {
                'success': result.success,
                'objective_value': result.objective_value,
                'price_rmse': result.price_rmse,
                'feller_condition': result.feller_condition
            }
            for method, result in results.items()
        },
        'surface_summary': surface.summary_stats()
    }
    
    return report