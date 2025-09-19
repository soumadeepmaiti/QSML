import numpy as np
from typing import Dict, List, Callable, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.optimize import minimize, differential_evolution, Bounds
from ..pricers.heston_fft import price_strikes_fft
from ..pricers.heston_integral import heston_option_price_integral
from .surface import SmileSlice, VolatilitySurface

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Types of calibration objectives."""
    PRICE_MSE = "price_mse"                # Mean squared error on prices
    PRICE_RMSE = "price_rmse"              # Root mean squared error on prices
    PRICE_WEIGHTED = "price_weighted"       # Vega-weighted price error
    VOL_MSE = "vol_mse"                    # Mean squared error on implied vols
    VOL_RMSE = "vol_rmse"                  # Root mean squared error on implied vols
    LOG_PRICE = "log_price"                # Log price error (relative)
    MIXED = "mixed"                        # Combination of price and vol errors


class CalibrationMethod(Enum):
    """Optimization algorithms for calibration."""
    SLSQP = "SLSQP"                       # Sequential Least Squares Programming
    L_BFGS_B = "L-BFGS-B"                 # Limited-memory BFGS with bounds
    NELDER_MEAD = "Nelder-Mead"           # Nelder-Mead simplex
    POWELL = "Powell"                      # Powell's method
    DIFFERENTIAL_EVOLUTION = "DE"          # Differential Evolution (global)
    DUAL_ANNEALING = "DA"                 # Dual Annealing (global)


@dataclass
class CalibrationConfig:
    """Configuration for Heston model calibration."""
    
    # Objective function settings
    objective_type: ObjectiveType = ObjectiveType.PRICE_WEIGHTED
    pricing_method: str = "fft"  # "fft" or "integral"
    
    # Optimization settings
    method: CalibrationMethod = CalibrationMethod.L_BFGS_B
    max_iterations: int = 1000
    tolerance: float = 1e-8
    
    # Parameter bounds (will be applied as constraints)
    kappa_bounds: Tuple[float, float] = (0.01, 20.0)
    theta_bounds: Tuple[float, float] = (0.001, 1.0)
    sigma_bounds: Tuple[float, float] = (0.01, 2.0)
    rho_bounds: Tuple[float, float] = (-0.99, 0.99)
    v0_bounds: Tuple[float, float] = (0.001, 1.0)
    
    # Constraints
    enforce_feller: bool = True  # 2*kappa*theta >= sigma^2
    
    # Weighting and filtering
    min_time_to_expiry: float = 1/365  # 1 day minimum
    max_time_to_expiry: float = 2.0    # 2 years maximum
    moneyness_range: Tuple[float, float] = (0.7, 1.3)
    vega_weight_power: float = 0.5     # For vega weighting: w = vega^power
    min_vega_weight: float = 1e-6      # Minimum weight to avoid division issues
    
    # Multi-start for global methods
    n_starts: int = 10                 # Number of random starts
    seed: Optional[int] = 42           # Random seed for reproducibility
    
    # Regularization
    regularization_weight: float = 0.0  # L2 penalty on parameters
    prior_params: Optional[Dict[str, float]] = None  # Prior parameter values


@dataclass
class CalibrationResult:
    """Results from Heston model calibration."""
    
    # Optimized parameters
    params: Dict[str, float]
    
    # Optimization details
    success: bool
    n_iterations: int
    objective_value: float
    optimization_message: str
    
    # Error metrics
    price_rmse: float
    vol_rmse: Optional[float] = None
    max_price_error: float = np.nan
    max_vol_error: Optional[float] = None
    
    # Market data used
    n_options: int = 0
    expiries_used: List[str] = None
    
    # Model diagnostics
    feller_condition: bool = False
    parameter_bounds_active: Dict[str, bool] = None
    
    # Pricing comparison
    market_prices: Optional[np.ndarray] = None
    model_prices: Optional[np.ndarray] = None
    price_errors: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None


class HestonCalibrator:
    """
    Calibration engine for the Heston stochastic volatility model.
    
    Supports multiple objective functions, optimization algorithms,
    and market data filtering/weighting schemes.
    """
    
    def __init__(self, config: CalibrationConfig = None):
        """Initialize calibrator with configuration."""
        self.config = config or CalibrationConfig()
        self.logger = logger
        
        # Parameter mapping for optimization
        self.param_names = ['kappa', 'theta', 'sigma', 'rho', 'v0']
        
        # Bounds for scipy optimizers
        self.bounds = Bounds(
            lb=np.array([
                self.config.kappa_bounds[0],
                self.config.theta_bounds[0], 
                self.config.sigma_bounds[0],
                self.config.rho_bounds[0],
                self.config.v0_bounds[0]
            ]),
            ub=np.array([
                self.config.kappa_bounds[1],
                self.config.theta_bounds[1],
                self.config.sigma_bounds[1], 
                self.config.rho_bounds[1],
                self.config.v0_bounds[1]
            ])
        )
    
    def calibrate_slice(
        self,
        slice_obj: SmileSlice,
        initial_guess: Optional[Dict[str, float]] = None
    ) -> CalibrationResult:
        """
        Calibrate Heston parameters to a single smile slice.
        
        Args:
            slice_obj: Market data for single expiry
            initial_guess: Initial parameter values
            
        Returns:
            CalibrationResult with optimized parameters
        """
        # Filter and prepare data
        filtered_slice = self._filter_slice(slice_obj)
        
        if len(filtered_slice.strikes) == 0:
            self.logger.warning("No valid options after filtering")
            return self._empty_result()
        
        # Extract market data
        market_data = self._extract_market_data(filtered_slice)
        
        # Set up initial guess
        x0 = self._get_initial_guess(initial_guess, filtered_slice)
        
        # Create objective function
        objective_fn = self._create_objective_function(filtered_slice, market_data)
        
        # Run optimization
        result = self._optimize(objective_fn, x0)
        
        # Process results
        calibration_result = self._process_result(result, filtered_slice, market_data)
        
        return calibration_result
    
    def calibrate_surface(
        self,
        surface: VolatilitySurface,
        joint_calibration: bool = False,
        initial_guess: Optional[Dict[str, float]] = None
    ) -> Union[CalibrationResult, Dict[str, CalibrationResult]]:
        """
        Calibrate Heston parameters to volatility surface.
        
        Args:
            surface: Complete volatility surface
            joint_calibration: If True, calibrate all expiries jointly
            initial_guess: Initial parameter values
            
        Returns:
            Single result (joint) or dict of results by expiry
        """
        if joint_calibration:
            return self._calibrate_surface_joint(surface, initial_guess)
        else:
            return self._calibrate_surface_slice_by_slice(surface, initial_guess)
    
    def _calibrate_surface_joint(
        self,
        surface: VolatilitySurface,
        initial_guess: Optional[Dict[str, float]] = None
    ) -> CalibrationResult:
        """Joint calibration across all expiries."""
        
        # Combine all filtered slices
        all_market_data = []
        all_slices = []
        
        for slice_obj in surface.slices.values():
            filtered_slice = self._filter_slice(slice_obj)
            if len(filtered_slice.strikes) > 0:
                market_data = self._extract_market_data(filtered_slice)
                all_market_data.append(market_data)
                all_slices.append(filtered_slice)
        
        if not all_market_data:
            self.logger.warning("No valid data for joint calibration")
            return self._empty_result()
        
        # Set up initial guess
        x0 = self._get_initial_guess(initial_guess, all_slices[0])
        
        # Create joint objective function
        objective_fn = self._create_joint_objective_function(all_slices, all_market_data)
        
        # Run optimization
        result = self._optimize(objective_fn, x0)
        
        # Process results
        combined_data = self._combine_market_data(all_market_data)
        calibration_result = self._process_joint_result(result, all_slices, combined_data)
        
        return calibration_result
    
    def _calibrate_surface_slice_by_slice(
        self,
        surface: VolatilitySurface,
        initial_guess: Optional[Dict[str, float]] = None
    ) -> Dict[str, CalibrationResult]:
        """Calibrate each expiry independently."""
        
        results = {}
        current_guess = initial_guess
        
        # Sort expiries by time to expiry
        sorted_expiries = sorted(
            surface.slices.items(),
            key=lambda x: x[1].time_to_expiry
        )
        
        for expiry_key, slice_obj in sorted_expiries:
            self.logger.info(f"Calibrating expiry {expiry_key}")
            
            result = self.calibrate_slice(slice_obj, current_guess)
            results[str(expiry_key)] = result
            
            # Use previous result as next initial guess
            if result.success:
                current_guess = result.params
        
        return results
    
    def _filter_slice(self, slice_obj: SmileSlice) -> SmileSlice:
        """Apply filtering criteria to slice."""
        
        # Time filter
        if (slice_obj.time_to_expiry < self.config.min_time_to_expiry or 
            slice_obj.time_to_expiry > self.config.max_time_to_expiry):
            self.logger.debug(f"Filtering slice with T={slice_obj.time_to_expiry}")
            return SmileSlice(
                expiry_date=slice_obj.expiry_date,
                time_to_expiry=slice_obj.time_to_expiry,
                spot=slice_obj.spot,
                forward=slice_obj.forward,
                risk_free_rate=slice_obj.risk_free_rate,
                dividend_yield=slice_obj.dividend_yield,
                strikes=np.array([])
            )
        
        # Apply moneyness filter
        filtered = slice_obj.filter_by_moneyness(
            min_moneyness=self.config.moneyness_range[0],
            max_moneyness=self.config.moneyness_range[1]
        )
        
        # Apply IV filter
        filtered = filtered.filter_valid_ivs()
        
        return filtered
    
    def _extract_market_data(self, slice_obj: SmileSlice) -> Dict[str, np.ndarray]:
        """Extract market prices and compute weights."""
        
        market_prices = []
        strikes = []
        
        # Use calls for OTM/ATM, puts for ITM
        for i, K in enumerate(slice_obj.strikes):
            if K >= slice_obj.forward and slice_obj.calls_mid is not None:
                market_prices.append(slice_obj.calls_mid[i])
                strikes.append(K)
            elif K < slice_obj.forward and slice_obj.puts_mid is not None:
                market_prices.append(slice_obj.puts_mid[i])
                strikes.append(K)
            elif slice_obj.calls_mid is not None:
                market_prices.append(slice_obj.calls_mid[i])
                strikes.append(K)
        
        market_prices = np.array(market_prices)
        strikes = np.array(strikes)
        
        # Compute weights
        weights = self._compute_weights(slice_obj, strikes, market_prices)
        
        return {
            'market_prices': market_prices,
            'strikes': strikes,
            'weights': weights,
            'n_options': len(strikes)
        }
    
    def _compute_weights(
        self,
        slice_obj: SmileSlice,
        strikes: np.ndarray,
        market_prices: np.ndarray
    ) -> np.ndarray:
        """Compute weights for objective function."""
        
        if self.config.objective_type == ObjectiveType.PRICE_WEIGHTED:
            # Vega weighting
            from ..pricers.bs import bs_vega
            
            weights = []
            for i, K in enumerate(strikes):
                # Use implied vol if available, otherwise estimate
                if slice_obj.implied_vols is not None:
                    vol_idx = np.where(slice_obj.strikes == K)[0]
                    if len(vol_idx) > 0 and not np.isnan(slice_obj.implied_vols[vol_idx[0]]):
                        vol = slice_obj.implied_vols[vol_idx[0]]
                    else:
                        vol = slice_obj.get_atm_vol()
                else:
                    vol = 0.2  # Default vol estimate
                
                try:
                    vega = bs_vega(
                        S=slice_obj.spot,
                        K=K,
                        T=slice_obj.time_to_expiry,
                        r=slice_obj.risk_free_rate,
                        q=slice_obj.dividend_yield,
                        sigma=vol
                    )
                    weight = max(vega ** self.config.vega_weight_power, self.config.min_vega_weight)
                    weights.append(weight)
                except:
                    weights.append(self.config.min_vega_weight)
            
            weights = np.array(weights)
            
        else:
            # Equal weighting
            weights = np.ones(len(strikes))
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return weights
    
    def _get_initial_guess(
        self,
        initial_guess: Optional[Dict[str, float]],
        slice_obj: SmileSlice
    ) -> np.ndarray:
        """Get initial parameter guess."""
        
        if initial_guess is not None:
            x0 = np.array([initial_guess[param] for param in self.param_names])
        else:
            # Default initial guess
            atm_vol = slice_obj.get_atm_vol()
            if np.isnan(atm_vol):
                atm_vol = 0.2
            
            x0 = np.array([
                2.0,           # kappa
                atm_vol**2,    # theta  
                0.3,           # sigma
                -0.5,          # rho
                atm_vol**2     # v0
            ])
        
        # Ensure within bounds
        x0 = np.clip(x0, self.bounds.lb, self.bounds.ub)
        
        return x0
    
    def _create_objective_function(
        self,
        slice_obj: SmileSlice,
        market_data: Dict[str, np.ndarray]
    ) -> Callable:
        """Create objective function for optimization."""
        
        def objective(x: np.ndarray) -> float:
            # Convert to parameter dict
            params = dict(zip(self.param_names, x))
            
            # Check Feller condition if enforced
            if self.config.enforce_feller:
                if 2 * params['kappa'] * params['theta'] < params['sigma']**2:
                    return 1e10  # Large penalty
            
            # Compute model prices
            try:
                model_prices = self._compute_model_prices(slice_obj, market_data['strikes'], params)
            except Exception as e:
                self.logger.debug(f"Pricing failed: {e}")
                return 1e10
            
            # Compute objective
            if self.config.objective_type in [ObjectiveType.PRICE_MSE, ObjectiveType.PRICE_RMSE, ObjectiveType.PRICE_WEIGHTED]:
                errors = model_prices - market_data['market_prices']
                weighted_errors = errors * market_data['weights']
                mse = np.sum(weighted_errors**2)
                
                if self.config.objective_type == ObjectiveType.PRICE_RMSE:
                    return np.sqrt(mse)
                else:
                    return mse
            
            elif self.config.objective_type == ObjectiveType.LOG_PRICE:
                log_errors = np.log(model_prices) - np.log(market_data['market_prices'])
                weighted_errors = log_errors * market_data['weights']
                return np.sum(weighted_errors**2)
            
            # Add regularization if specified
            if self.config.regularization_weight > 0:
                if self.config.prior_params is not None:
                    prior = np.array([self.config.prior_params.get(param, 0) for param in self.param_names])
                    reg_penalty = self.config.regularization_weight * np.sum((x - prior)**2)
                    return mse + reg_penalty
            
            return mse
        
        return objective
    
    def _create_joint_objective_function(
        self,
        slices: List[SmileSlice],
        market_data_list: List[Dict[str, np.ndarray]]
    ) -> Callable:
        """Create joint objective function for multiple expiries."""
        
        def objective(x: np.ndarray) -> float:
            params = dict(zip(self.param_names, x))
            
            # Check Feller condition
            if self.config.enforce_feller:
                if 2 * params['kappa'] * params['theta'] < params['sigma']**2:
                    return 1e10
            
            total_error = 0.0
            total_weight = 0.0
            
            for slice_obj, market_data in zip(slices, market_data_list):
                try:
                    model_prices = self._compute_model_prices(slice_obj, market_data['strikes'], params)
                    errors = model_prices - market_data['market_prices']
                    weighted_errors = errors * market_data['weights']
                    slice_error = np.sum(weighted_errors**2)
                    
                    # Weight by number of options in slice
                    slice_weight = market_data['n_options']
                    total_error += slice_error * slice_weight
                    total_weight += slice_weight
                    
                except Exception as e:
                    self.logger.debug(f"Pricing failed for slice: {e}")
                    return 1e10
            
            if total_weight > 0:
                return total_error / total_weight
            else:
                return 1e10
        
        return objective
    
    def _compute_model_prices(
        self,
        slice_obj: SmileSlice,
        strikes: np.ndarray,
        params: Dict[str, float]
    ) -> np.ndarray:
        """Compute model prices for given strikes and parameters."""
        
        if self.config.pricing_method == "fft":
            # Use FFT pricing
            call_prices = price_strikes_fft(
                S=slice_obj.spot,
                r=slice_obj.risk_free_rate,
                q=slice_obj.dividend_yield,
                T=slice_obj.time_to_expiry,
                heston_params=params,
                strikes=strikes
            )
            
            # Convert calls to puts where appropriate
            model_prices = []
            for i, K in enumerate(strikes):
                if K >= slice_obj.forward:
                    # Use call price
                    model_prices.append(call_prices[i])
                else:
                    # Convert to put using put-call parity
                    call_price = call_prices[i]
                    put_price = (call_price - slice_obj.spot * np.exp(-slice_obj.dividend_yield * slice_obj.time_to_expiry) + 
                                K * np.exp(-slice_obj.risk_free_rate * slice_obj.time_to_expiry))
                    model_prices.append(put_price)
            
            return np.array(model_prices)
        
        elif self.config.pricing_method == "integral":
            # Use semi-closed form integration
            model_prices = []
            for K in strikes:
                if K >= slice_obj.forward:
                    option_type = "call"
                else:
                    option_type = "put"
                
                price = heston_option_price_integral(
                    S=slice_obj.spot,
                    K=K,
                    r=slice_obj.risk_free_rate,
                    q=slice_obj.dividend_yield,
                    T=slice_obj.time_to_expiry,
                    params=params,
                    option_type=option_type
                )
                model_prices.append(price)
            
            return np.array(model_prices)
        
        else:
            raise ValueError(f"Unknown pricing method: {self.config.pricing_method}")
    
    def _optimize(self, objective_fn: Callable, x0: np.ndarray) -> Any:
        """Run optimization."""
        
        if self.config.method == CalibrationMethod.DIFFERENTIAL_EVOLUTION:
            # Global optimization
            result = differential_evolution(
                objective_fn,
                bounds=list(zip(self.bounds.lb, self.bounds.ub)),
                maxiter=self.config.max_iterations,
                tol=self.config.tolerance,
                seed=self.config.seed,
                workers=1  # Avoid multiprocessing issues
            )
        
        else:
            # Local optimization
            method_name = self.config.method.value
            
            result = minimize(
                objective_fn,
                x0=x0,
                method=method_name,
                bounds=self.bounds if method_name in ['L-BFGS-B', 'SLSQP'] else None,
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': self.config.tolerance
                }
            )
        
        return result
    
    def _process_result(
        self,
        result: Any,
        slice_obj: SmileSlice,
        market_data: Dict[str, np.ndarray]
    ) -> CalibrationResult:
        """Process optimization result."""
        
        # Extract optimized parameters
        params = dict(zip(self.param_names, result.x))
        
        # Compute final model prices and errors
        try:
            model_prices = self._compute_model_prices(slice_obj, market_data['strikes'], params)
            price_errors = model_prices - market_data['market_prices']
            price_rmse = np.sqrt(np.mean(price_errors**2))
            max_price_error = np.max(np.abs(price_errors))
        except:
            model_prices = None
            price_errors = None
            price_rmse = np.nan
            max_price_error = np.nan
        
        # Check Feller condition
        feller_condition = 2 * params['kappa'] * params['theta'] >= params['sigma']**2
        
        # Check if bounds are active
        bounds_active = {}
        for i, param in enumerate(self.param_names):
            tol = 1e-6
            bounds_active[param] = (
                abs(result.x[i] - self.bounds.lb[i]) < tol or
                abs(result.x[i] - self.bounds.ub[i]) < tol
            )
        
        return CalibrationResult(
            params=params,
            success=result.success,
            n_iterations=getattr(result, 'nit', 0),
            objective_value=result.fun,
            optimization_message=getattr(result, 'message', ''),
            price_rmse=price_rmse,
            max_price_error=max_price_error,
            n_options=market_data['n_options'],
            expiries_used=[str(slice_obj.expiry_date)],
            feller_condition=feller_condition,
            parameter_bounds_active=bounds_active,
            market_prices=market_data['market_prices'],
            model_prices=model_prices,
            price_errors=price_errors,
            weights=market_data['weights']
        )
    
    def _process_joint_result(
        self,
        result: Any,
        slices: List[SmileSlice],
        combined_data: Dict[str, np.ndarray]
    ) -> CalibrationResult:
        """Process joint calibration result."""
        
        params = dict(zip(self.param_names, result.x))
        
        # Compute errors across all slices
        all_errors = []
        all_model_prices = []
        
        for slice_obj in slices:
            try:
                slice_data = self._extract_market_data(slice_obj)
                model_prices = self._compute_model_prices(slice_obj, slice_data['strikes'], params)
                errors = model_prices - slice_data['market_prices']
                all_errors.extend(errors)
                all_model_prices.extend(model_prices)
            except:
                pass
        
        if all_errors:
            price_rmse = np.sqrt(np.mean(np.array(all_errors)**2))
            max_price_error = np.max(np.abs(all_errors))
        else:
            price_rmse = np.nan
            max_price_error = np.nan
        
        feller_condition = 2 * params['kappa'] * params['theta'] >= params['sigma']**2
        
        bounds_active = {}
        for i, param in enumerate(self.param_names):
            tol = 1e-6
            bounds_active[param] = (
                abs(result.x[i] - self.bounds.lb[i]) < tol or
                abs(result.x[i] - self.bounds.ub[i]) < tol
            )
        
        return CalibrationResult(
            params=params,
            success=result.success,
            n_iterations=getattr(result, 'nit', 0),
            objective_value=result.fun,
            optimization_message=getattr(result, 'message', ''),
            price_rmse=price_rmse,
            max_price_error=max_price_error,
            n_options=combined_data['n_options'],
            expiries_used=[str(slice_obj.expiry_date) for slice_obj in slices],
            feller_condition=feller_condition,
            parameter_bounds_active=bounds_active,
            market_prices=combined_data['market_prices'],
            model_prices=np.array(all_model_prices) if all_model_prices else None,
            price_errors=np.array(all_errors) if all_errors else None,
            weights=combined_data['weights']
        )
    
    def _combine_market_data(self, data_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine market data from multiple slices."""
        combined = {
            'market_prices': np.concatenate([d['market_prices'] for d in data_list]),
            'strikes': np.concatenate([d['strikes'] for d in data_list]),
            'weights': np.concatenate([d['weights'] for d in data_list]),
            'n_options': sum(d['n_options'] for d in data_list)
        }
        
        # Renormalize combined weights
        combined['weights'] = combined['weights'] / np.sum(combined['weights'])
        
        return combined
    
    def _empty_result(self) -> CalibrationResult:
        """Return empty calibration result for failed cases."""
        return CalibrationResult(
            params={param: np.nan for param in self.param_names},
            success=False,
            n_iterations=0,
            objective_value=np.inf,
            optimization_message="No valid data",
            price_rmse=np.nan,
            feller_condition=False,
            parameter_bounds_active={param: False for param in self.param_names}
        )