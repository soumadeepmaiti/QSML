import numpy as np
from typing import Tuple, Literal, Callable, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def fd_greek(
    price_fn: Callable,
    args: Tuple,
    h: float,
    kind: Literal["delta", "vega", "gamma", "theta", "rho"],
    param_idx: int = None
) -> float:
    """
    Finite-difference Greeks for any pricing function.
    
    Args:
        price_fn: Pricing function with signature price_fn(*args)
        args: Tuple of arguments for the pricing function
        h: Step size for finite differences
        kind: Type of Greek to compute
        param_idx: Index of parameter to differentiate with respect to
        
    Returns:
        Greek value
    """
    if kind == "delta":
        return _fd_delta(price_fn, args, h)
    elif kind == "vega":
        return _fd_vega(price_fn, args, h)
    elif kind == "gamma":
        return _fd_gamma(price_fn, args, h)
    elif kind == "theta":
        return _fd_theta(price_fn, args, h)
    elif kind == "rho":
        return _fd_rho(price_fn, args, h)
    else:
        raise ValueError(f"Unknown Greek type: {kind}")


def _fd_delta(price_fn: Callable, args: Tuple, h: float) -> float:
    """Finite difference delta (derivative w.r.t. spot price)."""
    # Assume spot price is the first argument
    args_list = list(args)
    S = args_list[0]
    
    # Central difference
    args_up = args_list.copy()
    args_down = args_list.copy()
    args_up[0] = S * (1 + h)
    args_down[0] = S * (1 - h)
    
    try:
        price_up = price_fn(*args_up)
        price_down = price_fn(*args_down)
        delta = (price_up - price_down) / (2 * S * h)
        return delta
    except Exception as e:
        logger.warning(f"Delta calculation failed: {e}")
        return np.nan


def _fd_vega(price_fn: Callable, args: Tuple, h: float) -> float:
    """Finite difference vega (derivative w.r.t. volatility)."""
    # Need to identify volatility parameter - this depends on the function signature
    # For BS functions, sigma is typically the last argument
    # For Heston functions, we need to modify the params dict
    
    args_list = list(args)
    
    # Try to find volatility parameter
    if isinstance(args_list[-1], dict) and 'sigma' in args_list[-1]:
        # Heston case - modify params dict
        params = args_list[-1].copy()
        sigma = params['sigma']
        
        params_up = params.copy()
        params_down = params.copy()
        params_up['sigma'] = sigma + h
        params_down['sigma'] = sigma - h
        
        args_up = args_list[:-1] + [params_up]
        args_down = args_list[:-1] + [params_down]
        
        try:
            price_up = price_fn(*args_up)
            price_down = price_fn(*args_down)
            vega = (price_up - price_down) / (2 * h)
            return vega
        except Exception as e:
            logger.warning(f"Vega calculation failed: {e}")
            return np.nan
    
    else:
        # BS case - assume sigma is the last argument
        sigma = args_list[-1]
        
        args_up = args_list[:-1] + [sigma + h]
        args_down = args_list[:-1] + [sigma - h]
        
        try:
            price_up = price_fn(*args_up)
            price_down = price_fn(*args_down)
            vega = (price_up - price_down) / (2 * h)
            return vega
        except Exception as e:
            logger.warning(f"Vega calculation failed: {e}")
            return np.nan


def _fd_gamma(price_fn: Callable, args: Tuple, h: float) -> float:
    """Finite difference gamma (second derivative w.r.t. spot price)."""
    args_list = list(args)
    S = args_list[0]
    
    # Second-order central difference
    args_up = args_list.copy()
    args_center = args_list.copy()
    args_down = args_list.copy()
    
    args_up[0] = S * (1 + h)
    args_down[0] = S * (1 - h)
    
    try:
        price_up = price_fn(*args_up)
        price_center = price_fn(*args_center)
        price_down = price_fn(*args_down)
        
        gamma = (price_up - 2 * price_center + price_down) / (S * h)**2
        return gamma
    except Exception as e:
        logger.warning(f"Gamma calculation failed: {e}")
        return np.nan


def _fd_theta(price_fn: Callable, args: Tuple, h: float) -> float:
    """Finite difference theta (derivative w.r.t. time)."""
    # Assume time to expiration T is typically the 5th argument (after S, K, r, q)
    args_list = list(args)
    
    # Find T parameter
    if len(args_list) >= 5:
        T_idx = 4  # Common position for T
    else:
        logger.warning("Cannot identify time parameter for theta calculation")
        return np.nan
    
    T = args_list[T_idx]
    
    if T <= h:
        # Too close to expiration, use backward difference
        args_back = args_list.copy()
        args_back[T_idx] = max(T - h, 1e-6)
        
        try:
            price_current = price_fn(*args_list)
            price_back = price_fn(*args_back)
            theta = -(price_current - price_back) / h  # Negative because time decreases
            return theta / 365  # Convert to per-day
        except Exception as e:
            logger.warning(f"Theta calculation failed: {e}")
            return np.nan
    else:
        # Forward difference
        args_forward = args_list.copy()
        args_forward[T_idx] = T - h
        
        try:
            price_current = price_fn(*args_list)
            price_forward = price_fn(*args_forward)
            theta = -(price_forward - price_current) / h  # Negative because time decreases
            return theta / 365  # Convert to per-day
        except Exception as e:
            logger.warning(f"Theta calculation failed: {e}")
            return np.nan


def _fd_rho(price_fn: Callable, args: Tuple, h: float) -> float:
    """Finite difference rho (derivative w.r.t. risk-free rate)."""
    # Assume risk-free rate r is typically the 3rd argument
    args_list = list(args)
    
    if len(args_list) >= 3:
        r_idx = 2  # Common position for r
    else:
        logger.warning("Cannot identify risk-free rate parameter for rho calculation")
        return np.nan
    
    r = args_list[r_idx]
    
    args_up = args_list.copy()
    args_down = args_list.copy()
    args_up[r_idx] = r + h
    args_down[r_idx] = r - h
    
    try:
        price_up = price_fn(*args_up)
        price_down = price_fn(*args_down)
        rho = (price_up - price_down) / (2 * h)
        return rho
    except Exception as e:
        logger.warning(f"Rho calculation failed: {e}")
        return np.nan


def adaptive_step_size(
    price_fn: Callable,
    args: Tuple,
    kind: str,
    initial_h: float = 0.01,
    target_precision: float = 1e-6,
    max_iterations: int = 10
) -> Tuple[float, float]:
    """
    Adaptively determine optimal step size for finite differences.
    
    Args:
        price_fn: Pricing function
        args: Function arguments
        kind: Type of Greek
        initial_h: Initial step size
        target_precision: Target precision for Greek calculation
        max_iterations: Maximum iterations for adaptation
        
    Returns:
        Tuple of (optimal_step_size, greek_value)
    """
    h = initial_h
    prev_greek = None
    
    for i in range(max_iterations):
        greek = fd_greek(price_fn, args, h, kind)
        
        if prev_greek is not None and not np.isnan(greek) and not np.isnan(prev_greek):
            error = abs(greek - prev_greek)
            if error < target_precision:
                return h, greek
        
        prev_greek = greek
        h = h / 2  # Halve step size
    
    logger.warning(f"Adaptive step size did not converge for {kind}")
    return h, greek if not np.isnan(greek) else 0.0


def all_greeks_fd(
    price_fn: Callable,
    args: Tuple,
    h_dict: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Calculate all Greeks using finite differences.
    
    Args:
        price_fn: Pricing function
        args: Function arguments
        h_dict: Dictionary of step sizes for each Greek
        
    Returns:
        Dictionary with all Greeks
    """
    if h_dict is None:
        h_dict = {
            'delta': 0.01,
            'vega': 0.01,
            'gamma': 0.01,
            'theta': 1/365,  # One day
            'rho': 0.0001    # 1 basis point
        }
    
    greeks = {}
    
    for greek_name in ['delta', 'vega', 'gamma', 'theta', 'rho']:
        try:
            h = h_dict.get(greek_name, 0.01)
            greek_value = fd_greek(price_fn, args, h, greek_name)
            greeks[greek_name] = greek_value
        except Exception as e:
            logger.warning(f"Failed to calculate {greek_name}: {e}")
            greeks[greek_name] = np.nan
    
    return greeks


def heston_greeks_fd(
    S0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    params: Dict[str, float],
    option_type: str = "call",
    method: str = "fft"
) -> Dict[str, float]:
    """
    Calculate Heston Greeks using finite differences.
    
    Args:
        S0: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Dividend yield
        T: Time to expiration
        params: Heston parameters
        option_type: "call" or "put"
        method: "fft" or "integral"
        
    Returns:
        Dictionary with Greeks
    """
    # Choose pricing function based on method
    if method == "fft":
        from .heston_fft import price_strikes_fft
        
        def price_fn(S, K, r, q, T, heston_params):
            if option_type == "call":
                return price_strikes_fft(S, r, q, T, heston_params, np.array([K]))[0]
            else:
                from .heston_fft import fft_put_prices
                return fft_put_prices(S, r, q, T, heston_params, np.array([K]))[0]
    
    elif method == "integral":
        from .heston_integral import heston_option_price_integral
        
        def price_fn(S, K, r, q, T, heston_params):
            return heston_option_price_integral(S, K, r, q, T, heston_params, option_type)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate standard Greeks
    args = (S0, K, r, q, T, params)
    standard_greeks = all_greeks_fd(price_fn, args)
    
    # Calculate Heston-specific Greeks (sensitivities to Heston parameters)
    heston_param_greeks = {}
    param_step_sizes = {
        'kappa': 0.01,
        'theta': 0.001,
        'sigma': 0.01,
        'rho': 0.01,
        'v0': 0.001
    }
    
    for param_name, step_size in param_step_sizes.items():
        try:
            params_up = params.copy()
            params_down = params.copy()
            params_up[param_name] = params[param_name] + step_size
            params_down[param_name] = params[param_name] - step_size
            
            price_up = price_fn(S0, K, r, q, T, params_up)
            price_down = price_fn(S0, K, r, q, T, params_down)
            
            greek_value = (price_up - price_down) / (2 * step_size)
            heston_param_greeks[f"d_d{param_name}"] = greek_value
            
        except Exception as e:
            logger.warning(f"Failed to calculate sensitivity to {param_name}: {e}")
            heston_param_greeks[f"d_d{param_name}"] = np.nan
    
    # Combine all Greeks
    all_greeks = {**standard_greeks, **heston_param_greeks}
    
    return all_greeks


def greek_scaling_factors() -> Dict[str, float]:
    """
    Return typical scaling factors for Greeks to make them more interpretable.
    
    Returns:
        Dictionary with scaling factors
    """
    return {
        'delta': 1.0,           # Per $1 move in underlying
        'gamma': 1.0,           # Per $1 move in underlying (second order)
        'vega': 0.01,           # Per 1% change in volatility
        'theta': 1.0,           # Per day (already scaled)
        'rho': 0.0001,          # Per 1 basis point change in rate
        'd_dkappa': 0.1,        # Per 0.1 change in kappa
        'd_dtheta': 0.01,       # Per 1% change in theta
        'd_dsigma': 0.01,       # Per 1% change in sigma
        'd_drho': 0.01,         # Per 1% change in rho
        'd_dv0': 0.01           # Per 1% change in v0
    }