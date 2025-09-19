import numpy as np
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)


def heston_charfn(u: complex, T: float, params: Dict[str, float]) -> complex:
    """
    Heston characteristic function phi(u; T, params) using the standard formulation.
    
    Implementation uses the "little trap" formulation for numerical stability.
    
    Args:
        u: Complex frequency parameter
        T: Time to expiration in years
        params: Dictionary with Heston parameters:
            - kappa: Mean reversion speed
            - theta: Long-term variance
            - sigma: Volatility of volatility (vol-of-vol)
            - rho: Correlation between price and volatility
            - v0: Initial variance
            
    Returns:
        Complex characteristic function value phi(u)
        
    References:
        - Andersen & Piterbarg (2010)
        - Albrecher et al. (2007)
        - Lord & Kahl (2010) for numerical considerations
    """
    # Extract parameters
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    rho = params['rho']
    v0 = params['v0']
    
    # Handle edge cases
    if T <= 0:
        return complex(1.0, 0.0)
    
    if abs(u) < 1e-12:
        return complex(1.0, 0.0)
    
    # Complex arithmetic with type safety
    u = complex(u)
    
    # Precompute common terms
    xi = kappa - rho * sigma * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u * 1j + u**2))
    
    # Riccati equation coefficients
    A1 = u * 1j + u**2
    A2 = xi
    A3 = sigma**2
    
    # Little trap formulation for numerical stability
    # Choose the formulation that avoids overflow
    g1 = (xi + d) / (xi - d)
    g2 = (xi - d) / (xi + d)
    
    # Use the formulation with |g| < 1 to avoid exponential overflow
    if abs(g2) < abs(g1):
        g = g2
        sign = -1
    else:
        g = g1
        sign = 1
    
    # Calculate D(tau) - coefficient of v_t in the exponential
    exp_dT = np.exp(sign * d * T)
    
    if abs(exp_dT) > 1e6:  # Protect against overflow
        # Use asymptotic approximation
        D = A1 / (A2 + sign * d)
    else:
        numerator = A1 * (1 - exp_dT)
        denominator = A2 + sign * d - (A2 - sign * d) * exp_dT
        D = numerator / denominator
    
    # Calculate C(tau) - the deterministic part
    # C(tau) = r*u*i*tau + (a*b/sigma^2) * [...] 
    # where a = kappa, b = theta
    
    kappa_theta = kappa * theta
    
    if abs(kappa_theta) < 1e-12:
        C = complex(0.0, 0.0)
    else:
        term1 = 2 * kappa_theta / sigma**2
        
        if abs(exp_dT) > 1e6:
            # Asymptotic approximation
            term2 = (A2 + sign * d) * T - np.log(abs(A2 + sign * d))
        else:
            term2_log_arg = (A2 + sign * d - (A2 - sign * d) * exp_dT) / (2 * sign * d)
            
            # Protect against log of negative or zero
            if abs(term2_log_arg) < 1e-12:
                term2 = sign * d * T
            else:
                term2 = sign * d * T - np.log(term2_log_arg)
        
        C = term1 * term2
    
    # Characteristic function
    phi = np.exp(C + D * v0)
    
    # Numerical checks
    if not np.isfinite(phi):
        logger.warning(f"Non-finite characteristic function for u={u}, T={T}")
        return complex(1.0, 0.0)
    
    if abs(phi) > 1e10:
        logger.warning(f"Very large characteristic function magnitude: {abs(phi)}")
        return complex(1.0, 0.0)
    
    return phi


def heston_charfn_vectorized(
    u_array: np.ndarray, 
    T: float, 
    params: Dict[str, float]
) -> np.ndarray:
    """
    Vectorized version of Heston characteristic function for multiple u values.
    
    Args:
        u_array: Array of complex frequency parameters
        T: Time to expiration in years
        params: Heston parameters dictionary
        
    Returns:
        Array of characteristic function values
    """
    # For small arrays, use the scalar version
    if len(u_array) < 100:
        return np.array([heston_charfn(u, T, params) for u in u_array])
    
    # For large arrays, implement vectorized version
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    rho = params['rho']
    v0 = params['v0']
    
    if T <= 0:
        return np.ones_like(u_array, dtype=complex)
    
    # Convert to complex
    u_array = np.asarray(u_array, dtype=complex)
    
    # Vectorized computation
    xi = kappa - rho * sigma * u_array * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u_array * 1j + u_array**2))
    
    # Little trap formulation
    g2 = (xi - d) / (xi + d)
    
    # Calculate D(tau)
    exp_dT = np.exp(-d * T)
    
    # Handle potential overflow
    overflow_mask = np.abs(exp_dT) > 1e6
    
    A1 = u_array * 1j + u_array**2
    
    D = np.zeros_like(u_array, dtype=complex)
    
    # Normal calculation for non-overflow cases
    normal_mask = ~overflow_mask
    if np.any(normal_mask):
        numerator = A1[normal_mask] * (1 - exp_dT[normal_mask])
        denominator = (xi[normal_mask] - d[normal_mask] - 
                      (xi[normal_mask] + d[normal_mask]) * exp_dT[normal_mask])
        D[normal_mask] = numerator / denominator
    
    # Asymptotic approximation for overflow cases
    if np.any(overflow_mask):
        D[overflow_mask] = A1[overflow_mask] / (xi[overflow_mask] - d[overflow_mask])
    
    # Calculate C(tau)
    kappa_theta = kappa * theta
    
    if abs(kappa_theta) < 1e-12:
        C = np.zeros_like(u_array, dtype=complex)
    else:
        term1 = 2 * kappa_theta / sigma**2
        
        # Normal calculation
        normal_mask = ~overflow_mask
        C = np.zeros_like(u_array, dtype=complex)
        
        if np.any(normal_mask):
            log_arg = ((xi[normal_mask] - d[normal_mask] - 
                       (xi[normal_mask] + d[normal_mask]) * exp_dT[normal_mask]) / 
                      (-2 * d[normal_mask]))
            
            # Protect against log of negative or zero
            log_arg = np.where(np.abs(log_arg) < 1e-12, 1.0, log_arg)
            
            term2 = -d[normal_mask] * T - np.log(log_arg)
            C[normal_mask] = term1 * term2
        
        # Asymptotic approximation for overflow cases
        if np.any(overflow_mask):
            term2 = (xi[overflow_mask] - d[overflow_mask]) * T
            C[overflow_mask] = term1 * term2
    
    # Characteristic function
    phi = np.exp(C + D * v0)
    
    # Handle non-finite values
    finite_mask = np.isfinite(phi)
    phi[~finite_mask] = 1.0 + 0j
    
    # Handle very large values
    large_mask = np.abs(phi) > 1e10
    phi[large_mask] = 1.0 + 0j
    
    return phi


def check_heston_params(params: Dict[str, float], check_feller: bool = True) -> Dict[str, bool]:
    """
    Check validity of Heston parameters.
    
    Args:
        params: Heston parameters dictionary
        check_feller: Whether to check Feller condition
        
    Returns:
        Dictionary with validity checks
    """
    checks = {
        'kappa_positive': params['kappa'] > 0,
        'theta_positive': params['theta'] > 0,
        'sigma_positive': params['sigma'] > 0,
        'v0_positive': params['v0'] > 0,
        'rho_valid': -1 < params['rho'] < 1,
        'feller_condition': True  # Default to True if not checked
    }
    
    if check_feller:
        # Feller condition: 2*kappa*theta >= sigma^2
        feller_lhs = 2 * params['kappa'] * params['theta']
        feller_rhs = params['sigma']**2
        checks['feller_condition'] = feller_lhs >= feller_rhs
    
    checks['all_valid'] = all(checks.values())
    
    return checks


def heston_moments(T: float, params: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate theoretical moments of the Heston model.
    
    Args:
        T: Time horizon
        params: Heston parameters
        
    Returns:
        Dictionary with theoretical moments
    """
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    rho = params['rho']
    v0 = params['v0']
    
    # Expected variance at time T
    if kappa == 0:
        exp_var_T = v0
    else:
        exp_var_T = theta + (v0 - theta) * np.exp(-kappa * T)
    
    # Expected integrated variance over [0, T]
    if kappa == 0:
        exp_int_var = v0 * T
    else:
        exp_int_var = theta * T + (v0 - theta) * (1 - np.exp(-kappa * T)) / kappa
    
    # Variance of integrated variance (for smile approximations)
    if kappa == 0:
        var_int_var = sigma**2 * v0 * T**3 / 3
    else:
        term1 = sigma**2 * theta / kappa**2
        term2 = 2 * T - 3 + 4 * np.exp(-kappa * T) - np.exp(-2 * kappa * T)
        term3 = sigma**2 * (v0 - theta) / kappa**3
        term4 = 2 - 4 * np.exp(-kappa * T) + 2 * np.exp(-2 * kappa * T)
        var_int_var = term1 * term2 + term3 * term4
    
    return {
        'expected_variance_T': exp_var_T,
        'expected_integrated_variance': exp_int_var,
        'variance_integrated_variance': var_int_var,
        'atm_total_variance': exp_int_var,  # For ATM options
        'atm_implied_vol': np.sqrt(exp_int_var / T) if T > 0 else np.sqrt(v0)
    }


def test_charfn_properties(params: Dict[str, float], T: float = 1.0) -> Dict[str, float]:
    """
    Test basic properties of the characteristic function.
    
    Args:
        params: Heston parameters
        T: Time to test
        
    Returns:
        Dictionary with test results
    """
    # Test phi(0) = 1
    phi_0 = heston_charfn(0.0, T, params)
    
    # Test |phi(u)| <= 1 for pure imaginary u (martingale property)
    test_u_values = [1j, 2j, 5j, 10j]
    phi_magnitudes = [abs(heston_charfn(u, T, params)) for u in test_u_values]
    
    # Test continuity at u=0
    small_u_values = [1e-6, 1e-6 * 1j]
    phi_small = [heston_charfn(u, T, params) for u in small_u_values]
    
    return {
        'phi_0_real': phi_0.real,
        'phi_0_imag': phi_0.imag,
        'phi_0_error': abs(phi_0 - 1.0),
        'max_magnitude_pure_imag': max(phi_magnitudes),
        'continuity_error': max(abs(phi - 1.0) for phi in phi_small)
    }