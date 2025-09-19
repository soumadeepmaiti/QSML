import numpy as np
from typing import Dict, Tuple, Optional
import logging
from .heston_charfn import heston_charfn

logger = logging.getLogger(__name__)


def carr_madan_fft_prices(
    S0: float, 
    r: float, 
    q: float, 
    T: float, 
    params: Dict[str, float],
    alpha: float = 1.5, 
    N: int = 4096, 
    eta: float = 0.025
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a grid of call prices C(K_i, T) using Carr-Madan FFT method.
    
    The Carr-Madan approach transforms the option pricing problem into a Fourier integral
    that can be efficiently computed using FFT.
    
    Args:
        S0: Current stock price
        r: Risk-free rate
        q: Dividend yield
        T: Time to expiration
        params: Heston parameters dictionary
        alpha: Damping parameter (>0, typically 1.0-2.0)
        N: Number of FFT points (power of 2)
        eta: Grid spacing in log-strike space
        
    Returns:
        Tuple of (K_array, C_array) where:
        - K_array: Array of strike prices
        - C_array: Array of corresponding call prices
        
    References:
        - Carr & Madan (1999): "Option valuation using the fast Fourier transform"
        - Lewis (2001): "A simple option formula for general jump-diffusion and other exponential Lévy processes"
    """
    # Input validation
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError("N must be a positive power of 2")
    
    if alpha <= 0:
        raise ValueError("Alpha must be positive")
    
    if eta <= 0:
        raise ValueError("Eta must be positive")
    
    if T <= 0:
        # Return intrinsic values for zero time
        lambda_max = (N - 1) * eta
        k_array = np.arange(N) * eta - lambda_max / 2
        K_array = S0 * np.exp(k_array)
        C_array = np.maximum(S0 - K_array, 0)
        return K_array, C_array
    
    # Step 1: Set up the grid
    lambda_max = (N - 1) * eta  # Maximum log-strike range
    
    # Frequency grid (dual to log-strike grid)
    du = 2 * np.pi / (N * eta)
    u_array = np.arange(N) * du
    
    # Log-strike grid (centered around log(S0))
    log_S0 = np.log(S0)
    k_array = np.arange(N) * eta - lambda_max / 2
    
    # Adjust grid to be centered around current log-price
    k_array = k_array + log_S0
    K_array = np.exp(k_array)
    
    # Step 2: Define the characteristic function of log(S_T)
    # For Heston model under risk-neutral measure:
    # log(S_T) = log(S0) + (r-q)*T - 0.5*∫v_s ds + ∫√v_s dW_s
    
    def char_fn_log_ST(u):
        """Characteristic function of log(S_T)"""
        # Drift adjustment
        drift = (r - q) * T
        
        # Heston characteristic function gives E[exp(i*u*log(S_T/S0))]
        # We need E[exp(i*u*log(S_T))] = exp(i*u*log(S0)) * heston_charfn(u, T, params)
        return np.exp(1j * u * (log_S0 + drift)) * heston_charfn(u - 1j * alpha, T, params)
    
    # Step 3: Build the integrand for Carr-Madan formula
    # The damped call price transform is:
    # ∫ exp(-i*u*k) * ψ(u) du
    # where ψ(u) = exp(-r*T) * φ(u - i(α+1)) / (α^2 + α - u^2 + i*u*(2α+1))
    
    integrand = np.zeros(N, dtype=complex)
    
    for j in range(N):
        u = u_array[j]
        
        # Skip u=0 to avoid division by zero (handle separately if needed)
        if abs(u) < 1e-12:
            integrand[j] = 0.0
            continue
        
        # Characteristic function at shifted argument
        phi_shifted = char_fn_log_ST(u)
        
        # Carr-Madan integrand
        denominator = alpha**2 + alpha - u**2 + 1j * u * (2 * alpha + 1)
        
        if abs(denominator) < 1e-12:
            integrand[j] = 0.0
            continue
        
        psi = np.exp(-r * T) * phi_shifted / denominator
        integrand[j] = psi
    
    # Handle u=0 case separately using L'Hôpital's rule or limiting behavior
    if abs(u_array[0]) < 1e-12:
        # For u→0, use the limit of the integrand
        # This requires careful analysis of the Heston CF at small arguments
        phi_0 = char_fn_log_ST(1e-10)  # Use very small value
        integrand[0] = np.exp(-r * T) * phi_0 / (alpha**2 + alpha)
    
    # Step 4: Apply Simpson's rule weights for more accurate integration
    simpson_weights = np.ones(N)
    simpson_weights[1::2] = 4  # Odd indices (1, 3, 5, ...)
    simpson_weights[2::2] = 2  # Even indices (2, 4, 6, ...)
    simpson_weights[0] = simpson_weights[-1] = 1  # Endpoints
    simpson_weights *= eta / 3
    
    weighted_integrand = integrand * simpson_weights
    
    # Step 5: Apply FFT
    fft_result = np.fft.fft(weighted_integrand)
    
    # Step 6: Extract damped call prices
    damped_prices = np.real(fft_result)
    
    # Step 7: Undamp to get actual call prices
    # C(k) = exp(α*k) * damped_C(k)
    C_array = np.exp(alpha * (k_array - log_S0)) * damped_prices
    
    # Step 8: Ensure non-negative prices and fix any numerical issues
    C_array = np.maximum(C_array, 0)
    
    # Basic arbitrage bounds
    intrinsic = np.maximum(S0 * np.exp(-q * T) - K_array * np.exp(-r * T), 0)
    upper_bound = S0 * np.exp(-q * T)
    
    C_array = np.minimum(C_array, upper_bound)
    C_array = np.maximum(C_array, intrinsic)
    
    logger.debug(f"FFT pricing: N={N}, alpha={alpha}, eta={eta}")
    logger.debug(f"Strike range: [{K_array.min():.2f}, {K_array.max():.2f}]")
    logger.debug(f"Price range: [{C_array.min():.6f}, {C_array.max():.6f}]")
    
    return K_array, C_array


def price_strikes_fft(
    S0: float, 
    r: float, 
    q: float, 
    T: float, 
    params: Dict[str, float], 
    K_array: np.ndarray,
    alpha: float = 1.5, 
    N: int = 4096, 
    eta: float = 0.025
) -> np.ndarray:
    """
    Price specific strikes using FFT with interpolation.
    
    Args:
        S0: Current stock price
        r: Risk-free rate
        q: Dividend yield
        T: Time to expiration
        params: Heston parameters
        K_array: Array of strikes to price
        alpha: Damping parameter
        N: Number of FFT points
        eta: Grid spacing
        
    Returns:
        Array of call prices for the specified strikes
    """
    # Get FFT prices on grid
    K_grid, C_grid = carr_madan_fft_prices(S0, r, q, T, params, alpha, N, eta)
    
    # Interpolate to desired strikes
    # Use log-linear interpolation for better behavior
    log_K_grid = np.log(K_grid)
    log_K_target = np.log(K_array)
    
    # Linear interpolation in log-space
    prices_interp = np.interp(log_K_target, log_K_grid, C_grid)
    
    # For strikes outside the grid, use boundary values or extrapolation
    out_of_bounds_low = log_K_target < log_K_grid[0]
    out_of_bounds_high = log_K_target > log_K_grid[-1]
    
    if np.any(out_of_bounds_low):
        # For very low strikes, use intrinsic value
        intrinsic_low = np.maximum(S0 * np.exp(-q * T) - K_array * np.exp(-r * T), 0)
        prices_interp[out_of_bounds_low] = intrinsic_low[out_of_bounds_low]
    
    if np.any(out_of_bounds_high):
        # For very high strikes, price should be near zero
        prices_interp[out_of_bounds_high] = np.maximum(C_grid[-1] * 0.1, 1e-10)
    
    # Apply arbitrage bounds
    intrinsic = np.maximum(S0 * np.exp(-q * T) - K_array * np.exp(-r * T), 0)
    upper_bound = S0 * np.exp(-q * T)
    
    prices_interp = np.minimum(prices_interp, upper_bound)
    prices_interp = np.maximum(prices_interp, intrinsic)
    
    return prices_interp


def optimize_fft_params(
    S0: float, 
    K_range: Tuple[float, float], 
    T: float,
    target_accuracy: float = 1e-4
) -> Dict[str, float]:
    """
    Optimize FFT parameters for given strike range and accuracy requirements.
    
    Args:
        S0: Current stock price
        K_range: (min_strike, max_strike) to cover
        T: Time to expiration
        target_accuracy: Target relative accuracy
        
    Returns:
        Dictionary with optimal alpha, N, eta parameters
    """
    K_min, K_max = K_range
    
    # Log-moneyness range
    log_m_min = np.log(K_min / S0)
    log_m_max = np.log(K_max / S0)
    log_m_range = log_m_max - log_m_min
    
    # Choose eta to have sufficient resolution
    # Rule of thumb: at least 50 points across the range
    eta_max = log_m_range / 50
    eta = min(0.05, eta_max)  # But not too large
    
    # Choose N to cover the range with some buffer
    buffer_factor = 2.0  # 100% buffer on each side
    required_range = log_m_range * buffer_factor
    N_min = int(2**np.ceil(np.log2(required_range / eta)))
    N = max(1024, N_min)  # Minimum 1024 for stability
    N = min(N, 16384)     # Maximum 16384 for memory
    
    # Choose alpha for stability
    # Higher alpha for OTM options, lower for ITM
    if abs(log_m_min) > abs(log_m_max):
        # More OTM puts
        alpha = 1.75
    elif log_m_max > abs(log_m_min):
        # More OTM calls
        alpha = 1.25
    else:
        # Balanced
        alpha = 1.5
    
    return {
        'alpha': alpha,
        'N': N,
        'eta': eta,
        'expected_accuracy': target_accuracy,
        'log_moneyness_range': log_m_range
    }


def fft_put_prices(
    S0: float, 
    r: float, 
    q: float, 
    T: float, 
    params: Dict[str, float],
    K_array: np.ndarray,
    alpha: float = 1.5, 
    N: int = 4096, 
    eta: float = 0.025
) -> np.ndarray:
    """
    Calculate put prices using put-call parity from FFT call prices.
    
    Args:
        S0: Current stock price
        r: Risk-free rate  
        q: Dividend yield
        T: Time to expiration
        params: Heston parameters
        K_array: Array of strike prices
        alpha: FFT damping parameter
        N: Number of FFT points
        eta: Grid spacing
        
    Returns:
        Array of put prices
    """
    # Get call prices
    call_prices = price_strikes_fft(S0, r, q, T, params, K_array, alpha, N, eta)
    
    # Apply put-call parity: P = C - S*exp(-q*T) + K*exp(-r*T)
    forward = S0 * np.exp(-q * T)
    discount = np.exp(-r * T)
    
    put_prices = call_prices - forward + K_array * discount
    
    # Ensure non-negative
    put_prices = np.maximum(put_prices, 0)
    
    # Apply arbitrage bounds for puts
    intrinsic = np.maximum(K_array - S0, 0) * np.exp(-r * T)
    upper_bound = K_array * discount
    
    put_prices = np.minimum(put_prices, upper_bound)
    put_prices = np.maximum(put_prices, intrinsic)
    
    return put_prices


def test_fft_convergence(
    S0: float, 
    r: float, 
    q: float, 
    T: float, 
    params: Dict[str, float],
    K_test: float,
    alpha: float = 1.5
) -> Dict[str, float]:
    """
    Test FFT convergence by varying N and eta parameters.
    
    Args:
        S0, r, q, T: Market parameters
        params: Heston parameters
        K_test: Test strike price
        alpha: Damping parameter
        
    Returns:
        Dictionary with convergence test results
    """
    N_values = [512, 1024, 2048, 4096, 8192]
    eta_values = [0.05, 0.025, 0.0125]
    
    results = {}
    
    for N in N_values:
        for eta in eta_values:
            try:
                price = price_strikes_fft(S0, r, q, T, params, np.array([K_test]), alpha, N, eta)[0]
                results[f'N_{N}_eta_{eta}'] = price
            except Exception as e:
                logger.warning(f"FFT failed for N={N}, eta={eta}: {e}")
                results[f'N_{N}_eta_{eta}'] = np.nan
    
    # Calculate convergence metrics
    prices = np.array([v for v in results.values() if not np.isnan(v)])
    if len(prices) > 1:
        price_std = np.std(prices)
        price_mean = np.mean(prices)
        relative_std = price_std / price_mean if price_mean > 0 else np.inf
    else:
        relative_std = np.inf
    
    results['convergence_relative_std'] = relative_std
    results['price_mean'] = price_mean if len(prices) > 1 else np.nan
    
    return results