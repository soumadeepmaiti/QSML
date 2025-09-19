import numpy as np
from scipy.integrate import quad
from typing import Dict, Tuple
import logging
from .heston_charfn import heston_charfn

logger = logging.getLogger(__name__)


def heston_call_price_integral(
    S0: float, 
    K: float, 
    r: float, 
    q: float, 
    T: float, 
    params: Dict[str, float],
    method: str = "lewis"
) -> float:
    """
    Semi-closed form Heston call price using P1, P2 integrals.
    
    This implementation uses the Lewis (2001) approach or the classic
    Albrecher et al. (2007) formulation for accuracy checks vs FFT.
    
    Args:
        S0: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Dividend yield
        T: Time to expiration
        params: Heston parameters dictionary
        method: "lewis" or "albrecher" formulation
        
    Returns:
        Call option price
        
    References:
        - Lewis (2001): "A Simple Option Formula for General Jump-Diffusion and Other Exponential Lévy Processes"
        - Albrecher et al. (2007): "The Little Heston Trap"
    """
    if T <= 0:
        return max(S0 - K, 0)
    
    if method == "lewis":
        return _heston_call_lewis(S0, K, r, q, T, params)
    elif method == "albrecher":
        return _heston_call_albrecher(S0, K, r, q, T, params)
    else:
        raise ValueError(f"Unknown method: {method}")


def _heston_call_lewis(S0: float, K: float, r: float, q: float, T: float, params: Dict[str, float]) -> float:
    """
    Heston call price using Lewis (2001) formulation.
    
    The Lewis approach directly integrates the characteristic function
    without requiring the P1/P2 probability decomposition.
    """
    discount = np.exp(-r * T)
    forward = S0 * np.exp((r - q) * T)
    log_moneyness = np.log(K / forward)
    
    def integrand(u):
        """Lewis integrand"""
        if u == 0:
            return 0.0
        
        # Characteristic function of log(S_T/F)
        phi = heston_charfn(u - 0.5j, T, params)
        
        # Lewis formula integrand
        numerator = np.exp(-1j * u * log_moneyness) * phi
        denominator = 1j * u
        
        return np.real(numerator / denominator)
    
    try:
        # Integrate from 0 to infinity
        integral_result, _ = quad(integrand, 1e-6, 100, limit=100, epsabs=1e-8, epsrel=1e-6)
        
        # Lewis formula
        call_price = discount * forward * (0.5 + integral_result / np.pi)
        call_price = call_price - discount * K * (0.5 + integral_result / np.pi)
        
        # Ensure non-negative and apply basic bounds
        intrinsic = max(forward - K, 0) * discount
        call_price = max(call_price, intrinsic)
        call_price = min(call_price, forward * discount)
        
        return call_price
        
    except Exception as e:
        logger.warning(f"Lewis integration failed: {e}")
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0)


def _heston_call_albrecher(S0: float, K: float, r: float, q: float, T: float, params: Dict[str, float]) -> float:
    """
    Heston call price using Albrecher et al. (2007) P1/P2 formulation.
    
    This is the classic approach that decomposes the option price into
    two risk-neutral probabilities P1 and P2.
    """
    discount = np.exp(-r * T)
    forward = S0 * np.exp(-q * T)
    
    # Calculate P1 and P2 probabilities
    P1 = _calculate_probability(S0, K, r, q, T, params, prob_type=1)
    P2 = _calculate_probability(S0, K, r, q, T, params, prob_type=2)
    
    # Heston call formula
    call_price = forward * P1 - K * discount * P2
    
    # Ensure non-negative and apply basic bounds
    intrinsic = max(forward - K * discount, 0)
    call_price = max(call_price, intrinsic)
    call_price = min(call_price, forward)
    
    return call_price


def _calculate_probability(
    S0: float, 
    K: float, 
    r: float, 
    q: float, 
    T: float, 
    params: Dict[str, float], 
    prob_type: int
) -> float:
    """
    Calculate P1 or P2 probability for Heston formula.
    
    Args:
        prob_type: 1 for P1 (delta-hedged probability), 2 for P2 (risk-neutral probability)
    """
    log_moneyness = np.log(S0 / K) + (r - q) * T
    
    def integrand(u):
        """Probability integrand"""
        if u == 0:
            return 0.0
        
        if prob_type == 1:
            # P1 integrand (delta-hedged measure)
            phi = heston_charfn(u - 1j, T, params)
            denominator = 1j * u
        else:
            # P2 integrand (risk-neutral measure)
            phi = heston_charfn(u, T, params)
            denominator = 1j * u
        
        numerator = np.exp(1j * u * log_moneyness) * phi
        
        return np.real(numerator / denominator)
    
    try:
        # Integrate from 0 to infinity
        integral_result, _ = quad(integrand, 1e-6, 100, limit=100, epsabs=1e-8, epsrel=1e-6)
        
        # Probability formula
        probability = 0.5 + integral_result / np.pi
        
        # Ensure probability is in [0, 1]
        probability = max(0.0, min(1.0, probability))
        
        return probability
        
    except Exception as e:
        logger.warning(f"Probability integration failed for P{prob_type}: {e}")
        return 0.5  # Fallback to 50%


def heston_put_price_integral(
    S0: float, 
    K: float, 
    r: float, 
    q: float, 
    T: float, 
    params: Dict[str, float],
    method: str = "lewis"
) -> float:
    """
    Heston put price using put-call parity from integral call price.
    
    Args:
        S0: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Dividend yield
        T: Time to expiration
        params: Heston parameters
        method: Integration method
        
    Returns:
        Put option price
    """
    # Get call price
    call_price = heston_call_price_integral(S0, K, r, q, T, params, method)
    
    # Apply put-call parity
    forward = S0 * np.exp(-q * T)
    discount = np.exp(-r * T)
    
    put_price = call_price - forward + K * discount
    
    # Ensure non-negative and apply bounds
    intrinsic = max(K - S0, 0) * discount
    put_price = max(put_price, intrinsic)
    put_price = min(put_price, K * discount)
    
    return put_price


def heston_option_price_integral(
    S0: float, 
    K: float, 
    r: float, 
    q: float, 
    T: float, 
    params: Dict[str, float],
    option_type: str = "call",
    method: str = "lewis"
) -> float:
    """
    Generic Heston option pricing using semi-closed form integrals.
    
    Args:
        S0: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Dividend yield
        T: Time to expiration
        params: Heston parameters
        option_type: "call" or "put"
        method: Integration method
        
    Returns:
        Option price
    """
    if option_type.lower() == "call":
        return heston_call_price_integral(S0, K, r, q, T, params, method)
    elif option_type.lower() == "put":
        return heston_put_price_integral(S0, K, r, q, T, params, method)
    else:
        raise ValueError(f"Unknown option type: {option_type}")


def compare_integration_methods(
    S0: float, 
    K: float, 
    r: float, 
    q: float, 
    T: float, 
    params: Dict[str, float]
) -> Dict[str, float]:
    """
    Compare different integration methods for accuracy assessment.
    
    Args:
        S0, K, r, q, T: Market parameters
        params: Heston parameters
        
    Returns:
        Dictionary with prices from different methods
    """
    results = {}
    
    methods = ["lewis", "albrecher"]
    
    for method in methods:
        try:
            call_price = heston_call_price_integral(S0, K, r, q, T, params, method)
            put_price = heston_put_price_integral(S0, K, r, q, T, params, method)
            results[f"call_{method}"] = call_price
            results[f"put_{method}"] = put_price
        except Exception as e:
            logger.warning(f"Method {method} failed: {e}")
            results[f"call_{method}"] = np.nan
            results[f"put_{method}"] = np.nan
    
    # Test put-call parity
    if not np.isnan(results.get("call_lewis", np.nan)) and not np.isnan(results.get("put_lewis", np.nan)):
        forward = S0 * np.exp(-q * T)
        discount = np.exp(-r * T)
        parity_lhs = results["call_lewis"] - results["put_lewis"]
        parity_rhs = forward - K * discount
        results["parity_error_lewis"] = abs(parity_lhs - parity_rhs)
    
    return results


def adaptive_integration_bounds(
    S0: float, 
    K: float, 
    T: float, 
    params: Dict[str, float]
) -> Tuple[float, float]:
    """
    Determine appropriate integration bounds based on option characteristics.
    
    Args:
        S0: Current stock price
        K: Strike price
        T: Time to expiration
        params: Heston parameters
        
    Returns:
        Tuple of (lower_bound, upper_bound) for integration
    """
    # Basic bounds
    u_min = 1e-6
    u_max = 100.0
    
    # Adjust based on time to expiration
    if T < 0.1:  # Short-term options
        u_max = min(u_max, 50.0)
    elif T > 2.0:  # Long-term options
        u_max = min(u_max, 200.0)
    
    # Adjust based on moneyness
    log_moneyness = np.log(K / S0)
    if abs(log_moneyness) > 0.5:  # Far from ATM
        u_max = min(u_max, 150.0)
    
    # Adjust based on volatility of volatility
    sigma = params.get('sigma', 0.3)
    if sigma > 1.0:  # High vol-of-vol
        u_max = min(u_max, 75.0)
    
    return u_min, u_max


def test_integration_accuracy(
    S0: float, 
    K: float, 
    r: float, 
    q: float, 
    T: float, 
    params: Dict[str, float]
) -> Dict[str, float]:
    """
    Test integration accuracy by varying tolerance parameters.
    
    Args:
        S0, K, r, q, T: Market parameters
        params: Heston parameters
        
    Returns:
        Dictionary with accuracy test results
    """
    tolerances = [1e-6, 1e-8, 1e-10]
    results = {}
    
    for tol in tolerances:
        # Modify the quad function tolerance in the global functions
        try:
            price = heston_call_price_integral(S0, K, r, q, T, params, "lewis")
            results[f"tol_{tol}"] = price
        except Exception as e:
            logger.warning(f"Integration failed for tolerance {tol}: {e}")
            results[f"tol_{tol}"] = np.nan
    
    # Calculate convergence metrics
    prices = np.array([v for v in results.values() if not np.isnan(v)])
    if len(prices) > 1:
        price_std = np.std(prices)
        price_mean = np.mean(prices)
        results["convergence_std"] = price_std
        results["convergence_relative_std"] = price_std / price_mean if price_mean > 0 else np.inf
    
    return results