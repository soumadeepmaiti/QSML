import numpy as np
from scipy.stats import norm
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def bs_call_price(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """
    Black-Scholes European call option price with continuous dividend yield.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        
    Returns:
        Call option price
    """
    if T <= 0:
        return max(S - K, 0)
    
    if sigma <= 0:
        # Intrinsic value for zero volatility
        forward_price = S * np.exp((r - q) * T)
        return np.exp(-r * T) * max(forward_price - K, 0)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = (S * np.exp(-q * T) * norm.cdf(d1) - 
                  K * np.exp(-r * T) * norm.cdf(d2))
    
    return max(call_price, 0)  # Ensure non-negative


def bs_put_price(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """
    Black-Scholes European put option price with continuous dividend yield.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        
    Returns:
        Put option price
    """
    if T <= 0:
        return max(K - S, 0)
    
    if sigma <= 0:
        # Intrinsic value for zero volatility
        forward_price = S * np.exp((r - q) * T)
        return np.exp(-r * T) * max(K - forward_price, 0)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - 
                 S * np.exp(-q * T) * norm.cdf(-d1))
    
    return max(put_price, 0)  # Ensure non-negative


def bs_delta_call(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """
    Black-Scholes call delta (sensitivity to underlying price).
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        
    Returns:
        Call delta
    """
    if T <= 0:
        return 1.0 if S > K else 0.0
    
    if sigma <= 0:
        forward_price = S * np.exp((r - q) * T)
        return np.exp(-q * T) if forward_price > K else 0.0
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.cdf(d1)


def bs_delta_put(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """
    Black-Scholes put delta (sensitivity to underlying price).
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        
    Returns:
        Put delta
    """
    if T <= 0:
        return -1.0 if S < K else 0.0
    
    if sigma <= 0:
        forward_price = S * np.exp((r - q) * T)
        return -np.exp(-q * T) if forward_price < K else 0.0
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return -np.exp(-q * T) * norm.cdf(-d1)


def bs_gamma(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """
    Black-Scholes gamma (second derivative with respect to underlying price).
    Same for calls and puts.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        
    Returns:
        Gamma
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))


def bs_vega(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """
    Black-Scholes vega (sensitivity to volatility).
    Same for calls and puts.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        
    Returns:
        Vega (per unit change in volatility, not percentage point)
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_theta_call(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """
    Black-Scholes call theta (time decay).
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        
    Returns:
        Call theta (per day, so divided by 365)
    """
    if T <= 0:
        return 0.0
    
    if sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    theta = (-(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) -
             r * K * np.exp(-r * T) * norm.cdf(d2) +
             q * S * np.exp(-q * T) * norm.cdf(d1))
    
    return theta / 365  # Convert to per-day


def bs_theta_put(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """
    Black-Scholes put theta (time decay).
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        
    Returns:
        Put theta (per day, so divided by 365)
    """
    if T <= 0:
        return 0.0
    
    if sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    theta = (-(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) +
             r * K * np.exp(-r * T) * norm.cdf(-d2) -
             q * S * np.exp(-q * T) * norm.cdf(-d1))
    
    return theta / 365  # Convert to per-day


def bs_rho_call(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """
    Black-Scholes call rho (sensitivity to interest rate).
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        
    Returns:
        Call rho
    """
    if T <= 0:
        return 0.0
    
    if sigma <= 0:
        forward_price = S * np.exp((r - q) * T)
        if forward_price > K:
            return K * T * np.exp(-r * T)
        else:
            return 0.0
    
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return K * T * np.exp(-r * T) * norm.cdf(d2)


def bs_rho_put(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """
    Black-Scholes put rho (sensitivity to interest rate).
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        
    Returns:
        Put rho
    """
    if T <= 0:
        return 0.0
    
    if sigma <= 0:
        forward_price = S * np.exp((r - q) * T)
        if forward_price < K:
            return -K * T * np.exp(-r * T)
        else:
            return 0.0
    
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def bs_option_price(
    S: float, 
    K: float, 
    r: float, 
    q: float, 
    T: float, 
    sigma: float, 
    option_type: str = "call"
) -> float:
    """
    Generic Black-Scholes option pricing function.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        option_type: "call" or "put"
        
    Returns:
        Option price
    """
    if option_type.lower() == "call":
        return bs_call_price(S, K, r, q, T, sigma)
    elif option_type.lower() == "put":
        return bs_put_price(S, K, r, q, T, sigma)
    else:
        raise ValueError(f"Unknown option type: {option_type}")


def bs_delta(
    S: float, 
    K: float, 
    r: float, 
    q: float, 
    T: float, 
    sigma: float, 
    option_type: str = "call"
) -> float:
    """
    Generic Black-Scholes delta function.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        option_type: "call" or "put"
        
    Returns:
        Delta
    """
    if option_type.lower() == "call":
        return bs_delta_call(S, K, r, q, T, sigma)
    elif option_type.lower() == "put":
        return bs_delta_put(S, K, r, q, T, sigma)
    else:
        raise ValueError(f"Unknown option type: {option_type}")


def bs_theta(
    S: float, 
    K: float, 
    r: float, 
    q: float, 
    T: float, 
    sigma: float, 
    option_type: str = "call"
) -> float:
    """
    Generic Black-Scholes theta function.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        option_type: "call" or "put"
        
    Returns:
        Theta (per day)
    """
    if option_type.lower() == "call":
        return bs_theta_call(S, K, r, q, T, sigma)
    elif option_type.lower() == "put":
        return bs_theta_put(S, K, r, q, T, sigma)
    else:
        raise ValueError(f"Unknown option type: {option_type}")


def bs_rho(
    S: float, 
    K: float, 
    r: float, 
    q: float, 
    T: float, 
    sigma: float, 
    option_type: str = "call"
) -> float:
    """
    Generic Black-Scholes rho function.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        option_type: "call" or "put"
        
    Returns:
        Rho
    """
    if option_type.lower() == "call":
        return bs_rho_call(S, K, r, q, T, sigma)
    elif option_type.lower() == "put":
        return bs_rho_put(S, K, r, q, T, sigma)
    else:
        raise ValueError(f"Unknown option type: {option_type}")


def bs_all_greeks(
    S: float, 
    K: float, 
    r: float, 
    q: float, 
    T: float, 
    sigma: float, 
    option_type: str = "call"
) -> dict:
    """
    Calculate all Black-Scholes Greeks at once for efficiency.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        sigma: Volatility
        option_type: "call" or "put"
        
    Returns:
        Dictionary with all Greeks and price
    """
    return {
        'price': bs_option_price(S, K, r, q, T, sigma, option_type),
        'delta': bs_delta(S, K, r, q, T, sigma, option_type),
        'gamma': bs_gamma(S, K, r, q, T, sigma),
        'vega': bs_vega(S, K, r, q, T, sigma),
        'theta': bs_theta(S, K, r, q, T, sigma, option_type),
        'rho': bs_rho(S, K, r, q, T, sigma, option_type)
    }


def bs_implied_vol_newton(
    price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    option_type: str = "call",
    initial_guess: float = 0.2,
    max_iter: int = 100,
    tolerance: float = 1e-6
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Args:
        price: Market price of the option
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Continuous dividend yield
        T: Time to expiration in years
        option_type: "call" or "put"
        initial_guess: Initial volatility guess
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Implied volatility
    """
    sigma = initial_guess
    
    for i in range(max_iter):
        # Calculate price and vega
        model_price = bs_option_price(S, K, r, q, T, sigma, option_type)
        vega = bs_vega(S, K, r, q, T, sigma)
        
        # Check convergence
        price_diff = model_price - price
        if abs(price_diff) < tolerance:
            return sigma
        
        # Newton-Raphson update
        if vega <= 0:
            break
        
        sigma_new = sigma - price_diff / vega
        
        # Ensure reasonable bounds
        sigma_new = max(0.001, min(5.0, sigma_new))
        
        sigma = sigma_new
    
    # If Newton-Raphson fails, return NaN
    logger.warning(f"IV Newton-Raphson failed to converge for price={price}")
    return np.nan


class BlackScholesPricer:
    """
    Black-Scholes pricer class wrapper around the functional implementation.
    """
    
    def __init__(self):
        """Initialize the Black-Scholes pricer."""
        pass
    
    def price(self, S: float, K: float, r: float, q: float, T: float, 
              sigma: float, option_type: str = "call") -> float:
        """Calculate option price."""
        return bs_option_price(S, K, r, q, T, sigma, option_type)
    
    def delta(self, S: float, K: float, r: float, q: float, T: float, 
              sigma: float, option_type: str = "call") -> float:
        """Calculate delta."""
        return bs_delta(S, K, r, q, T, sigma, option_type)
    
    def gamma(self, S: float, K: float, r: float, q: float, T: float, 
              sigma: float) -> float:
        """Calculate gamma."""
        return bs_gamma(S, K, r, q, T, sigma)
    
    def vega(self, S: float, K: float, r: float, q: float, T: float, 
             sigma: float) -> float:
        """Calculate vega."""
        return bs_vega(S, K, r, q, T, sigma)
    
    def theta(self, S: float, K: float, r: float, q: float, T: float, 
              sigma: float, option_type: str = "call") -> float:
        """Calculate theta."""
        return bs_theta(S, K, r, q, T, sigma, option_type)
    
    def rho(self, S: float, K: float, r: float, q: float, T: float, 
            sigma: float, option_type: str = "call") -> float:
        """Calculate rho."""
        return bs_rho(S, K, r, q, T, sigma, option_type)
    
    def all_greeks(self, S: float, K: float, r: float, q: float, T: float, 
                   sigma: float, option_type: str = "call") -> dict:
        """Calculate all Greeks at once."""
        return bs_all_greeks(S, K, r, q, T, sigma, option_type)
    
    def implied_vol(self, price: float, S: float, K: float, r: float, 
                    q: float, T: float, option_type: str = "call") -> float:
        """Calculate implied volatility."""
        return bs_implied_vol_newton(price, S, K, r, q, T, option_type)