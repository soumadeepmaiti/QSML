import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def yearfrac(d1: Union[pd.Timestamp, str], d2: Union[pd.Timestamp, str], basis: str = "act/365") -> float:
    """
    Calculate year fraction between two dates.
    
    Args:
        d1: Start date
        d2: End date
        basis: Day count convention ("act/365", "act/360", "30/360")
        
    Returns:
        Year fraction as float
    """
    if isinstance(d1, str):
        d1 = pd.to_datetime(d1)
    if isinstance(d2, str):
        d2 = pd.to_datetime(d2)
    
    if basis == "act/365":
        return (d2 - d1).days / 365.0
    elif basis == "act/360":
        return (d2 - d1).days / 360.0
    elif basis == "30/360":
        # Simplified 30/360 calculation
        years = d2.year - d1.year
        months = d2.month - d1.month
        days = d2.day - d1.day
        return (years * 360 + months * 30 + days) / 360.0
    else:
        raise ValueError(f"Unsupported day count basis: {basis}")


def prices_to_iv(
    prices: Union[float, np.ndarray],
    S: float,
    r: float,
    q: float,
    T: float,
    K: Union[float, np.ndarray],
    option_type: str = "call",
    max_iter: int = 100,
    tolerance: float = 1e-6
) -> Union[float, np.ndarray]:
    """
    Convert option prices to implied volatilities using numerical inversion.
    
    Args:
        prices: Option prices
        S: Current stock price
        r: Risk-free rate
        q: Dividend yield
        T: Time to expiration
        K: Strike prices
        option_type: "call" or "put"
        max_iter: Maximum iterations for root finding
        tolerance: Convergence tolerance
        
    Returns:
        Implied volatilities
    """
    from ..pricers.bs import bs_call_price, bs_put_price
    
    price_func = bs_call_price if option_type.lower() == "call" else bs_put_price
    
    def iv_objective(sigma: float, target_price: float, strike: float) -> float:
        """Objective function for IV root finding."""
        if sigma <= 0:
            return float('inf')
        try:
            model_price = price_func(S, strike, r, q, T, sigma)
            return model_price - target_price
        except:
            return float('inf')
    
    # Handle scalar inputs
    if np.isscalar(prices) and np.isscalar(K):
        try:
            # Check bounds
            intrinsic = max(0, S - K) if option_type.lower() == "call" else max(0, K - S)
            if prices <= intrinsic * np.exp(-r * T):
                return 0.001  # Minimum volatility
            
            # Use Brent's method for root finding
            iv = brentq(
                lambda sigma: iv_objective(sigma, prices, K),
                0.001, 5.0,  # Search bounds
                xtol=tolerance,
                maxiter=max_iter
            )
            return max(iv, 0.001)
        except:
            logger.warning(f"IV inversion failed for price={prices}, K={K}")
            return np.nan
    
    # Handle array inputs
    prices = np.asarray(prices)
    K = np.asarray(K)
    
    # Broadcast to compatible shapes
    prices, K = np.broadcast_arrays(prices, K)
    ivs = np.full_like(prices, np.nan, dtype=float)
    
    for i in np.ndindex(prices.shape):
        try:
            price = prices[i]
            strike = K[i]
            
            # Check bounds
            intrinsic = max(0, S - strike) if option_type.lower() == "call" else max(0, strike - S)
            if price <= intrinsic * np.exp(-r * T):
                ivs[i] = 0.001
                continue
            
            iv = brentq(
                lambda sigma: iv_objective(sigma, price, strike),
                0.001, 5.0,
                xtol=tolerance,
                maxiter=max_iter
            )
            ivs[i] = max(iv, 0.001)
        except:
            logger.warning(f"IV inversion failed for price={prices[i]}, K={K[i]}")
            ivs[i] = np.nan
    
    return ivs


def iv_to_prices(
    iv: Union[float, np.ndarray],
    S: float,
    r: float,
    q: float,
    T: float,
    K: Union[float, np.ndarray],
    option_type: str = "call"
) -> Union[float, np.ndarray]:
    """
    Convert implied volatilities to option prices.
    
    Args:
        iv: Implied volatilities
        S: Current stock price
        r: Risk-free rate
        q: Dividend yield
        T: Time to expiration
        K: Strike prices
        option_type: "call" or "put"
        
    Returns:
        Option prices
    """
    from ..pricers.bs import bs_call_price, bs_put_price
    
    price_func = bs_call_price if option_type.lower() == "call" else bs_put_price
    
    # Handle scalar inputs
    if np.isscalar(iv) and np.isscalar(K):
        return price_func(S, K, r, q, T, iv)
    
    # Handle array inputs
    iv = np.asarray(iv)
    K = np.asarray(K)
    iv, K = np.broadcast_arrays(iv, K)
    
    prices = np.full_like(iv, np.nan, dtype=float)
    
    for i in np.ndindex(iv.shape):
        try:
            prices[i] = price_func(S, K[i], r, q, T, iv[i])
        except:
            prices[i] = np.nan
    
    return prices


def moneyness(S: float, K: Union[float, np.ndarray], log: bool = True) -> Union[float, np.ndarray]:
    """
    Calculate moneyness of options.
    
    Args:
        S: Current stock price
        K: Strike prices
        log: If True, return log-moneyness ln(K/S), else K/S
        
    Returns:
        Moneyness values
    """
    if log:
        return np.log(K / S)
    else:
        return K / S


def atm_forward(S: float, r: float, q: float, T: float) -> float:
    """
    Calculate at-the-money forward price.
    
    Args:
        S: Current stock price
        r: Risk-free rate
        q: Dividend yield
        T: Time to expiration
        
    Returns:
        Forward price
    """
    return S * np.exp((r - q) * T)


def discount_factor(r: float, T: float) -> float:
    """
    Calculate discount factor.
    
    Args:
        r: Risk-free rate
        T: Time to expiration
        
    Returns:
        Discount factor
    """
    return np.exp(-r * T)


def forward_value(S: float, r: float, q: float, T: float) -> float:
    """
    Calculate forward value of the stock.
    
    Args:
        S: Current stock price
        r: Risk-free rate
        q: Dividend yield
        T: Time to expiration
        
    Returns:
        Forward value
    """
    return S * np.exp((r - q) * T)


def put_call_parity_check(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    tolerance: float = 1e-6
) -> bool:
    """
    Check if call and put prices satisfy put-call parity.
    
    Args:
        call_price: Call option price
        put_price: Put option price
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        q: Dividend yield
        T: Time to expiration
        tolerance: Tolerance for parity check
        
    Returns:
        True if parity holds within tolerance
    """
    forward = S * np.exp(-q * T)
    discount = np.exp(-r * T)
    
    left_side = call_price - put_price
    right_side = forward - K * discount
    
    return abs(left_side - right_side) < tolerance


def interpolate_rates(
    dates: pd.DatetimeIndex,
    rates: np.ndarray,
    target_dates: pd.DatetimeIndex,
    method: str = "linear"
) -> np.ndarray:
    """
    Interpolate interest rates for given dates.
    
    Args:
        dates: Original rate dates
        rates: Original rates
        target_dates: Dates to interpolate for
        method: Interpolation method
        
    Returns:
        Interpolated rates
    """
    if method == "linear":
        return np.interp(
            target_dates.astype(np.int64),
            dates.astype(np.int64),
            rates
        )
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")


def ensure_positive(x: Union[float, np.ndarray], min_val: float = 1e-10) -> Union[float, np.ndarray]:
    """
    Ensure values are positive by clipping to minimum value.
    
    Args:
        x: Input values
        min_val: Minimum allowed value
        
    Returns:
        Clipped values
    """
    return np.maximum(x, min_val)