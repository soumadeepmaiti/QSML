import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_constant_rates(r: float = 0.02, q: float = 0.0, date_range: Optional[pd.DatetimeIndex] = None) -> Dict[str, pd.Series]:
    """
    Create constant risk-free rate and dividend yield series.
    
    Args:
        r: Constant risk-free rate
        q: Constant dividend yield
        date_range: Date range for the series (if None, creates single-value series)
        
    Returns:
        Dictionary with 'riskfree' and 'dividend' Series
    """
    if date_range is None:
        # Return scalar values
        return {
            'riskfree': pd.Series([r]),
            'dividend': pd.Series([q])
        }
    
    return {
        'riskfree': pd.Series(r, index=date_range, name='riskfree'),
        'dividend': pd.Series(q, index=date_range, name='dividend')
    }


def load_treasury_rates(file_path: str, rate_column: str = 'rate') -> pd.Series:
    """
    Load daily Treasury rates from a CSV file.
    
    Expected format:
    - date, rate columns
    - rate should be in decimal form (e.g., 0.02 for 2%)
    
    Args:
        file_path: Path to CSV file with Treasury rates
        rate_column: Name of the rate column
        
    Returns:
        Series with dates as index and rates as values
    """
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Ensure rate is in decimal form
        rates = df[rate_column]
        if rates.max() > 1:  # Assume percentage format if max > 1
            rates = rates / 100
        
        # Forward fill missing values
        rates = rates.sort_index().fillna(method='ffill')
        
        logger.info(f"Loaded Treasury rates: {len(rates)} observations, "
                   f"range: {rates.min():.3f} to {rates.max():.3f}")
        
        return rates
        
    except Exception as e:
        logger.error(f"Failed to load Treasury rates from {file_path}: {e}")
        raise


def load_dividend_yields(
    file_path: str,
    ticker_column: str = 'ticker',
    yield_column: str = 'dividend_yield'
) -> pd.DataFrame:
    """
    Load dividend yields by ticker and date.
    
    Expected format:
    - date, ticker, dividend_yield columns
    - yield should be in decimal form (e.g., 0.015 for 1.5%)
    
    Args:
        file_path: Path to CSV file with dividend yields
        ticker_column: Name of the ticker column
        yield_column: Name of the yield column
        
    Returns:
        DataFrame with date, ticker, and dividend yield
    """
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure yield is in decimal form
        if df[yield_column].max() > 1:  # Assume percentage format if max > 1
            df[yield_column] = df[yield_column] / 100
        
        df = df[['date', ticker_column, yield_column]].dropna()
        
        logger.info(f"Loaded dividend yields: {len(df)} observations, "
                   f"{df[ticker_column].nunique()} tickers")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load dividend yields from {file_path}: {e}")
        raise


def interpolate_rates_for_dates(
    rates: pd.Series,
    target_dates: pd.DatetimeIndex,
    method: str = 'linear'
) -> pd.Series:
    """
    Interpolate rates for specific target dates.
    
    Args:
        rates: Series with dates as index and rates as values
        target_dates: Dates to interpolate rates for
        method: Interpolation method ('linear', 'nearest', 'ffill', 'bfill')
        
    Returns:
        Series with interpolated rates for target dates
    """
    # Ensure rates are sorted by date
    rates = rates.sort_index()
    
    # Create new series with target dates
    if method == 'linear':
        # Linear interpolation
        interpolated = rates.reindex(
            rates.index.union(target_dates)
        ).interpolate(method='linear').reindex(target_dates)
        
    elif method == 'nearest':
        # Nearest neighbor
        interpolated = rates.reindex(target_dates, method='nearest')
        
    elif method == 'ffill':
        # Forward fill
        interpolated = rates.reindex(
            rates.index.union(target_dates)
        ).fillna(method='ffill').reindex(target_dates)
        
    elif method == 'bfill':
        # Backward fill
        interpolated = rates.reindex(
            rates.index.union(target_dates)
        ).fillna(method='bfill').reindex(target_dates)
        
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return interpolated


def compute_dividend_yield_from_prices(
    price_df: pd.DataFrame,
    dividend_events: Optional[pd.DataFrame] = None,
    lookback_days: int = 252
) -> pd.Series:
    """
    Estimate continuous dividend yield from price data and dividend events.
    
    Args:
        price_df: DataFrame with columns ['date', 'ticker', 'close', 'adj_close']
        dividend_events: Optional DataFrame with dividend payments
        lookback_days: Rolling window for yield calculation
        
    Returns:
        Series with estimated dividend yields by date
    """
    if 'adj_close' not in price_df.columns or 'close' not in price_df.columns:
        logger.warning("Cannot compute dividend yield without both close and adj_close prices")
        return pd.Series(0.0, index=price_df['date'].unique())
    
    # Simple method: compare close vs adjusted close
    price_df = price_df.sort_values('date')
    
    # Calculate dividend yield as the difference between price return and adjusted return
    price_df['price_return'] = price_df.groupby('ticker')['close'].pct_change()
    price_df['adj_return'] = price_df.groupby('ticker')['adj_close'].pct_change()
    price_df['dividend_component'] = price_df['price_return'] - price_df['adj_return']
    
    # Annualize the rolling sum of dividend components
    rolling_div = price_df.groupby('ticker')['dividend_component'].rolling(
        window=lookback_days, min_periods=20
    ).sum() * (252 / lookback_days)
    
    # Take average across tickers for each date
    daily_yield = rolling_div.groupby('date').mean().fillna(0.0)
    
    logger.info(f"Computed dividend yields: mean {daily_yield.mean():.3f}, "
               f"range [{daily_yield.min():.3f}, {daily_yield.max():.3f}]")
    
    return daily_yield


def prepare_rate_curves(
    options_dates: pd.DatetimeIndex,
    config: Dict[str, Any],
    external_data_path: Optional[str] = None
) -> Dict[str, pd.Series]:
    """
    Prepare risk-free rate and dividend yield curves for option pricing.
    
    Args:
        options_dates: Unique dates from options data
        config: Configuration dictionary
        external_data_path: Path to external data directory
        
    Returns:
        Dictionary with 'riskfree' and 'dividend' Series indexed by date
    """
    rf_config = config.get('riskfree', {})
    div_config = config.get('dividends', {})
    
    # Risk-free rates
    if rf_config.get('mode') == 'constant':
        r = rf_config.get('constant_rate', 0.02)
        riskfree_series = pd.Series(r, index=options_dates, name='riskfree')
        logger.info(f"Using constant risk-free rate: {r:.3f}")
        
    elif rf_config.get('mode') == 'daily' and external_data_path:
        try:
            treasury_file = f"{external_data_path}/treasury_rates.csv"
            treasury_rates = load_treasury_rates(treasury_file)
            riskfree_series = interpolate_rates_for_dates(treasury_rates, options_dates)
            logger.info("Using daily Treasury rates")
        except Exception as e:
            logger.warning(f"Failed to load daily rates, using constant: {e}")
            r = rf_config.get('constant_rate', 0.02)
            riskfree_series = pd.Series(r, index=options_dates, name='riskfree')
    else:
        r = rf_config.get('constant_rate', 0.02)
        riskfree_series = pd.Series(r, index=options_dates, name='riskfree')
    
    # Dividend yields
    if div_config.get('mode') == 'constant':
        q = div_config.get('constant_yield', 0.0)
        dividend_series = pd.Series(q, index=options_dates, name='dividend')
        logger.info(f"Using constant dividend yield: {q:.3f}")
        
    elif div_config.get('mode') == 'daily' and external_data_path:
        try:
            dividend_file = f"{external_data_path}/dividend_yields.csv"
            dividend_data = load_dividend_yields(dividend_file)
            # Take average yield across tickers for each date (simplified)
            daily_yields = dividend_data.groupby('date')['dividend_yield'].mean()
            dividend_series = interpolate_rates_for_dates(daily_yields, options_dates)
            logger.info("Using daily dividend yields")
        except Exception as e:
            logger.warning(f"Failed to load daily dividend yields, using constant: {e}")
            q = div_config.get('constant_yield', 0.0)
            dividend_series = pd.Series(q, index=options_dates, name='dividend')
    else:
        q = div_config.get('constant_yield', 0.0)
        dividend_series = pd.Series(q, index=options_dates, name='dividend')
    
    return {
        'riskfree': riskfree_series,
        'dividend': dividend_series
    }


def estimate_rates_from_options(
    options_df: pd.DataFrame,
    method: str = 'put_call_parity'
) -> Dict[str, pd.Series]:
    """
    Estimate risk-free rates and dividend yields from option prices using arbitrage relationships.
    
    Args:
        options_df: Options DataFrame with call and put prices
        method: Estimation method ('put_call_parity')
        
    Returns:
        Dictionary with estimated 'riskfree' and 'dividend' Series
    """
    if method == 'put_call_parity':
        return estimate_rates_put_call_parity(options_df)
    else:
        raise ValueError(f"Unknown rate estimation method: {method}")


def estimate_rates_put_call_parity(options_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Estimate rates using put-call parity for ATM options.
    
    Put-call parity: C - P = S*exp(-q*T) - K*exp(-r*T)
    
    Args:
        options_df: Options DataFrame with call and put prices
        
    Returns:
        Dictionary with estimated rates
    """
    # Find matched call-put pairs
    calls = options_df[options_df['type'] == 'call'].copy()
    puts = options_df[options_df['type'] == 'put'].copy()
    
    # Merge on date, underlying, expiration, strike
    merge_cols = ['date', 'underlying', 'expiration', 'strike', 'S', 'T']
    pairs = calls.merge(
        puts[merge_cols + ['mid']],
        on=merge_cols,
        suffixes=('_call', '_put')
    )
    
    if len(pairs) == 0:
        logger.warning("No call-put pairs found for rate estimation")
        return {
            'riskfree': pd.Series([0.02]),
            'dividend': pd.Series([0.0])
        }
    
    # Focus on near-ATM options (|log_moneyness| < 0.1)
    pairs = pairs[pairs['log_moneyness'].abs() < 0.1]
    
    if len(pairs) == 0:
        logger.warning("No ATM call-put pairs found for rate estimation")
        return {
            'riskfree': pd.Series([0.02]),
            'dividend': pd.Series([0.0])
        }
    
    # Estimate implied forward and discount factor
    # C - P = F*exp(-r*T) - K*exp(-r*T) = (F - K)*exp(-r*T)
    # where F = S*exp((r-q)*T)
    
    call_put_diff = pairs['mid_call'] - pairs['mid_put']
    forward_strike_diff = pairs['S'] - pairs['strike']  # Approximation for ATM
    
    # Estimate discount factor (simplified)
    estimated_discount = call_put_diff / forward_strike_diff
    estimated_discount = estimated_discount.clip(0.5, 1.0)  # Reasonable bounds
    
    # Estimate risk-free rate
    estimated_r = -np.log(estimated_discount) / pairs['T']
    estimated_r = estimated_r.clip(0.0, 0.1)  # Reasonable bounds
    
    # Group by date and take median
    daily_rates = pairs.groupby('date').agg({
        'estimated_r': 'median'
    })['estimated_r']
    
    # For simplicity, assume zero dividend yield
    daily_dividends = pd.Series(0.0, index=daily_rates.index)
    
    logger.info(f"Estimated rates from put-call parity: {len(daily_rates)} days, "
               f"mean r = {daily_rates.mean():.3f}")
    
    return {
        'riskfree': daily_rates,
        'dividend': daily_dividends
    }