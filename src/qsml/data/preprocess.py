import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_mid_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'mid' column = (bid+ask)/2 and drop rows with invalid quotes.
    
    Args:
        df: Options DataFrame with 'bid' and 'ask' columns
        
    Returns:
        DataFrame with 'mid' column added and invalid rows removed
    """
    df = df.copy()
    
    # Calculate mid price
    df['mid'] = (df['bid'] + df['ask']) / 2
    
    # Remove invalid quotes
    initial_rows = len(df)
    
    # Remove where both bid and ask are zero
    df = df[~((df['bid'] == 0) & (df['ask'] == 0))]
    
    # Remove where bid > ask (crossed quotes)
    df = df[df['bid'] <= df['ask']]
    
    # Remove where mid price is zero or negative
    df = df[df['mid'] > 0]
    
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} rows with invalid quotes ({removed_rows/initial_rows:.1%})")
    
    return df


def add_moneyness_T(options: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """
    Merge options with spot prices and add moneyness and time to expiration.
    
    Args:
        options: Options DataFrame with columns ['date', 'underlying', 'expiration', 'strike', ...]
        spot: Spot price DataFrame with columns ['date', 'ticker', 'close', ...]
        
    Returns:
        Merged DataFrame with 'log_moneyness', 'T', and 'S' columns added
    """
    # Rename ticker to underlying for consistent merging
    spot_renamed = spot.rename(columns={'ticker': 'underlying', 'close': 'S'})
    
    # Merge options with spot prices on date and underlying
    merged = options.merge(
        spot_renamed[['date', 'underlying', 'S']],
        on=['date', 'underlying'],
        how='inner'
    )
    
    initial_rows = len(options)
    merged_rows = len(merged)
    
    if merged_rows < initial_rows:
        logger.warning(f"Lost {initial_rows - merged_rows} options rows in spot price merge "
                      f"({(initial_rows - merged_rows)/initial_rows:.1%})")
    
    # Calculate log-moneyness: ln(K/S)
    merged['log_moneyness'] = np.log(merged['strike'] / merged['S'])
    
    # Calculate time to expiration in years (ACT/365)
    merged['days_to_expiry'] = (merged['expiration'] - merged['date']).dt.days
    merged['T'] = merged['days_to_expiry'] / 365.0
    
    # Remove options with negative or zero time to expiry
    merged = merged[merged['T'] > 0]
    
    logger.info(f"Added moneyness and T: {len(merged)} rows, "
                f"log-moneyness range: [{merged['log_moneyness'].min():.2f}, {merged['log_moneyness'].max():.2f}], "
                f"T range: [{merged['T'].min():.3f}, {merged['T'].max():.3f}] years")
    
    return merged


def filter_chain(
    df: pd.DataFrame,
    min_dte: int = 5,
    max_abs_lm: float = 1.5,
    min_oi: int = 10,
    min_volume: int = 1,
    max_bid_ask_spread_pct: float = 0.5
) -> pd.DataFrame:
    """
    Apply filters to ensure reasonable options for calibration.
    
    Args:
        df: Options DataFrame with required columns
        min_dte: Minimum days to expiration
        max_abs_lm: Maximum absolute log-moneyness
        min_oi: Minimum open interest
        min_volume: Minimum volume
        max_bid_ask_spread_pct: Maximum bid-ask spread as % of mid price
        
    Returns:
        Filtered DataFrame
    """
    initial_rows = len(df)
    logger.info(f"Starting filter_chain with {initial_rows} rows")
    
    # Filter by days to expiration
    if 'days_to_expiry' in df.columns:
        df = df[df['days_to_expiry'] >= min_dte]
        logger.info(f"After min DTE filter ({min_dte}): {len(df)} rows")
    
    # Filter by absolute log-moneyness
    if 'log_moneyness' in df.columns:
        df = df[df['log_moneyness'].abs() <= max_abs_lm]
        logger.info(f"After log-moneyness filter (±{max_abs_lm}): {len(df)} rows")
    
    # Filter by open interest
    if 'open_interest' in df.columns and min_oi > 0:
        df = df[df['open_interest'] >= min_oi]
        logger.info(f"After open interest filter (≥{min_oi}): {len(df)} rows")
    
    # Filter by volume
    if 'volume' in df.columns and min_volume > 0:
        df = df[df['volume'] >= min_volume]
        logger.info(f"After volume filter (≥{min_volume}): {len(df)} rows")
    
    # Filter by bid-ask spread
    if 'bid' in df.columns and 'ask' in df.columns and 'mid' in df.columns:
        spread_pct = (df['ask'] - df['bid']) / df['mid']
        df = df[spread_pct <= max_bid_ask_spread_pct]
        logger.info(f"After bid-ask spread filter (≤{max_bid_ask_spread_pct:.1%}): {len(df)} rows")
    
    removed_rows = initial_rows - len(df)
    logger.info(f"Filtering complete: removed {removed_rows} rows ({removed_rows/initial_rows:.1%})")
    
    return df


def compute_implied_volatility(
    df: pd.DataFrame,
    price_col: str = 'mid',
    S_col: str = 'S',
    r: float = 0.02,
    q: float = 0.0
) -> pd.DataFrame:
    """
    Compute implied volatility for options where it's missing.
    
    Args:
        df: Options DataFrame
        price_col: Column name for option prices
        S_col: Column name for spot prices
        r: Risk-free rate (constant)
        q: Dividend yield (constant)
        
    Returns:
        DataFrame with 'iv' column computed
    """
    from ..utils.math_helpers import prices_to_iv
    
    df = df.copy()
    
    # Only compute IV for rows where it's missing
    missing_iv = df['iv'].isna()
    n_missing = missing_iv.sum()
    
    if n_missing == 0:
        logger.info("All options already have implied volatility")
        return df
    
    logger.info(f"Computing implied volatility for {n_missing} options")
    
    # Compute IV for missing values
    for option_type in ['call', 'put']:
        mask = missing_iv & (df['type'] == option_type)
        if not mask.any():
            continue
        
        iv_computed = prices_to_iv(
            prices=df.loc[mask, price_col].values,
            S=df.loc[mask, S_col].values,
            r=r,
            q=q,
            T=df.loc[mask, 'T'].values,
            K=df.loc[mask, 'strike'].values,
            option_type=option_type
        )
        
        df.loc[mask, 'iv'] = iv_computed
    
    # Count successful computations
    still_missing = df['iv'].isna().sum()
    computed = n_missing - still_missing
    
    logger.info(f"Successfully computed IV for {computed}/{n_missing} options "
                f"({computed/n_missing:.1%} success rate)")
    
    return df


def add_liquidity_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add liquidity-based weights for calibration.
    
    Weights are based on:
    - Open interest (higher = more weight)
    - Volume (higher = more weight)  
    - Bid-ask spread (tighter = more weight)
    - Moneyness (ATM gets higher weight)
    
    Args:
        df: Options DataFrame
        
    Returns:
        DataFrame with 'weight' column added
    """
    df = df.copy()
    
    # Initialize weight to 1
    df['weight'] = 1.0
    
    # Weight by open interest (square root to avoid over-weighting)
    if 'open_interest' in df.columns:
        oi_weight = np.sqrt(df['open_interest'])
        oi_weight = oi_weight / oi_weight.median()  # Normalize
        df['weight'] *= oi_weight
    
    # Weight by volume (square root)
    if 'volume' in df.columns:
        vol_weight = np.sqrt(df['volume'].clip(lower=1))
        vol_weight = vol_weight / vol_weight.median()  # Normalize
        df['weight'] *= vol_weight
    
    # Weight by bid-ask spread (tighter spreads get higher weight)
    if 'bid' in df.columns and 'ask' in df.columns and 'mid' in df.columns:
        spread_pct = (df['ask'] - df['bid']) / df['mid']
        spread_weight = 1 / (1 + spread_pct)  # Inverse relationship
        df['weight'] *= spread_weight
    
    # Weight by moneyness (ATM gets higher weight)
    if 'log_moneyness' in df.columns:
        # Gaussian weight centered at ATM
        moneyness_weight = np.exp(-0.5 * (df['log_moneyness'] / 0.2) ** 2)
        df['weight'] *= moneyness_weight
    
    # Normalize weights to have mean 1
    df['weight'] = df['weight'] / df['weight'].mean()
    
    logger.info(f"Added liquidity weights: range [{df['weight'].min():.2f}, {df['weight'].max():.2f}], "
                f"mean: {df['weight'].mean():.2f}")
    
    return df


def bucket_by_maturity(
    df: pd.DataFrame,
    bucket_method: str = 'fixed',
    n_buckets: int = 4,
    bucket_boundaries: Optional[list] = None
) -> pd.DataFrame:
    """
    Bucket options by maturity for separate calibration.
    
    Args:
        df: Options DataFrame with 'T' column
        bucket_method: 'fixed' (predefined buckets) or 'quantile' (equal counts)
        n_buckets: Number of buckets for quantile method
        bucket_boundaries: Custom bucket boundaries in years
        
    Returns:
        DataFrame with 'maturity_bucket' column added
    """
    df = df.copy()
    
    if bucket_method == 'fixed':
        # Use predefined maturity buckets
        if bucket_boundaries is None:
            bucket_boundaries = [0, 0.083, 0.25, 0.5, 1.0, 2.0, np.inf]  # 1M, 3M, 6M, 1Y, 2Y+
        
        bucket_labels = [f"{bucket_boundaries[i]:.2f}-{bucket_boundaries[i+1]:.2f}Y" 
                        for i in range(len(bucket_boundaries)-1)]
        bucket_labels[-1] = f"{bucket_boundaries[-2]:.2f}Y+"
        
        df['maturity_bucket'] = pd.cut(
            df['T'],
            bins=bucket_boundaries,
            labels=bucket_labels,
            include_lowest=True
        )
        
    elif bucket_method == 'quantile':
        # Use quantile-based buckets for equal counts
        df['maturity_bucket'] = pd.qcut(
            df['T'],
            q=n_buckets,
            labels=[f"Q{i+1}" for i in range(n_buckets)],
            duplicates='drop'
        )
    
    else:
        raise ValueError(f"Unknown bucket_method: {bucket_method}")
    
    # Count options per bucket
    bucket_counts = df['maturity_bucket'].value_counts().sort_index()
    logger.info(f"Maturity buckets:\n{bucket_counts}")
    
    return df


def prepare_calibration_data(
    options_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for calibration.
    
    Args:
        options_df: Raw options DataFrame
        spot_df: Raw spot prices DataFrame  
        config: Configuration dictionary with filter parameters
        
    Returns:
        Processed DataFrame ready for calibration
    """
    logger.info("Starting calibration data preparation")
    
    # Step 1: Compute mid prices
    processed_df = compute_mid_price(options_df)
    
    # Step 2: Merge with spot prices and add moneyness/time
    processed_df = add_moneyness_T(processed_df, spot_df)
    
    # Step 3: Apply filters
    filter_params = config.get('filters', {})
    processed_df = filter_chain(
        processed_df,
        min_dte=filter_params.get('min_days_to_expiry', 5),
        max_abs_lm=filter_params.get('max_log_moneyness_abs', 1.5),
        min_oi=filter_params.get('min_open_interest', 10),
        min_volume=filter_params.get('min_volume', 1),
        max_bid_ask_spread_pct=filter_params.get('max_bid_ask_spread_pct', 0.5)
    )
    
    # Step 4: Compute implied volatility if missing
    r = config.get('riskfree', {}).get('constant_rate', 0.02)
    q = config.get('dividends', {}).get('constant_yield', 0.0)
    processed_df = compute_implied_volatility(processed_df, r=r, q=q)
    
    # Step 5: Add liquidity weights
    processed_df = add_liquidity_weights(processed_df)
    
    # Step 6: Bucket by maturity
    processed_df = bucket_by_maturity(processed_df)
    
    logger.info(f"Calibration data preparation complete: {len(processed_df)} options ready")
    
    return processed_df