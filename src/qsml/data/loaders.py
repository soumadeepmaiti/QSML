import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_equities(raw_dir: str, file_pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load raw equities (OHLCV) data from CSV files.
    
    Expected input format from Kaggle datasets:
    - Columns: Date, Open, High, Low, Close, Adj Close, Volume
    - Or: date, symbol, open, high, low, close, adj_close, volume
    
    Args:
        raw_dir: Directory containing equity CSV files
        file_pattern: Pattern to match CSV files
        
    Returns:
        DataFrame with standardized columns:
        ['date','ticker','open','high','low','close','adj_close','volume']
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw equities directory not found: {raw_dir}")
    
    csv_files = list(raw_path.glob(file_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir} matching {file_pattern}")
    
    logger.info(f"Found {len(csv_files)} equity files to load")
    
    all_data = []
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # Try to infer ticker from filename if not in data
            if 'ticker' not in df.columns and 'symbol' not in df.columns:
                ticker = file_path.stem.upper()
                if ticker.endswith('_HISTORICAL_DATA'):
                    ticker = ticker.replace('_HISTORICAL_DATA', '')
                df['ticker'] = ticker
            
            # Standardize column names
            df = standardize_equity_columns(df)
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Basic data validation
            df = validate_equity_data(df)
            
            all_data.append(df)
            logger.debug(f"Loaded {len(df)} rows from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid equity data could be loaded")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by ticker and date
    combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    logger.info(f"Loaded equity data: {len(combined_df)} rows, "
                f"{combined_df['ticker'].nunique()} tickers, "
                f"date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    return combined_df


def load_options(raw_dir: str, file_pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load raw option chains data from CSV files.
    
    Expected input format:
    - Columns include: date, underlying, expiration, strike, type/option_type, 
      bid, ask, last, iv (optional), open_interest, volume
    
    Args:
        raw_dir: Directory containing options CSV files
        file_pattern: Pattern to match CSV files
        
    Returns:
        DataFrame with standardized columns:
        ['date','underlying','expiration','strike','type','bid','ask','last','iv','open_interest','volume']
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw options directory not found: {raw_dir}")
    
    csv_files = list(raw_path.glob(file_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir} matching {file_pattern}")
    
    logger.info(f"Found {len(csv_files)} option files to load")
    
    all_data = []
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # Standardize column names
            df = standardize_options_columns(df)
            
            # Convert date columns
            df['date'] = pd.to_datetime(df['date'])
            df['expiration'] = pd.to_datetime(df['expiration'])
            
            # Basic data validation
            df = validate_options_data(df)
            
            all_data.append(df)
            logger.debug(f"Loaded {len(df)} rows from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid options data could be loaded")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by underlying, date, expiration, strike
    combined_df = combined_df.sort_values(
        ['underlying', 'date', 'expiration', 'strike']
    ).reset_index(drop=True)
    
    logger.info(f"Loaded options data: {len(combined_df)} rows, "
                f"{combined_df['underlying'].nunique()} underlyings, "
                f"date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    return combined_df


def standardize_equity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize equity DataFrame column names.
    
    Args:
        df: Input DataFrame with various column naming conventions
        
    Returns:
        DataFrame with standardized columns
    """
    # Create mapping for common column name variations
    column_mapping = {
        # Date columns
        'Date': 'date',
        'DATE': 'date',
        'timestamp': 'date',
        'Timestamp': 'date',
        
        # Price columns
        'Open': 'open',
        'OPEN': 'open',
        'High': 'high',
        'HIGH': 'high',
        'Low': 'low',
        'LOW': 'low',
        'Close': 'close',
        'CLOSE': 'close',
        'Adj Close': 'adj_close',
        'Adjusted_Close': 'adj_close',
        'adjusted_close': 'adj_close',
        
        # Volume
        'Volume': 'volume',
        'VOLUME': 'volume',
        'vol': 'volume',
        
        # Symbol/Ticker
        'Symbol': 'ticker',
        'SYMBOL': 'ticker',
        'symbol': 'ticker',
        'Ticker': 'ticker',
        'TICKER': 'ticker',
    }
    
    # Apply column mapping
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create adj_close if it doesn't exist (use close as fallback)
    if 'adj_close' not in df.columns:
        df['adj_close'] = df['close']
    
    # Create ticker if it doesn't exist
    if 'ticker' not in df.columns:
        df['ticker'] = 'UNKNOWN'
    
    return df[['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]


def standardize_options_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize options DataFrame column names.
    
    Args:
        df: Input DataFrame with various column naming conventions
        
    Returns:
        DataFrame with standardized columns
    """
    # Create mapping for common column name variations
    column_mapping = {
        # Date columns
        'Date': 'date',
        'DATE': 'date',
        'trade_date': 'date',
        'quote_date': 'date',
        'Expiration': 'expiration',
        'expiry': 'expiration',
        'exp_date': 'expiration',
        'maturity': 'expiration',
        
        # Option identification
        'Strike': 'strike',
        'STRIKE': 'strike',
        'strike_price': 'strike',
        'Type': 'type',
        'TYPE': 'type',
        'option_type': 'type',
        'call_put': 'type',
        'cp_flag': 'type',
        
        # Underlying
        'Underlying': 'underlying',
        'UNDERLYING': 'underlying',
        'symbol': 'underlying',
        'Symbol': 'underlying',
        'ticker': 'underlying',
        'root': 'underlying',
        
        # Prices
        'Bid': 'bid',
        'BID': 'bid',
        'bid_price': 'bid',
        'Ask': 'ask',
        'ASK': 'ask',
        'ask_price': 'ask',
        'Last': 'last',
        'LAST': 'last',
        'last_price': 'last',
        'close': 'last',
        
        # Volume and Open Interest
        'Volume': 'volume',
        'VOLUME': 'volume',
        'vol': 'volume',
        'Open_Interest': 'open_interest',
        'OpenInterest': 'open_interest',
        'open_int': 'open_interest',
        'oi': 'open_interest',
        
        # Implied Volatility
        'IV': 'iv',
        'implied_vol': 'iv',
        'implied_volatility': 'iv',
        'impliedVolatility': 'iv',
    }
    
    # Apply column mapping
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_cols = ['date', 'underlying', 'expiration', 'strike', 'type', 'bid', 'ask']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create optional columns with defaults if missing
    if 'last' not in df.columns:
        df['last'] = (df['bid'] + df['ask']) / 2  # Use mid as fallback
    
    if 'iv' not in df.columns:
        df['iv'] = np.nan  # Will compute later
    
    if 'volume' not in df.columns:
        df['volume'] = 0
    
    if 'open_interest' not in df.columns:
        df['open_interest'] = 0
    
    # Standardize option type values
    df['type'] = df['type'].str.upper().map({'C': 'call', 'P': 'put', 'CALL': 'call', 'PUT': 'put'})
    
    return df[['date', 'underlying', 'expiration', 'strike', 'type', 'bid', 'ask', 'last', 'iv', 'open_interest', 'volume']]


def validate_equity_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean equity data.
    
    Args:
        df: Equity DataFrame to validate
        
    Returns:
        Cleaned DataFrame
    """
    initial_rows = len(df)
    
    # Remove rows with missing or invalid prices
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'adj_close'])
    df = df[(df[['open', 'high', 'low', 'close', 'adj_close']] > 0).all(axis=1)]
    
    # Basic price consistency checks
    df = df[df['high'] >= df['low']]
    df = df[df['high'] >= df[['open', 'close']].max(axis=1)]
    df = df[df['low'] <= df[['open', 'close']].min(axis=1)]
    
    # Remove extreme outliers (prices that change by more than 50% in one day)
    df = df.sort_values(['ticker', 'date'])
    df['price_change'] = df.groupby('ticker')['close'].pct_change()
    df = df[df['price_change'].abs() <= 0.5]
    df = df.drop('price_change', axis=1)
    
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} invalid equity rows ({removed_rows/initial_rows:.1%})")
    
    return df


def validate_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean options data.
    
    Args:
        df: Options DataFrame to validate
        
    Returns:
        Cleaned DataFrame
    """
    initial_rows = len(df)
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['date', 'underlying', 'expiration', 'strike', 'type'])
    
    # Remove rows with invalid prices
    df = df[(df['strike'] > 0)]
    df = df[(df['bid'] >= 0) & (df['ask'] >= 0)]
    df = df[df['ask'] >= df['bid']]  # Ask should be >= Bid
    
    # Remove rows where bid-ask spread is too wide (>50% of mid)
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['spread_pct'] = (df['ask'] - df['bid']) / df['mid']
    df = df[(df['spread_pct'] <= 0.5) | (df['mid'] == 0)]
    df = df.drop(['spread_pct'], axis=1)
    
    # Remove options with negative time to expiration
    df = df[df['expiration'] > df['date']]
    
    # Remove extreme outliers in implied volatility
    if 'iv' in df.columns and not df['iv'].isna().all():
        df = df[(df['iv'].isna()) | ((df['iv'] > 0) & (df['iv'] < 5.0))]
    
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} invalid option rows ({removed_rows/initial_rows:.1%})")
    
    return df


def sample_data_for_testing(
    df: pd.DataFrame,
    n_days: int = 30,
    n_underlyings: int = 1,
    start_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Sample a subset of data for testing purposes.
    
    Args:
        df: Full DataFrame
        n_days: Number of days to sample
        n_underlyings: Number of underlyings to include
        start_date: Start date for sampling (if None, uses random period)
        
    Returns:
        Sampled DataFrame
    """
    if 'underlying' in df.columns:
        # Options data
        underlyings = df['underlying'].value_counts().head(n_underlyings).index.tolist()
        df_filtered = df[df['underlying'].isin(underlyings)]
    elif 'ticker' in df.columns:
        # Equity data
        tickers = df['ticker'].value_counts().head(n_underlyings).index.tolist()
        df_filtered = df[df['ticker'].isin(tickers)]
    else:
        df_filtered = df
    
    # Sample date range
    if start_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = start_dt + pd.Timedelta(days=n_days)
        df_sampled = df_filtered[
            (df_filtered['date'] >= start_dt) & (df_filtered['date'] <= end_dt)
        ]
    else:
        # Random sampling
        available_dates = sorted(df_filtered['date'].unique())
        if len(available_dates) >= n_days:
            start_idx = np.random.randint(0, len(available_dates) - n_days + 1)
            selected_dates = available_dates[start_idx:start_idx + n_days]
            df_sampled = df_filtered[df_filtered['date'].isin(selected_dates)]
        else:
            df_sampled = df_filtered
    
    logger.info(f"Sampled {len(df_sampled)} rows for testing")
    return df_sampled