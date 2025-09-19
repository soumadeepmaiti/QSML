#!/usr/bin/env python3
"""
Create synthetic market data for testing the QSML pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def create_synthetic_equity_data(base_path):
    """Create realistic synthetic equity data"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define tickers and their characteristics
    tickers_data = {
        'AAPL': {'initial_price': 150, 'volatility': 0.25, 'drift': 0.08},
        'MSFT': {'initial_price': 300, 'volatility': 0.23, 'drift': 0.12},
        'GOOGL': {'initial_price': 2500, 'volatility': 0.28, 'drift': 0.10},
        'SPY': {'initial_price': 400, 'volatility': 0.18, 'drift': 0.09},
        'QQQ': {'initial_price': 350, 'volatility': 0.22, 'drift': 0.11}
    }
    
    # Date range
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    dates = dates[dates.dayofweek < 5]  # Only weekdays
    
    all_data = []
    
    for ticker, params in tickers_data.items():
        logging.info(f"Creating synthetic data for {ticker}")
        
        # Generate price series using geometric Brownian motion
        n_days = len(dates)
        dt = 1/252  # Daily time step
        
        # Random walks
        random_walks = np.random.normal(0, 1, n_days)
        
        # Price evolution
        returns = (params['drift'] - 0.5 * params['volatility']**2) * dt + \
                 params['volatility'] * np.sqrt(dt) * random_walks
        
        # Calculate prices
        log_prices = np.log(params['initial_price']) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        # Create OHLCV data
        # Open = previous close (with small gap)
        opens = np.roll(prices, 1)
        opens[0] = params['initial_price']
        opens = opens * (1 + np.random.normal(0, 0.005, n_days))  # Small gaps
        
        # High/Low based on daily volatility
        daily_vol = params['volatility'] / np.sqrt(252)
        highs = prices * (1 + np.abs(np.random.normal(0, daily_vol, n_days)))
        lows = prices * (1 - np.abs(np.random.normal(0, daily_vol, n_days)))
        
        # Ensure High >= Open, Close and Low <= Open, Close
        highs = np.maximum(highs, np.maximum(opens, prices))
        lows = np.minimum(lows, np.minimum(opens, prices))
        
        # Volume (somewhat correlated with price changes)
        price_changes = np.abs(np.diff(np.concatenate([[params['initial_price']], prices])))
        base_volume = 1000000 if ticker == 'SPY' else 500000
        volumes = base_volume * (1 + price_changes / np.mean(prices) * 2) * \
                 (1 + np.random.normal(0, 0.3, n_days))
        volumes = np.maximum(volumes, base_volume * 0.1)  # Minimum volume
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes.astype(int),
            'Symbol': ticker
        })
        
        # Save individual file
        csv_path = base_path / 'data' / 'raw' / 'equities' / f'{ticker}.csv'
        df.to_csv(csv_path, index=False)
        
        all_data.append(df)
        logging.info(f"✓ {ticker}: {len(df)} rows created")
    
    # Create combined file
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_path = base_path / 'data' / 'raw' / 'equities' / 'combined_equities.csv'
    combined_df.to_csv(combined_path, index=False)
    
    logging.info(f"Combined equity data: {len(combined_df)} total rows")
    return combined_df

def create_synthetic_options_data(base_path):
    """Create synthetic SPY options data"""
    
    np.random.seed(42)
    
    # Generate options for SPY only
    spy_price = 400
    
    # Create expiration dates (monthly options)
    start_date = pd.Timestamp('2024-01-01')
    exp_dates = pd.date_range(start_date + pd.DateOffset(months=1), 
                             periods=12, freq='MS') + pd.DateOffset(days=14)  # 3rd Friday approx
    
    options_data = []
    
    for exp_date in exp_dates:
        days_to_exp = (exp_date - start_date).days
        
        # Strike prices around current price
        strikes = np.arange(spy_price * 0.8, spy_price * 1.2, 5)
        
        for strike in strikes:
            # Simple Black-Scholes approximation for realistic prices
            moneyness = spy_price / strike
            time_to_exp = days_to_exp / 365
            
            # Implied volatility (varies by moneyness)
            iv = 0.20 + 0.05 * abs(1 - moneyness) + np.random.normal(0, 0.02)
            iv = max(0.05, min(iv, 0.60))  # Reasonable bounds
            
            # Rough option pricing
            intrinsic_call = max(0, spy_price - strike)
            intrinsic_put = max(0, strike - spy_price)
            
            time_value = spy_price * iv * np.sqrt(time_to_exp) * 0.4
            
            call_price = intrinsic_call + time_value
            put_price = intrinsic_put + time_value
            
            # Add some bid-ask spread
            spread = max(0.05, call_price * 0.02)
            
            # Call option
            options_data.append({
                'Date': start_date.strftime('%Y-%m-%d'),
                'Symbol': 'SPY',
                'Expiration': exp_date.strftime('%Y-%m-%d'),
                'Strike': strike,
                'Type': 'C',
                'Bid': max(0.01, call_price - spread/2),
                'Ask': call_price + spread/2,
                'Last': call_price,
                'Volume': np.random.randint(10, 1000),
                'OpenInt': np.random.randint(100, 10000),
                'IV': iv
            })
            
            # Put option
            options_data.append({
                'Date': start_date.strftime('%Y-%m-%d'),
                'Symbol': 'SPY',
                'Expiration': exp_date.strftime('%Y-%m-%d'),
                'Strike': strike,
                'Type': 'P',
                'Bid': max(0.01, put_price - spread/2),
                'Ask': put_price + spread/2,
                'Last': put_price,
                'Volume': np.random.randint(10, 1000),
                'OpenInt': np.random.randint(100, 10000),
                'IV': iv
            })
    
    # Create DataFrame
    options_df = pd.DataFrame(options_data)
    
    # Save options data
    options_path = base_path / 'data' / 'raw' / 'options' / 'spy_options.csv'
    options_df.to_csv(options_path, index=False)
    
    logging.info(f"Synthetic options data: {len(options_df)} rows created")
    return options_df

def main():
    """Main function"""
    
    project_root = Path(__file__).parent.parent
    
    print("Creating synthetic market data for QSML testing...")
    
    # Create equity data
    equity_df = create_synthetic_equity_data(project_root)
    
    # Create options data
    options_df = create_synthetic_options_data(project_root)
    
    print(f"\n✅ Synthetic data created:")
    print(f"  - Equity data: {len(equity_df)} rows")
    print(f"  - Options data: {len(options_df)} rows")
    print(f"  - Treasury data: Already available")
    print(f"\n📁 Data saved to: {project_root / 'data'}")

if __name__ == "__main__":
    main()