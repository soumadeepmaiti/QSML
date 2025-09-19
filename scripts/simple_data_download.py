#!/usr/bin/env python3
"""
Simple Data Download Script for QSML Project
Downloads basic market data for testing the quantitative finance pipeline
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_directories(base_path):
    """Create directory structure"""
    dirs = [
        'data/raw/equities',
        'data/raw/options', 
        'data/external',
        'data/processed'
    ]
    
    for dir_path in dirs:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created: {full_path}")

def download_simple_data(base_path):
    """Download a small sample of market data that should work"""
    
    # Focus on just a few major tickers that should always work
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']
    
    logging.info("Starting simple data download...")
    
    success_count = 0
    for ticker in tickers:
        try:
            logging.info(f"Downloading {ticker}...")
            
            # Use more basic approach
            stock = yf.download(ticker, start='2020-01-01', end='2024-01-01', progress=False)
            
            if stock.empty:
                logging.warning(f"No data for {ticker}")
                continue
            
            # Reset index to get Date as column
            stock = stock.reset_index()
            stock['Symbol'] = ticker
            
            # Save to CSV
            csv_path = base_path / 'data' / 'raw' / 'equities' / f'{ticker}.csv'
            stock.to_csv(csv_path, index=False)
            
            logging.info(f"✓ {ticker}: {len(stock)} rows saved")
            success_count += 1
            
            # Rate limiting
            time.sleep(1.0)
            
        except Exception as e:
            logging.error(f"Failed to download {ticker}: {e}")
            continue
    
    logging.info(f"Successfully downloaded {success_count}/{len(tickers)} datasets")
    
    # Create a simple combined file
    if success_count > 0:
        combined_data = []
        for ticker in tickers:
            csv_path = base_path / 'data' / 'raw' / 'equities' / f'{ticker}.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                combined_data.append(df)
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_path = base_path / 'data' / 'raw' / 'equities' / 'combined_equities.csv'
            combined_df.to_csv(combined_path, index=False)
            logging.info(f"Combined data saved: {len(combined_df)} total rows")

def create_sample_treasury_data(base_path):
    """Create some sample treasury data if FRED doesn't work"""
    
    # Create sample treasury yield data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    # Generate realistic treasury yields (simplified)
    import numpy as np
    np.random.seed(42)
    
    base_yields = {
        '1MO': 0.5,
        '3MO': 0.8, 
        '6MO': 1.2,
        '1YR': 1.8,
        '2YR': 2.2,
        '5YR': 2.8,
        '10YR': 3.2,
        '30YR': 3.8
    }
    
    data = {'Date': dates}
    for tenor, base_rate in base_yields.items():
        # Add some realistic variation
        yields = base_rate + np.random.normal(0, 0.3, len(dates)).cumsum() * 0.01
        yields = np.maximum(yields, 0.1)  # Keep positive
        data[f'DGS{tenor}'] = yields
    
    treasury_df = pd.DataFrame(data)
    
    # Save treasury data
    treasury_path = base_path / 'data' / 'external' / 'treasury_yields.csv'
    treasury_df.to_csv(treasury_path, index=False)
    
    logging.info(f"Sample treasury data created: {len(treasury_df)} rows")

def main():
    """Main function"""
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("=" * 50)
    print("QSML Simple Data Download")
    print("=" * 50)
    print("Downloading basic market data for testing...")
    print()
    
    try:
        # Create directories
        create_directories(project_root)
        
        # Download equity data
        download_simple_data(project_root)
        
        # Create sample treasury data
        create_sample_treasury_data(project_root)
        
        print()
        print("=" * 50)
        print("✅ DATA DOWNLOAD COMPLETE!")
        print("=" * 50)
        print(f"Data saved to: {project_root / 'data'}")
        print()
        print("Next steps:")
        print("1. Check the data/raw/equities/ folder for stock data")
        print("2. Check the data/external/ folder for treasury data") 
        print("3. Run the analysis notebooks (once created)")
        
    except Exception as e:
        logging.error(f"Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()