#!/usr/bin/env python3
"""
Data Download Script for QSML Project
=====================================

Downloads real market data from various sources:
- Yahoo Finance: Equity data (S&P 500 stocks)
- FRED (Federal Reserve): Treasury yields, economic indicators
- CBOE: VIX and volatility data
- Alpha Vantage: Additional financial data (requires API key)

This script creates the proper directory structure and downloads authentic market data
for the quantitative finance pipeline.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import io
from datetime import datetime, timedelta
import time
import logging
from typing import List
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataDownloader:
    """Download real market data from various sources."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """Create the required directory structure."""
        directories = [
            self.data_dir / "raw" / "equities",
            self.data_dir / "raw" / "options", 
            self.data_dir / "external",
            self.data_dir / "processed"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Created directory structure in {self.data_dir}")

    def get_sp500_tickers(self):
        """Get S&P 500 ticker list from Wikipedia."""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            
            # Clean tickers (remove dots which cause issues)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            logger.info(f"Retrieved {len(tickers)} S&P 500 tickers")
            return tickers[:50]  # Limit to first 50 for demo
            
        except Exception as e:
            logger.warning(f"Could not fetch S&P 500 list: {e}")
            # Fallback to major stocks
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
                'JPM', 'JNJ', 'V', 'PG', 'UNH', 'MA', 'HD', 'BAC', 'ABBV', 'PFE',
                'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'DIS', 'ABT', 'CRM', 'VZ',
                'ADBE', 'DHR', 'NKE', 'LIN', 'NEE', 'XOM', 'TXN', 'QCOM', 'CVX',
                'WMT', 'BMY', 'RTX', 'LOW', 'ORCL', 'AMD', 'SCHW', 'PM', 'HON',
                'INTU', 'C', 'GS', 'SPGI', 'SPY'  # Include SPY ETF
            ]

    def download_equity_data(self, tickers: List[str] = None):
        """Download equity OHLCV data from Yahoo Finance"""
        if not tickers:
            tickers = self.get_sp500_tickers()
        
        os.makedirs(self.data_dir / 'raw' / 'equities', exist_ok=True)
        
        logging.info(f"Downloading equity data for {len(tickers)} tickers...")
        all_data = []
        failed = []
        
        # Set user agent to avoid blocking
        import requests
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
        
        for i, ticker in enumerate(tickers, 1):
            logging.info(f"Downloading {ticker} ({i}/{len(tickers)})")
            
            try:
                # Download with session and longer period
                stock = yf.Ticker(ticker, session=session)
                data = stock.history(period="max", interval="1d")
                
                if data.empty:
                    # Try with different parameters
                    data = stock.history(start="2019-01-01", end="2024-12-31")
                
                if data.empty:
                    logging.warning(f"No data for {ticker}")
                    failed.append(ticker)
                    continue
                
                # Reset index to make Date a column
                data = data.reset_index()
                data['Symbol'] = ticker
                
                # Clean column names
                data.columns = [col.replace(' ', '_').lower() for col in data.columns]
                
                # Save individual file
                csv_path = self.data_dir / 'raw' / 'equities' / f'{ticker}.csv'
                data.to_csv(csv_path, index=False)
                
                all_data.append(data)
                logging.info(f"✓ {ticker}: {len(data)} rows")
                
                # Rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                logging.error(f"Error downloading {ticker}: {e}")
                failed.append(ticker)
                time.sleep(1.0)  # Longer wait on error
        
        if all_data:
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_path = self.data_dir / 'raw' / 'equities' / 'all_equities.csv'
            combined_df.to_csv(combined_path, index=False)
            logging.info(f"Successfully downloaded {len(all_data)} equity datasets")
        else:
            logging.info("Successfully downloaded 0 equity datasets")
        
        if failed:
            logging.warning(f"Failed downloads: {failed}")
        
        return all_data, failed

    def download_treasury_data(self):
        """Download Treasury yield data from FRED."""
        logger.info("Downloading Treasury yield data from FRED...")
        
        # Treasury securities and their FRED series IDs
        treasury_series = {
            '1MO': 'DGS1MO',    # 1-Month Treasury
            '3MO': 'DGS3MO',    # 3-Month Treasury  
            '6MO': 'DGS6MO',    # 6-Month Treasury
            '1YR': 'DGS1',      # 1-Year Treasury
            '2YR': 'DGS2',      # 2-Year Treasury
            '3YR': 'DGS3',      # 3-Year Treasury
            '5YR': 'DGS5',      # 5-Year Treasury
            '7YR': 'DGS7',      # 7-Year Treasury
            '10YR': 'DGS10',    # 10-Year Treasury
            '20YR': 'DGS20',    # 20-Year Treasury
            '30YR': 'DGS30'     # 30-Year Treasury
        }
        
        all_yields = []
        
        for tenor, series_id in treasury_series.items():
            try:
                # FRED API endpoint
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
                
                # Download data
                response = requests.get(url)
                response.raise_for_status()
                
                # Parse CSV
                df = pd.read_csv(io.StringIO(response.text))
                df.columns = ['Date', 'Yield']
                
                # Clean data
                df['Date'] = pd.to_datetime(df['Date'])
                df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce') / 100  # Convert to decimal
                df = df.dropna()
                
                # Add tenor information
                df['Tenor'] = tenor
                df['Years'] = {
                    '1MO': 1/12, '3MO': 0.25, '6MO': 0.5, '1YR': 1, '2YR': 2,
                    '3YR': 3, '5YR': 5, '7YR': 7, '10YR': 10, '20YR': 20, '30YR': 30
                }[tenor]
                
                all_yields.append(df)
                logger.info(f"Downloaded {tenor} Treasury data: {len(df)} rows")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Failed to download {tenor} Treasury data: {e}")
        
        if all_yields:
            # Combine all yield data
            combined_yields = pd.concat(all_yields, ignore_index=True)
            combined_yields = combined_yields.sort_values(['Date', 'Years'])
            
            # Save to external folder
            filepath = self.data_dir / "external" / "treasury_yields.csv"
            combined_yields.to_csv(filepath, index=False)
            
            logger.info(f"Saved Treasury yields: {len(combined_yields):,} rows")
            return len(combined_yields)
        
        return 0

    def download_vix_data(self):
        """Download VIX volatility data"""
        os.makedirs(self.data_dir / 'external', exist_ok=True)
        
        logging.info("Downloading VIX data...")
        
        try:
            # Try with session and user agent
            import requests
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
            
            vix = yf.Ticker("^VIX", session=session)
            data = vix.history(period="max", interval="1d")
            
            if data.empty:
                # Try alternative period
                data = vix.history(start="2019-01-01", end="2024-12-31")
            
            if data.empty:
                logging.warning("No VIX data available")
                return None
            
            # Reset index and clean
            data = data.reset_index()
            data.columns = [col.replace(' ', '_').lower() for col in data.columns]
            
            csv_path = self.data_dir / 'external' / 'vix.csv'
            data.to_csv(csv_path, index=False)
            
            logging.info(f"✓ VIX data: {len(data)} rows")
            return data
            
        except Exception as e:
            logging.error(f"Failed to download VIX: {e}")
            return None

    def download_spy_options_sample(self):
        """Download sample SPY options data (limited by Yahoo Finance)."""
        logger.info("Downloading SPY options data...")
        
        try:
            spy = yf.Ticker("SPY")
            
            # Get available expiration dates
            expirations = spy.options
            
            if not expirations:
                logger.warning("No options data available for SPY")
                return 0
            
            all_options = []
            
            # Download first few expiration dates
            for exp_date in expirations[:5]:  # Limit to first 5 expirations
                try:
                    logger.info(f"Downloading options for expiration: {exp_date}")
                    
                    # Get options chain
                    options_chain = spy.option_chain(exp_date)
                    
                    # Process calls
                    calls = options_chain.calls.copy()
                    calls['Type'] = 'call'
                    calls['Expiration'] = exp_date
                    calls['UnderlyingSymbol'] = 'SPY'
                    
                    # Process puts
                    puts = options_chain.puts.copy() 
                    puts['Type'] = 'put'
                    puts['Expiration'] = exp_date
                    puts['UnderlyingSymbol'] = 'SPY'
                    
                    # Combine
                    exp_options = pd.concat([calls, puts], ignore_index=True)
                    
                    # Add current date
                    exp_options['DataDate'] = datetime.now().strftime('%Y-%m-%d')
                    
                    all_options.append(exp_options)
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Failed to download options for {exp_date}: {e}")
            
            if all_options:
                # Combine all options data
                combined_options = pd.concat(all_options, ignore_index=True)
                
                # Clean column names
                column_mapping = {
                    'contractSymbol': 'ContractSymbol',
                    'lastTradeDate': 'LastTradeDate',
                    'strike': 'Strike',
                    'lastPrice': 'Last',
                    'bid': 'Bid',
                    'ask': 'Ask',
                    'change': 'Change',
                    'percentChange': 'PercentChange',
                    'volume': 'Volume',
                    'openInterest': 'OpenInterest',
                    'impliedVolatility': 'IV'
                }
                
                # Rename columns that exist
                existing_columns = {k: v for k, v in column_mapping.items() 
                                  if k in combined_options.columns}
                combined_options = combined_options.rename(columns=existing_columns)
                
                # Save options data
                filepath = self.data_dir / "raw" / "options" / "SPY_options_sample.csv"
                combined_options.to_csv(filepath, index=False)
                
                logger.info(f"Downloaded SPY options: {len(combined_options)} contracts")
                return len(combined_options)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to download SPY options: {e}")
            return 0

    def download_economic_indicators(self):
        """Download additional economic indicators."""
        logger.info("Downloading economic indicators...")
        
        indicators = {
            'FEDFUNDS': 'Federal Funds Rate',
            'UNRATE': 'Unemployment Rate', 
            'CPIAUCSL': 'Consumer Price Index',
            'GDP': 'Gross Domestic Product'
        }
        
        economic_data = []
        
        for series_id, description in indicators.items():
            try:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
                response = requests.get(url)
                response.raise_for_status()
                
                df = pd.read_csv(io.StringIO(response.text))
                df.columns = ['Date', 'Value']
                df['Date'] = pd.to_datetime(df['Date'])
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                df = df.dropna()
                
                df['Indicator'] = series_id
                df['Description'] = description
                
                economic_data.append(df)
                logger.info(f"Downloaded {description}: {len(df)} rows")
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to download {description}: {e}")
        
        if economic_data:
            combined_indicators = pd.concat(economic_data, ignore_index=True)
            filepath = self.data_dir / "external" / "economic_indicators.csv"
            combined_indicators.to_csv(filepath, index=False)
            
            logger.info(f"Saved economic indicators: {len(combined_indicators):,} rows")
            return len(combined_indicators)
        
        return 0

    def process_to_parquet(self):
        """Process raw CSV data and convert to parquet format."""
        logger.info("Processing data to parquet format...")
        
        try:
            # Process equities
            equity_files = list((self.data_dir / "raw" / "equities").glob("*.csv"))
            if equity_files:
                equity_dfs = []
                for file in equity_files:
                    df = pd.read_csv(file)
                    df['Date'] = pd.to_datetime(df['Date'])
                    equity_dfs.append(df)
                
                equities_combined = pd.concat(equity_dfs, ignore_index=True)
                equities_combined = equities_combined.sort_values(['Symbol', 'Date'])
                
                output_path = self.data_dir / "processed" / "equities.parquet"
                equities_combined.to_parquet(output_path, index=False)
                logger.info(f"Processed equities: {len(equities_combined):,} rows")
            
            # Process treasury yields
            treasury_file = self.data_dir / "external" / "treasury_yields.csv"
            if treasury_file.exists():
                treasury_df = pd.read_csv(treasury_file)
                treasury_df['Date'] = pd.to_datetime(treasury_df['Date'])
                
                output_path = self.data_dir / "processed" / "riskfree_daily.parquet"
                treasury_df.to_parquet(output_path, index=False)
                logger.info(f"Processed treasury yields: {len(treasury_df):,} rows")
            
            # Process options (if any)
            options_files = list((self.data_dir / "raw" / "options").glob("*.csv"))
            if options_files:
                options_dfs = []
                for file in options_files:
                    df = pd.read_csv(file)
                    if 'DataDate' in df.columns:
                        df['DataDate'] = pd.to_datetime(df['DataDate'])
                    if 'Expiration' in df.columns:
                        df['Expiration'] = pd.to_datetime(df['Expiration'])
                    options_dfs.append(df)
                
                options_combined = pd.concat(options_dfs, ignore_index=True)
                
                # Clean options data
                if 'Bid' in options_combined.columns and 'Ask' in options_combined.columns:
                    options_combined = options_combined[
                        (options_combined['Bid'] > 0) & 
                        (options_combined['Ask'] > options_combined['Bid'])
                    ].copy()
                
                output_path = self.data_dir / "processed" / "options_clean.parquet"
                options_combined.to_parquet(output_path, index=False)
                logger.info(f"Processed options: {len(options_combined):,} rows")
            
            # Create dividend yield placeholder (real dividend data would need separate source)
            spy_file = self.data_dir / "raw" / "equities" / "SPY_data.csv"
            if spy_file.exists():
                spy_df = pd.read_csv(spy_file)
                spy_df['Date'] = pd.to_datetime(spy_df['Date'])
                
                # Estimate dividend yield (placeholder - would need real dividend data)
                spy_df['DividendYield'] = 0.015  # Approximate SPY yield
                spy_df = spy_df[['Date', 'Symbol', 'DividendYield']].copy()
                
                output_path = self.data_dir / "processed" / "div_yield.parquet"
                spy_df.to_parquet(output_path, index=False)
                logger.info(f"Created dividend yield placeholder: {len(spy_df):,} rows")
            
        except Exception as e:
            logger.error(f"Error processing to parquet: {e}")

    def download_all(self):
        """Download all available data sources."""
        logger.info("Starting comprehensive data download...")
        
        results = {}
        
        # Download equity data
        successful, failed = self.download_equity_data()
        results['equities'] = {'successful': successful, 'failed': len(failed)}
        
        # Download Treasury yields
        treasury_rows = self.download_treasury_data()
        results['treasury'] = {'rows': treasury_rows}
        
        # Download VIX data
        vix_rows = self.download_vix_data()
        results['vix'] = {'rows': vix_rows}
        
        # Download SPY options sample
        options_rows = self.download_spy_options_sample()
        results['options'] = {'rows': options_rows}
        
        # Download economic indicators
        econ_rows = self.download_economic_indicators()
        results['economic'] = {'rows': econ_rows}
        
        # Process to parquet
        self.process_to_parquet()
        
        # Summary
        logger.info("=" * 60)
        logger.info("DATA DOWNLOAD COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Equities: {results['equities']['successful']} successful downloads")
        logger.info(f"Treasury yields: {results['treasury']['rows']:,} rows")
        logger.info(f"VIX data: {results['vix']['rows']:,} rows") 
        logger.info(f"Options data: {results['options']['rows']:,} rows")
        logger.info(f"Economic indicators: {results['economic']['rows']:,} rows")
        logger.info(f"Data saved to: {self.data_dir.absolute()}")
        
        return results

def main():
    """Main download script."""
    print("QSML Data Download Script")
    print("=" * 50)
    print("Downloading real market data from:")
    print("- Yahoo Finance (Equities, VIX, Options)")
    print("- FRED (Treasury yields, Economic indicators)")
    print()
    
    # Initialize downloader
    downloader = DataDownloader()
    
    # Download all data
    results = downloader.download_all()
    
    print("\n🎉 Data download completed!")
    print(f"📁 Check the 'data' folder for all downloaded files")
    print(f"📊 Processed data available in 'data/processed/' as parquet files")
    
    return results

if __name__ == "__main__":
    main()