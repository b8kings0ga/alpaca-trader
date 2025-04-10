#!/usr/bin/env python3
"""
Entry point for the Alpaca Trading Bot.
"""
import argparse
import os
from src.bot import AlpacaBot
from src.yfinance_data import YFinanceData
from src.yfinance_db_client import YFinanceDBClient

def main():
    """
    Main entry point for the trading bot.
    """
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Alpaca Trading Bot')
        parser.add_argument('--wait-for-open', action='store_true', help='Wait for market to open before starting')
        parser.add_argument('--use-alpaca', action='store_true', help='Use Alpaca API for data instead of yfinance')
        parser.add_argument('--strategy', type=str, default='moving_average_crossover',
                            choices=['moving_average_crossover', 'rsi', 'ml', 'dual_ma_yf'],
                            help='Trading strategy to use')
        parser.add_argument('--data-source', type=str, default='yfinance',
                            choices=['yfinance', 'alpaca', 'yfinance-db'],
                            help='Data source to use (yfinance, alpaca, or yfinance-db)')
        parser.add_argument('--use-yfinance-db', action='store_true',
                            help='Use YFinance DB service for data')
        parser.add_argument('--use-real-positions', action='store_true',
                            help='Use real positions from Alpaca API instead of simulated positions')
        args = parser.parse_args()
        
        # Determine data source
        data_source = args.data_source
        
        # Check environment variable for YFinance DB
        use_yfinance_db = os.getenv('USE_YFINANCE_DB', 'false').lower() == 'true'
        
        # Command line argument overrides environment variable
        if args.use_yfinance_db:
            use_yfinance_db = True
            data_source = 'yfinance-db'
            
        # Alpaca option takes precedence
        if args.use_alpaca:
            data_source = 'alpaca'
            use_yfinance_db = False
        
        # Set USE_REAL_POSITIONS in config based on command-line argument
        from config import config
        if args.use_real_positions:
            # Override the config setting with the command-line argument
            config.USE_REAL_POSITIONS = True
            print(f"Using real positions from Alpaca API")
            
        # Log the data source being used
        print(f"Using data source: {data_source}")
            
        # Initialize the bot with the specified strategy and data source
        bot = AlpacaBot(strategy_name=args.strategy, data_source=data_source)
        
        # Start scheduled trading with options
        bot.start_scheduled(wait_for_open=args.wait_for_open)
        
        # Keep the main thread alive
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        import traceback
        print(f"Unhandled exception: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()