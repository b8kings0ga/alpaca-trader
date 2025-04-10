#!/usr/bin/env python3
"""
Test script to verify account information retrieval.
"""
import argparse
from src.bot import AlpacaBot
from src.logger import get_logger

logger = get_logger()

def main():
    """
    Test account information retrieval.
    """
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Test Account Info')
        parser.add_argument('--data-source', type=str, default='yfinance',
                            choices=['yfinance', 'alpaca', 'yfinance-db'],
                            help='Data source to use (yfinance, alpaca, or yfinance-db)')
        parser.add_argument('--use-real-positions', action='store_true',
                            help='Use real positions from Alpaca API instead of simulated positions')
        args = parser.parse_args()
        
        # Set USE_REAL_POSITIONS in config based on command-line argument
        from config import config
        if args.use_real_positions:
            # Override the config setting with the command-line argument
            config.USE_REAL_POSITIONS = True
            print(f"Using real positions from Alpaca API")
            
        # Log the data source being used
        print(f"Using data source: {args.data_source}")
            
        # Initialize the bot with the specified data source
        bot = AlpacaBot(data_source=args.data_source)
        
        # Get account information
        account_info = bot.market_data.get_account()
        print(f"Account info type: {type(account_info)}")
        print(f"Account info: {account_info}")
        
        # Try to access the equity using the .get() method
        equity = account_info.get('equity', 0)
        print(f"Account equity: ${float(equity):.2f}")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()