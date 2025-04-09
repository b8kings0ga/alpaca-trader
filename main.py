#!/usr/bin/env python3
"""
Entry point for the Alpaca Trading Bot.
"""
import argparse
from src.bot import AlpacaBot

def main():
    """
    Main entry point for the trading bot.
    """
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Alpaca Trading Bot')
        parser.add_argument('--wait-for-open', action='store_true', help='Wait for market to open before starting')
        parser.add_argument('--use-yfinance', action='store_true', help='Use yfinance for data instead of Alpaca API')
        parser.add_argument('--strategy', type=str, default='moving_average_crossover',
                            choices=['moving_average_crossover', 'rsi', 'ml', 'dual_ma_yf'],
                            help='Trading strategy to use')
        args = parser.parse_args()
        
        # Initialize the bot with the specified strategy
        bot = AlpacaBot(strategy_name=args.strategy)
        
        # Start scheduled trading with options
        bot.start_scheduled(wait_for_open=args.wait_for_open, use_yfinance=args.use_yfinance)
        
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