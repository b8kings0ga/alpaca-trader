"""
Run the Alpaca Trading Bot with the Ensemble ML strategy.

This script runs the Alpaca Trading Bot with the Ensemble ML strategy,
which combines multiple ML models to generate more accurate trading signals.
"""
import os
import argparse
from datetime import datetime
import time

from config import config
from src.logger import get_logger
from src.bot import AlpacaBot, setup_logging

logger = get_logger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Alpaca Trading Bot with Ensemble ML strategy')
    parser.add_argument('--strategy', type=str, default='ensemble_ml',
                        help='Trading strategy to use (default: ensemble_ml)')
    parser.add_argument('--data-source', type=str, default='yfinance',
                        help='Data source to use (default: yfinance)')
    parser.add_argument('--wait-for-market', action='store_true',
                        help='Wait for market to open before starting')
    parser.add_argument('--force-initial-positions', action='store_true',
                        help='Force creating initial positions if none exist')
    parser.add_argument('--run-once', action='store_true',
                        help='Run the bot once and exit')
    parser.add_argument('--interval', type=int, default=config.RUN_INTERVAL,
                        help=f'Run interval in minutes (default: {config.RUN_INTERVAL})')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging()
    
    # Log start time
    start_time = datetime.now()
    logger.info(f"Starting Alpaca Trading Bot with Ensemble ML strategy at {start_time}")
    
    # Initialize bot
    bot = AlpacaBot(strategy_name=args.strategy, data_source=args.data_source)
    
    # Update run interval if specified
    if args.interval != config.RUN_INTERVAL:
        logger.info(f"Overriding run interval from {config.RUN_INTERVAL} to {args.interval} minutes")
        config.RUN_INTERVAL = args.interval
    
    try:
        if args.run_once:
            # Run the bot once
            logger.info("Running bot once")
            bot.run(force_initial_positions=args.force_initial_positions)
        else:
            # Start scheduled trading
            logger.info("Starting scheduled trading")
            job_id = bot.start_scheduled(wait_for_open=args.wait_for_market)
            logger.info(f"Scheduled trading started with job ID: {job_id}")
            
            # Keep the script running
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Stopping bot...")
                bot.stop_scheduled()
                logger.info("Bot stopped")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
    finally:
        # Log end time
        end_time = datetime.now()
        logger.info(f"Alpaca Trading Bot finished at {end_time}")
        logger.info(f"Total run time: {end_time - start_time}")

if __name__ == "__main__":
    main()