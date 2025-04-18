"""
Strategy optimizer for the Alpaca Trading Bot.

This module provides functionality to optimize trading strategies
after market close by fine-tuning parameters or retraining models.
"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

from config import config
from src.logger import get_logger
from src.strategies import get_strategy
from src.scheduler import TradingScheduler

logger = get_logger()

def fetch_historical_data(symbols=None, period="1y", interval="1d"):
    """
    Fetch historical data for optimization.
    
    Args:
        symbols (list): List of stock symbols
        period (str): Period to fetch data for (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Interval between data points (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        dict: Dictionary of DataFrames with historical market data
    """
    symbols = symbols or config.SYMBOLS
    logger.info(f"Fetching historical data for {symbols} with period={period}, interval={interval}")
    
    data = {}
    
    for symbol in symbols:
        try:
            # Fetch data from yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                continue
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Rename columns to match Alpaca API format
            df = df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add technical indicators
            # Moving averages
            df['sma_short'] = df['close'].rolling(window=config.SHORT_WINDOW).mean()
            df['sma_long'] = df['close'].rolling(window=config.LONG_WINDOW).mean()
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=config.RSI_PERIOD).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=config.RSI_PERIOD).mean()
            
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Add more technical indicators as needed
            
            data[symbol] = df
            
            logger.info(f"Fetched {len(df)} data points for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    return data

def optimize_strategies(strategies=None, target_metric='profit', test_period='1mo'):
    """
    Optimize all strategies by fine-tuning parameters or retraining models.
    
    Args:
        strategies (list): List of strategy names to optimize
        target_metric (str): Metric to optimize for ('profit', 'sharpe', 'win_rate', etc.)
        test_period (str): Period to use for testing optimization results
        
    Returns:
        dict: Dictionary containing optimization results for each strategy
    """
    # Get list of strategies to optimize
    if strategies is None:
        strategies = ['moving_average_crossover', 'rsi', 'ml']
    
    # Check if 'ml' is in strategies and replace it with 'ensemble_ml'
    if 'ml' in strategies:
        strategies.remove('ml')
        strategies.append('ensemble_ml')
        logger.info("Replaced 'ml' strategy with 'ensemble_ml' strategy for optimization")
    
    logger.info(f"Optimizing strategies: {strategies}")
    
    # Fetch historical data for optimization
    historical_data = fetch_historical_data(period="1y", interval="1d")
    
    if not historical_data:
        logger.error("No historical data available for optimization")
        return {'success': False, 'message': 'No historical data available'}
    
    results = {}
    
    # Optimize each strategy
    for strategy_name in strategies:
        logger.info(f"Optimizing strategy: {strategy_name}")
        
        try:
            # Get strategy instance
            strategy = get_strategy(strategy_name)
            
            if strategy is None:
                logger.warning(f"Strategy not found: {strategy_name}")
                continue
            
            # Optimize strategy
            start_time = time.time()
            optimization_result = strategy.optimize(historical_data, target_metric, test_period)
            end_time = time.time()
            
            # Add execution time to results
            optimization_result['execution_time'] = end_time - start_time
            
            # Store results
            results[strategy_name] = optimization_result
            
            logger.info(f"Optimization completed for {strategy_name} in {end_time - start_time:.2f} seconds")
            logger.info(f"Optimization results: {optimization_result}")
            
        except Exception as e:
            logger.error(f"Error optimizing strategy {strategy_name}: {e}")
            results[strategy_name] = {
                'success': False,
                'message': str(e)
            }
    
    return results

def run_optimization_after_market_close():
    """
    Run optimization after market close.
    This function is called by the scheduler.
    """
    logger.info("Running strategy optimization after market close")
    
    try:
        # Optimize strategies
        results = optimize_strategies()
        
        # Log results
        logger.info(f"Optimization results: {results}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("logs", "optimization_results")
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f"optimization_results_{timestamp}.txt")
        
        with open(results_file, "w") as f:
            f.write(f"Optimization Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for strategy_name, result in results.items():
                f.write(f"Strategy: {strategy_name}\n")
                f.write("-" * 40 + "\n")
                
                if result.get('success', False):
                    f.write(f"Success: Yes\n")
                    f.write(f"Original parameters: {result.get('original_params', {})}\n")
                    f.write(f"New parameters: {result.get('new_params', {})}\n")
                    
                    improvement = result.get('performance_improvement', {})
                    for metric, value in improvement.items():
                        f.write(f"Improvement in {metric}: {value:.4f}\n")
                else:
                    f.write(f"Success: No\n")
                    f.write(f"Message: {result.get('message', 'Unknown error')}\n")
                
                f.write(f"Execution time: {result.get('execution_time', 0):.2f} seconds\n\n")
            
            f.write("=" * 80 + "\n")
        
        logger.info(f"Optimization results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error running optimization after market close: {e}")

def setup_optimization_scheduler():
    """
    Set up the scheduler to run optimization after market close.
    
    Returns:
        TradingScheduler: The scheduler instance
    """
    logger.info("Setting up optimization scheduler")
    
    # Create scheduler
    scheduler = TradingScheduler()
    
    # Add job to run optimization after market close
    job_id = scheduler.add_market_close_job(run_optimization_after_market_close)
    
    logger.info(f"Added optimization job with ID: {job_id}")
    
    # Start scheduler
    scheduler.start()
    
    return scheduler

if __name__ == "__main__":
    # Set up scheduler
    scheduler = setup_optimization_scheduler()
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop scheduler on keyboard interrupt
        scheduler.stop()
        logger.info("Optimization scheduler stopped")