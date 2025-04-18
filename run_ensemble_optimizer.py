#!/usr/bin/env python
"""
Run the strategy optimizer for the Ensemble ML Strategy.

This script provides a command-line interface to run the strategy optimizer
for the Ensemble ML Strategy, which can optimize the weights of different
ML models in the ensemble.
"""
import os
import sys
import time
import typer
from datetime import datetime
from typing import List, Optional

from config import config
from src.logger import get_logger
from src.strategy_optimizer import optimize_strategies, run_optimization_after_market_close
from src.ensemble_ml_strategy import EnsembleMLStrategy, train_ensemble_model

logger = get_logger()
app = typer.Typer()

@app.command()
def optimize_now(
    model_types: Optional[List[str]] = typer.Option(
        None, "--models", "-m", help="Model types to include in the ensemble (default: random_forest,gradient_boosting)"
    ),
    target_metric: str = typer.Option(
        "profit", "--metric", "-t", help="Metric to optimize for (profit, sharpe, win_rate, etc.)"
    ),
    test_period: str = typer.Option(
        "1mo", "--period", "-p", help="Period to use for testing optimization results"
    ),
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbols", "-s", help="Symbols to optimize for (default: from config)"
    )
):
    """
    Run the strategy optimizer for the Ensemble ML Strategy immediately.
    """
    # Set default model types if not provided
    if model_types is None:
        model_types = ['random_forest', 'gradient_boosting']
    
    # Set default symbols if not provided
    if symbols is None:
        symbols = config.SYMBOLS
    
    print(f"Running ensemble strategy optimization with models: {model_types}, metric: {target_metric}, period: {test_period}, symbols: {symbols}")
    
    # Train models if they don't exist
    for model_type in model_types:
        model_path = os.path.join("models", f"{model_type}.joblib")
        if not os.path.exists(model_path):
            print(f"Model {model_type} not found. Training...")
            train_ensemble_model([model_type], symbols, period="1y")
    
    # Run optimization
    results = optimize_strategies(['ensemble_ml'], target_metric, test_period)
    
    # Print results
    print("\nOptimization results:")
    if 'ensemble_ml' in results:
        result = results['ensemble_ml']
        if result.get('success', False):
            print(f"Success: Yes")
            print(f"Original weights: {result.get('original_params', {})}")
            print(f"New weights: {result.get('new_params', {})}")
            
            improvement = result.get('performance_improvement', {})
            for metric, value in improvement.items():
                print(f"Improvement in {metric}: {value:.4f}")
            
            print(f"Execution time: {result.get('execution_time', 0):.2f} seconds")
        else:
            print(f"Success: No")
            print(f"Message: {result.get('message', 'Unknown error')}")
    else:
        print("No results for ensemble_ml strategy")

@app.command()
def schedule(
    time_str: str = typer.Option(
        "16:05", "--time", "-t", help="Time to run optimization (HH:MM in market timezone)"
    )
):
    """
    Schedule the strategy optimizer to run at a specific time after market close.
    """
    from src.scheduler import TradingScheduler
    
    # Parse time
    try:
        hour, minute = map(int, time_str.split(':'))
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            raise ValueError("Invalid time format")
    except Exception as e:
        print(f"Error parsing time: {e}")
        print("Time should be in format HH:MM (24-hour)")
        return
    
    print(f"Scheduling ensemble strategy optimization to run at {time_str} (market timezone)")
    
    # Create scheduler
    scheduler = TradingScheduler()
    
    # Schedule optimization
    scheduler.schedule_after_market_close(
        hour, minute, run_optimization_after_market_close
    )
    
    # Run scheduler
    try:
        scheduler.start()
        print("Scheduler started. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping scheduler...")
        scheduler.stop()
        print("Scheduler stopped.")

if __name__ == "__main__":
    app()