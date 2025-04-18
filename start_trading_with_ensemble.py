#!/usr/bin/env python
"""
Start trading with the optimized Ensemble ML Strategy.

This script provides a command-line interface to start trading with the
Ensemble ML Strategy, which combines multiple ML models to generate more
accurate trading signals.
"""
import os
import sys
import time
import typer
from datetime import datetime
from typing import List, Optional

from config import config
from src.logger import get_logger
from src.bot import TradingBot
from src.ensemble_ml_strategy import EnsembleMLStrategy, train_ensemble_model
from src.scheduler import TradingScheduler

logger = get_logger()
app = typer.Typer()

@app.command()
def start(
    model_types: Optional[List[str]] = typer.Option(
        None, "--models", "-m", help="Model types to include in the ensemble (default: random_forest,gradient_boosting)"
    ),
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbols", "-s", help="Symbols to trade (default: from config)"
    ),
    paper: bool = typer.Option(
        True, "--paper/--live", help="Use paper trading (default) or live trading"
    ),
    optimize_at_close: bool = typer.Option(
        True, "--optimize/--no-optimize", help="Optimize strategy after market close (default: True)"
    ),
    train_models: bool = typer.Option(
        False, "--train/--no-train", help="Train models before starting (default: False)"
    )
):
    """
    Start trading with the optimized Ensemble ML Strategy.
    """
    # Set default model types if not provided
    if model_types is None:
        model_types = ['random_forest', 'gradient_boosting']
    
    # Set default symbols if not provided
    if symbols is None:
        symbols = config.SYMBOLS
    
    print(f"Starting trading with Ensemble ML Strategy")
    print(f"Models: {model_types}")
    print(f"Symbols: {symbols}")
    print(f"Mode: {'Paper' if paper else 'Live'}")
    print(f"Optimize at close: {optimize_at_close}")
    
    # Train models if requested or if they don't exist
    if train_models:
        print("Training models...")
        train_ensemble_model(model_types, symbols, period="1y")
    else:
        # Check if models exist
        for model_type in model_types:
            model_path = os.path.join("models", f"{model_type}.joblib")
            if not os.path.exists(model_path):
                print(f"Model {model_type} not found. Training...")
                train_ensemble_model([model_type], symbols, period="1y")
    
    # Create strategy
    strategy = EnsembleMLStrategy(model_types=model_types)
    
    # Create trading bot
    bot = TradingBot(
        strategy=strategy,
        symbols=symbols,
        paper=paper
    )
    
    # Create scheduler
    scheduler = TradingScheduler()
    
    # Schedule trading during market hours
    scheduler.schedule_during_market_hours(bot.run)
    
    # Schedule optimization after market close if requested
    if optimize_at_close:
        from run_ensemble_optimizer import optimize_now
        
        # Schedule optimization at 16:05 (market timezone)
        scheduler.schedule_after_market_close(
            16, 5, lambda: optimize_now(model_types=model_types, symbols=symbols)
        )
    
    # Start scheduler
    try:
        scheduler.start()
        print("Trading bot started. Press Ctrl+C to exit.")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping trading bot...")
        scheduler.stop()
        print("Trading bot stopped.")

if __name__ == "__main__":
    app()