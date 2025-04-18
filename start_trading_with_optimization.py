#!/usr/bin/env python
"""
Start the Alpaca Trading Bot with strategy optimization.

This script starts the trading bot to run during market hours and
schedules strategy optimization to run after market close.
"""
import os
import sys
import time
import typer
from typing import List, Optional
from enum import Enum
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from config import config
from src.logger import get_logger
from src.scheduler import TradingScheduler
from src.strategy_optimizer import run_optimization_after_market_close
from src.bot import TradingBot

logger = get_logger()

app = typer.Typer(help="Start the Alpaca Trading Bot with strategy optimization")

class Strategy(str, Enum):
    MOVING_AVERAGE = "moving_average_crossover"
    RSI = "rsi"
    ML = "ml"
    DUAL_MA = "dual_ma_yf"
    ENSEMBLE = "ensemble_ml"

@app.command()
def start(
    strategy: Strategy = typer.Option(Strategy.ENSEMBLE, help="Trading strategy to use"),
    symbols: Optional[List[str]] = typer.Option(None, help="Stock symbols to trade"),
    paper: bool = typer.Option(True, help="Use paper trading"),
    live: bool = typer.Option(False, help="Use live trading (overrides --paper)"),
    optimize: bool = typer.Option(True, help="Enable strategy optimization after market close"),
    interactive: bool = typer.Option(False, help="Run in interactive mode"),
):
    """
    Start the trading bot with strategy optimization.
    """
    if interactive:
        # Ask user which strategy to use
        strategy_choice = inquirer.select(
            message="Select trading strategy:",
            choices=[
                Choice(value=Strategy.MOVING_AVERAGE.value, name="Moving Average Crossover"),
                Choice(value=Strategy.RSI.value, name="RSI Strategy"),
                Choice(value=Strategy.ML.value, name="ML Strategy"),
                Choice(value=Strategy.DUAL_MA.value, name="Dual Moving Average YF"),
                Choice(value=Strategy.ENSEMBLE.value, name="Ensemble ML Strategy"),
            ],
            default=Strategy.ENSEMBLE.value
        ).execute()
        
        # Ask user which symbols to trade
        default_symbols = ", ".join(config.SYMBOLS)
        symbols_input = inquirer.text(
            message="Enter stock symbols to trade (comma-separated):",
            default=default_symbols
        ).execute()
        
        symbols = [s.strip() for s in symbols_input.split(",")]
        
        # Ask user whether to use paper or live trading
        trading_mode = inquirer.select(
            message="Select trading mode:",
            choices=[
                Choice(value="paper", name="Paper Trading"),
                Choice(value="live", name="Live Trading"),
            ],
            default="paper"
        ).execute()
        
        paper = trading_mode == "paper"
        live = trading_mode == "live"
        
        # Ask user whether to enable optimization
        optimize = inquirer.confirm(
            message="Enable strategy optimization after market close?",
            default=True
        ).execute()
        
        # Update options based on user choices
        strategy = strategy_choice
    else:
        # Process command-line options
        if symbols is None:
            symbols = config.SYMBOLS
    
    # Create scheduler
    scheduler = TradingScheduler()
    
    # Create trading bot
    bot = TradingBot(
        strategy=strategy,
        symbols=symbols,
        paper=paper,
        live=live
    )
    
    # Schedule trading during market hours
    market_open_job_id = scheduler.add_market_hours_job(bot.run)
    logger.info(f"Scheduled trading bot to run at market open with job ID: {market_open_job_id}")
    
    # Schedule optimization after market close if enabled
    if optimize:
        market_close_job_id = scheduler.add_market_close_job(run_optimization_after_market_close)
        logger.info(f"Scheduled strategy optimization to run at market close with job ID: {market_close_job_id}")
    
    # Start scheduler
    scheduler.start()
    
    # Log configuration
    logger.info(f"Trading bot started with strategy: {strategy}")
    logger.info(f"Trading symbols: {symbols}")
    logger.info(f"Trading mode: {'Live' if live else 'Paper'}")
    logger.info(f"Strategy optimization: {'Enabled' if optimize else 'Disabled'}")
    
    # Keep the script running
    typer.echo("Trading bot is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop scheduler on keyboard interrupt
        scheduler.stop()
        logger.info("Trading bot stopped")
        typer.echo("Trading bot stopped.")

@app.command()
def run_optimization_only():
    """
    Run strategy optimization immediately without starting the trading bot.
    """
    typer.echo("Running strategy optimization immediately...")
    try:
        run_optimization_after_market_close()
        typer.echo("Strategy optimization completed.")
    except Exception as e:
        typer.echo(f"Error running optimization: {e}")

if __name__ == "__main__":
    app()