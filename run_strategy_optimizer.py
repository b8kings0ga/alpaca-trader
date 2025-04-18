#!/usr/bin/env python
"""
Run the strategy optimizer for the Alpaca Trading Bot.

This script provides a command-line interface to run the strategy optimizer
either immediately or scheduled after market close.
"""
import os
import sys
import time
import typer
from typing import List, Optional
from enum import Enum
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from src.logger import get_logger
from src.strategy_optimizer import optimize_strategies, setup_optimization_scheduler, run_optimization_after_market_close

logger = get_logger()

app = typer.Typer(help="Run strategy optimizer for the Alpaca Trading Bot")

class TargetMetric(str, Enum):
    PROFIT = "profit"
    SHARPE = "sharpe"
    WIN_RATE = "win_rate"
    RETURNS = "strategy_returns"
    CUMULATIVE_RETURNS = "strategy_cumulative_returns"
    MAX_DRAWDOWN = "max_drawdown"

class Strategy(str, Enum):
    MOVING_AVERAGE = "moving_average_crossover"
    RSI = "rsi"
    ML = "ml"
    ALL = "all"

@app.command()
def run(
    strategies: Optional[List[Strategy]] = typer.Option(None, help="Strategies to optimize"),
    target_metric: TargetMetric = typer.Option(TargetMetric.PROFIT, help="Metric to optimize for"),
    test_period: str = typer.Option("1mo", help="Period to use for testing optimization results"),
    immediate: bool = typer.Option(False, help="Run optimization immediately"),
    schedule: bool = typer.Option(True, help="Schedule optimization after market close"),
    interactive: bool = typer.Option(False, help="Run in interactive mode"),
):
    """
    Run the strategy optimizer.
    """
    if interactive:
        # Ask user which strategies to optimize
        strategy_choices = inquirer.checkbox(
            message="Select strategies to optimize:",
            choices=[
                Choice(value=Strategy.MOVING_AVERAGE.value, name="Moving Average Crossover"),
                Choice(value=Strategy.RSI.value, name="RSI Strategy"),
                Choice(value=Strategy.ML.value, name="ML Strategy"),
                Choice(value=Strategy.ALL.value, name="All Strategies"),
            ],
            default=[Strategy.ALL.value]
        ).execute()
        
        # Ask user which metric to optimize for
        target_metric_choice = inquirer.select(
            message="Select metric to optimize for:",
            choices=[
                Choice(value=TargetMetric.PROFIT.value, name="Profit"),
                Choice(value=TargetMetric.SHARPE.value, name="Sharpe Ratio"),
                Choice(value=TargetMetric.WIN_RATE.value, name="Win Rate"),
                Choice(value=TargetMetric.RETURNS.value, name="Strategy Returns"),
                Choice(value=TargetMetric.CUMULATIVE_RETURNS.value, name="Cumulative Returns"),
                Choice(value=TargetMetric.MAX_DRAWDOWN.value, name="Max Drawdown"),
            ],
            default=TargetMetric.PROFIT.value
        ).execute()
        
        # Ask user whether to run immediately or schedule
        run_choice = inquirer.select(
            message="When to run optimization:",
            choices=[
                Choice(value="immediate", name="Run Immediately"),
                Choice(value="schedule", name="Schedule After Market Close"),
                Choice(value="both", name="Both (Run Now and Schedule)"),
            ],
            default="both"
        ).execute()
        
        # Update options based on user choices
        if Strategy.ALL.value in strategy_choices:
            strategies = None  # All strategies
        else:
            strategies = strategy_choices
            
        target_metric = target_metric_choice
        immediate = run_choice in ["immediate", "both"]
        schedule = run_choice in ["schedule", "both"]
    else:
        # Process command-line options
        if strategies and Strategy.ALL in strategies:
            strategies = None  # All strategies
        elif strategies:
            strategies = [s.value for s in strategies]
    
    # Run optimization immediately if requested
    if immediate:
        typer.echo("Running strategy optimization immediately...")
        try:
            results = optimize_strategies(strategies, target_metric, test_period)
            typer.echo(f"Optimization results: {results}")
        except Exception as e:
            typer.echo(f"Error running optimization: {e}")
    
    # Schedule optimization after market close if requested
    if schedule:
        typer.echo("Setting up scheduler to run optimization after market close...")
        try:
            scheduler = setup_optimization_scheduler()
            
            # Keep the script running if only scheduling
            if not immediate:
                typer.echo("Scheduler is running. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    scheduler.stop()
                    typer.echo("Scheduler stopped.")
        except Exception as e:
            typer.echo(f"Error setting up scheduler: {e}")

@app.command()
def optimize_now():
    """
    Run optimization immediately with default settings.
    """
    typer.echo("Running strategy optimization immediately with default settings...")
    try:
        results = optimize_strategies()
        typer.echo(f"Optimization results: {results}")
    except Exception as e:
        typer.echo(f"Error running optimization: {e}")

@app.command()
def schedule_only():
    """
    Schedule optimization after market close without running immediately.
    """
    typer.echo("Setting up scheduler to run optimization after market close...")
    try:
        scheduler = setup_optimization_scheduler()
        
        typer.echo("Scheduler is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop()
            typer.echo("Scheduler stopped.")
    except Exception as e:
        typer.echo(f"Error setting up scheduler: {e}")

if __name__ == "__main__":
    app()