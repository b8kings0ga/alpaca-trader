"""
Optimize trading strategies for the Alpaca Trading Bot.

This script provides a command-line interface to optimize trading strategies,
retrain models, and evaluate performance.
"""
import os
import sys
import subprocess
import typer
from typing import List, Optional
from enum import Enum
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from src.logger import get_logger

logger = get_logger()

app = typer.Typer(help="Optimize trading strategies for the Alpaca Trading Bot")

class Strategy(str, Enum):
    MOVING_AVERAGE = "moving_average_crossover"
    RSI = "rsi"
    ML = "ml"
    DUAL_MA = "dual_ma_yf"
    ENSEMBLE = "ensemble_ml"

class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"

@app.command()
def train_model(
    model_type: ModelType = typer.Option(ModelType.GRADIENT_BOOSTING, help="Type of ML model to train"),
    symbols: Optional[List[str]] = typer.Option(None, help="Stock symbols to train on"),
    period: str = typer.Option("1y", help="Period of historical data to use"),
    force: bool = typer.Option(False, help="Force retraining even if model exists"),
    evaluate: bool = typer.Option(True, help="Evaluate model performance after training"),
    plot: bool = typer.Option(True, help="Plot feature importance and model performance"),
    compare: bool = typer.Option(True, help="Compare with baseline model performance"),
):
    """
    Train an individual ML model with optimized parameters.
    """
    typer.echo(f"Training {model_type.value} model with optimized parameters...")
    
    cmd = ["python", "train_optimized_models.py", "--model-type", model_type.value]
    
    if symbols:
        cmd.extend(["--symbols"] + symbols)
    
    cmd.extend(["--period", period])
    
    if force:
        cmd.append("--force")
    
    if evaluate:
        cmd.append("--evaluate")
    
    if plot:
        cmd.append("--plot")
    
    if compare:
        cmd.append("--compare")
    
    typer.echo(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

@app.command()
def train_ensemble(
    model_types: Optional[List[ModelType]] = typer.Option(None, help="Types of ML models to use in the ensemble"),
    symbols: Optional[List[str]] = typer.Option(None, help="Stock symbols to train on"),
    period: str = typer.Option("1y", help="Period of historical data to use"),
    test_period: str = typer.Option("1mo", help="Period of historical data to use for testing"),
    force: bool = typer.Option(False, help="Force retraining even if models exist"),
    evaluate: bool = typer.Option(True, help="Evaluate model performance after training"),
    plot: bool = typer.Option(True, help="Plot signals and performance"),
    compare: bool = typer.Option(True, help="Compare with baseline ensemble performance"),
):
    """
    Train an ensemble of ML models with optimized parameters.
    """
    typer.echo("Training ensemble models with optimized parameters...")
    
    cmd = ["python", "train_optimized_ensemble.py"]
    
    if model_types:
        cmd.extend(["--model-types"] + [mt.value for mt in model_types])
    
    if symbols:
        cmd.extend(["--symbols"] + symbols)
    
    cmd.extend(["--period", period, "--test-period", test_period])
    
    if force:
        cmd.append("--force")
    
    if evaluate:
        cmd.append("--evaluate")
    
    if plot:
        cmd.append("--plot")
    
    if compare:
        cmd.append("--compare")
    
    typer.echo(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

@app.command()
def run_bot(
    strategy: Strategy = typer.Option(Strategy.ENSEMBLE, help="Trading strategy to use"),
    symbols: Optional[List[str]] = typer.Option(None, help="Stock symbols to trade"),
    paper: bool = typer.Option(True, help="Use paper trading"),
    live: bool = typer.Option(False, help="Use live trading (overrides --paper)"),
    backtest: bool = typer.Option(True, help="Run in backtest mode"),
    period: str = typer.Option("1mo", help="Period for backtest"),
    plot: bool = typer.Option(True, help="Plot backtest results"),
    compare: bool = typer.Option(True, help="Compare with baseline strategy"),
):
    """
    Run the trading bot with optimized strategies.
    """
    typer.echo(f"Running bot with {strategy.value} strategy...")
    
    cmd = ["python", "run_optimized_bot.py", "--strategy", strategy.value]
    
    if symbols:
        cmd.extend(["--symbols"] + symbols)
    
    if paper:
        cmd.append("--paper")
    
    if live:
        cmd.append("--live")
    
    if backtest:
        cmd.append("--backtest")
        cmd.extend(["--period", period])
    
    if plot:
        cmd.append("--plot")
    
    if compare:
        cmd.append("--compare")
    
    typer.echo(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

@app.command()
def optimize_all(
    symbols: Optional[List[str]] = typer.Option(None, help="Stock symbols to use"),
    period: str = typer.Option("1y", help="Period of historical data to use for training"),
    test_period: str = typer.Option("1mo", help="Period of historical data to use for testing"),
    force: bool = typer.Option(False, help="Force retraining even if models exist"),
):
    """
    Run the complete optimization pipeline: train models, train ensemble, and run backtest.
    """
    typer.echo("Starting complete optimization pipeline...")
    
    # Ask user which steps to run
    steps = inquirer.checkbox(
        message="Select optimization steps to run:",
        choices=[
            Choice(value="train_rf", name="Train Random Forest Model"),
            Choice(value="train_gb", name="Train Gradient Boosting Model"),
            Choice(value="train_ensemble", name="Train Ensemble Model"),
            Choice(value="run_backtest", name="Run Backtest"),
        ],
        default=["train_rf", "train_gb", "train_ensemble", "run_backtest"]
    ).execute()
    
    # Ask user which strategies to test
    strategies = inquirer.checkbox(
        message="Select strategies to test:",
        choices=[
            Choice(value=Strategy.MOVING_AVERAGE.value, name="Moving Average Crossover"),
            Choice(value=Strategy.RSI.value, name="RSI Strategy"),
            Choice(value=Strategy.ML.value, name="ML Strategy"),
            Choice(value=Strategy.ENSEMBLE.value, name="Ensemble ML Strategy"),
        ],
        default=[Strategy.ENSEMBLE.value]
    ).execute()
    
    # Train Random Forest model
    if "train_rf" in steps:
        typer.echo("\n=== Training Random Forest Model ===")
        train_model(
            model_type=ModelType.RANDOM_FOREST,
            symbols=symbols,
            period=period,
            force=force,
            evaluate=True,
            plot=True,
            compare=True
        )
    
    # Train Gradient Boosting model
    if "train_gb" in steps:
        typer.echo("\n=== Training Gradient Boosting Model ===")
        train_model(
            model_type=ModelType.GRADIENT_BOOSTING,
            symbols=symbols,
            period=period,
            force=force,
            evaluate=True,
            plot=True,
            compare=True
        )
    
    # Train Ensemble model
    if "train_ensemble" in steps:
        typer.echo("\n=== Training Ensemble Model ===")
        train_ensemble(
            symbols=symbols,
            period=period,
            test_period=test_period,
            force=force,
            evaluate=True,
            plot=True,
            compare=True
        )
    
    # Run backtest for each selected strategy
    if "run_backtest" in steps:
        for strategy in strategies:
            typer.echo(f"\n=== Running Backtest for {strategy} Strategy ===")
            run_bot(
                strategy=Strategy(strategy),
                symbols=symbols,
                backtest=True,
                period=test_period,
                plot=True,
                compare=True
            )
    
    typer.echo("\n=== Optimization Pipeline Complete ===")
    typer.echo("Check the logs and plots for results.")

@app.command()
def show_guide():
    """
    Display the optimized strategy guide.
    """
    try:
        with open("optimized_strategy_guide.md", "r") as f:
            guide = f.read()
        
        typer.echo(guide)
    except Exception as e:
        typer.echo(f"Error reading guide: {e}")

if __name__ == "__main__":
    app()