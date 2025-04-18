# Ensemble ML Strategy with Backtesting and Optimization

This document explains how to use the Ensemble ML Strategy with backtesting and optimization capabilities.

## Overview

The Ensemble ML Strategy combines multiple machine learning models to generate more accurate trading signals. It uses a weighted voting approach to combine the predictions from different models, which can lead to more robust and reliable trading decisions.

The strategy includes:
- Support for multiple ML model types (random_forest, gradient_boosting, etc.)
- Weighted voting mechanism to combine model predictions
- Backtesting capabilities to evaluate performance on historical data
- Optimization capabilities to fine-tune model weights

## Available Scripts

### 1. Run Ensemble Optimizer

The `run_ensemble_optimizer.py` script allows you to optimize the weights of the ensemble ML strategy:

```bash
# Run optimization immediately with default settings
python run_ensemble_optimizer.py optimize-now

# Run optimization with specific models
python run_ensemble_optimizer.py optimize-now --models random_forest gradient_boosting

# Run optimization with specific metric
python run_ensemble_optimizer.py optimize-now --metric sharpe_ratio

# Run optimization with specific symbols
python run_ensemble_optimizer.py optimize-now --symbols AAPL MSFT GOOGL

# Schedule optimization to run after market close
python run_ensemble_optimizer.py schedule --time 16:05
```

### 2. Start Trading with Ensemble Strategy

The `start_trading_with_ensemble.py` script allows you to start trading with the optimized ensemble ML strategy:

```bash
# Start trading with default settings
python start_trading_with_ensemble.py start

# Start trading with specific models
python start_trading_with_ensemble.py start --models random_forest gradient_boosting

# Start trading with specific symbols
python start_trading_with_ensemble.py start --symbols AAPL MSFT GOOGL

# Start trading without optimization after market close
python start_trading_with_ensemble.py start --no-optimize

# Start trading with model training before starting
python start_trading_with_ensemble.py start --train

# Start live trading (not paper trading)
python start_trading_with_ensemble.py start --live
```

## How It Works

### Ensemble ML Strategy

The Ensemble ML Strategy works by:

1. Loading pre-trained ML models (random_forest, gradient_boosting, etc.)
2. Generating trading signals from each model
3. Combining the signals using a weighted voting mechanism
4. Determining the final signal based on the weighted vote and confidence threshold

### Optimization

The optimization process works by:

1. Testing different weight combinations for the ensemble models
2. Running backtests with each weight combination
3. Evaluating performance based on the target metric (profit, sharpe_ratio, etc.)
4. Selecting the best weight combination
5. Updating the strategy with the optimized weights

### Backtesting

The backtesting process works by:

1. Running the strategy on historical data
2. Simulating trades based on the generated signals
3. Calculating performance metrics (returns, profit, sharpe_ratio, etc.)
4. Providing a comprehensive evaluation of the strategy's performance

## Configuration

The ensemble ML strategy can be configured in the `config.py` file:

- `ENSEMBLE_WEIGHTS`: Dictionary of weights for each model type
- `ML_CONFIDENCE_THRESHOLD`: Minimum confidence score to execute a trade
- `SYMBOLS`: List of symbols to trade

## Adding New Models

To add a new model type to the ensemble:

1. Implement the model in `src/ml_models.py`
2. Add the model type to the `get_ml_model` function
3. Train the model using `train_ensemble_model` function
4. Add the model type to the ensemble when creating the strategy

## Troubleshooting

If you encounter issues with the ensemble ML strategy:

1. Check if the required models are trained and available in the `models` directory
2. Verify that the model types specified are supported by the `get_ml_model` function
3. Check the logs for any errors during model loading or signal generation
4. Try running the optimization with a single model type to isolate issues
5. Ensure that the historical data is available and properly formatted