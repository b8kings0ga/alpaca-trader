# Strategy Optimization System for Alpaca Trader

This document provides an overview of the strategy optimization system that automatically fine-tunes trading strategy parameters after market close.

## Overview

The strategy optimization system is designed to automatically improve trading strategies by:

1. Fine-tuning strategy parameters based on historical performance
2. Retraining machine learning models with optimized hyperparameters
3. Running after market close to prepare for the next trading day

## Key Components

The system consists of the following components:

1. **Strategy Base Class (`src/strategy_base.py`)**: Enhanced with `optimize()` and `backtest()` methods that all strategy classes inherit.

2. **Strategy Implementations (`src/strategies.py`)**: Each strategy now implements its own `optimize()` method to fine-tune its specific parameters.

3. **Strategy Optimizer (`src/strategy_optimizer.py`)**: Core module that handles the optimization process, including:
   - Fetching historical data
   - Running optimization for each strategy
   - Saving optimization results

4. **Scheduler Enhancement (`src/scheduler.py`)**: Added functionality to schedule jobs after market close.

5. **Command-line Scripts**:
   - `run_strategy_optimizer.py`: Standalone script to run optimization manually or schedule it
   - `start_trading_with_optimization.py`: Integrated script that runs trading during market hours and optimization after market close

## Strategy Optimization Details

### MovingAverageCrossover Strategy

Optimizes:
- Short window period
- Long window period

The optimization process tests different combinations of these parameters to find the optimal settings that maximize the target metric (profit, Sharpe ratio, win rate, etc.).

### RSI Strategy

Optimizes:
- RSI period
- Oversold threshold
- Overbought threshold

The optimization process tests different combinations of these parameters to find the optimal settings that maximize the target metric.

### ML Strategy

Optimizes:
- Model hyperparameters (specific to the model type)
  - For Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf
  - For Gradient Boosting: n_estimators, learning_rate, max_depth, subsample

The optimization process trains models with different hyperparameter combinations and selects the best performing one.

## How to Use

### Running Optimization Manually

To run optimization manually:

```bash
python run_strategy_optimizer.py optimize-now
```

### Scheduling Optimization After Market Close

To schedule optimization to run after market close:

```bash
python run_strategy_optimizer.py schedule-only
```

### Running Trading with Optimization

To start the trading bot during market hours and run optimization after market close:

```bash
python start_trading_with_optimization.py
```

### Interactive Mode

Both scripts support an interactive mode that guides you through the configuration:

```bash
python run_strategy_optimizer.py run --interactive
```

```bash
python start_trading_with_optimization.py start --interactive
```

## Configuration

The system uses the following configuration parameters from `config/config.py`:

- `MARKET_CLOSE_TIME`: Time when the market closes (default: '16:00')
- `TIMEZONE`: Timezone for scheduling (default: 'America/New_York')
- `SYMBOLS`: List of stock symbols to trade and optimize for
- Strategy-specific parameters (initial values that will be optimized)

## Optimization Results

Optimization results are saved to `logs/optimization_results/` with timestamps. Each result file contains:

- Original parameters
- New optimized parameters
- Performance improvement metrics
- Execution time

## Extending the System

To add optimization for a new strategy:

1. Ensure your strategy class inherits from the `Strategy` base class
2. Implement the `optimize()` method in your strategy class
3. Add your strategy to the list of strategies in `src/strategy_optimizer.py`

## Troubleshooting

If optimization is not running as expected:

1. Check the logs for error messages
2. Verify that the market close time is correctly configured
3. Ensure that historical data is available for optimization
4. Check that the strategy parameters are within reasonable ranges

For more detailed information, refer to the source code documentation.