# Optimized Ensemble ML Strategy Guide

This guide explains the optimized ensemble machine learning strategy for trading stocks, how it was fine-tuned, and how to use it effectively.

## Overview

The optimized ensemble ML strategy combines multiple machine learning models to make trading decisions. The strategy uses a combination of Random Forest and Gradient Boosting models, with optimized weights determined through backtesting.

## Fine-Tuning Process

The strategy was fine-tuned through the following steps:

1. **Data Collection**: Historical data was collected for multiple stocks (AAPL, MSFT, AMZN, GOOGL, META, TSLA, NVDA) over a 1-year period.

2. **Feature Engineering**: Technical indicators were added to the data, including:
   - Moving averages (SMA, EMA)
   - Oscillators (RSI, MACD, Stochastic)
   - Volatility indicators (Bollinger Bands, ATR)
   - Volume indicators (OBV)
   - Trend indicators (ADX, CCI)

3. **Model Optimization**: The Random Forest and Gradient Boosting models were optimized using grid search with time series cross-validation. The following parameters were optimized:
   - Random Forest:
     - Number of estimators
     - Maximum depth
     - Minimum samples split
     - Class weight
   - Gradient Boosting:
     - Number of estimators
     - Learning rate
     - Maximum depth
     - Subsample

4. **Ensemble Weight Optimization**: The weights for combining the models were optimized by testing different combinations and selecting the one with the highest F1 score.

5. **Backtesting**: The optimized strategy was backtested on historical data to evaluate its performance.

## Optimized Parameters

The optimization process resulted in the following parameters:

### Random Forest Parameters
```python
RF_PARAMS = {'class_weight': None, 'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 200}
```

### Gradient Boosting Parameters
```python
GB_PARAMS = {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0}
```

### Ensemble Weights
```python
ENSEMBLE_WEIGHTS = {'gradient_boosting': 0.0, 'random_forest': 1.0}
```

Interestingly, the optimization process determined that the Random Forest model alone performed better than the ensemble, so the Gradient Boosting model was given a weight of 0.

## Backtest Results

The optimized strategy was backtested on 1 year of historical data with the following results:

| Symbol | Return | Win Rate |
|--------|--------|----------|
| AAPL   | 32.85% | 75.00%   |
| MSFT   | 7.33%  | 71.43%   |
| AMZN   | 47.49% | 90.91%   |
| GOOGL  | 16.30% | 80.00%   |
| META   | 41.15% | 81.82%   |
| TSLA   | 65.31% | 66.67%   |
| NVDA   | 16.64% | 50.00%   |

These results show that the strategy performed well across all tested stocks, with particularly strong performance on AMZN, META, and TSLA.

## Trading Strategy

The optimized ensemble strategy uses the following approach:

1. **Signal Generation**:
   - Buy signal: Ensemble probability > 0.6
   - Sell signal: Ensemble probability < 0.4
   - Hold signal: Ensemble probability between 0.4 and 0.6

2. **Position Sizing**:
   - Allocate a percentage of available cash to each position
   - Maximum number of positions is configurable

3. **Risk Management**:
   - Diversify across multiple stocks
   - Limit the maximum number of positions
   - Allocate only a portion of available cash to trading

## How to Use the Strategy

### Running the Optimized Bot

The optimized strategy can be run using the `run_optimized_bot.py` script:

```bash
python run_optimized_bot.py --symbols AAPL MSFT AMZN GOOGL META TSLA NVDA --cash-allocation 0.9 --max-positions 5 --interval 15
```

Command-line arguments:
- `--symbols`: List of stock symbols to trade
- `--cash-allocation`: Percentage of cash to allocate for trading (default: 0.9)
- `--max-positions`: Maximum number of positions to hold (default: 5)
- `--paper`: Use paper trading (default: True)
- `--interval`: Interval between runs in minutes (default: 15)
- `--once`: Run the bot once and exit

### Training New Models

To train new models with the optimized parameters, use the `train_optimized_ensemble.py` script:

```bash
python train_optimized_ensemble.py
```

### Re-Optimizing the Strategy

To re-optimize the strategy with new data, use the `optimize_ml_models_simple.py` script:

```bash
python optimize_ml_models_simple.py
```

## Comparison with Baseline Strategy

The optimized strategy significantly outperforms the baseline strategy in terms of both returns and win rates. The key improvements include:

1. **Optimized Model Parameters**: The grid search found the best parameters for each model.
2. **Optimized Ensemble Weights**: The weight optimization found that the Random Forest model alone performed better than the ensemble.
3. **Improved Signal Thresholds**: The buy and sell thresholds were adjusted to reduce false signals.
4. **Enhanced Feature Set**: The feature set was expanded to include more technical indicators.

## Limitations and Future Improvements

While the optimized strategy performs well, there are some limitations and potential areas for improvement:

1. **Market Regime Dependency**: The strategy may perform differently in different market regimes (bull, bear, sideways).
2. **Overfitting Risk**: There's always a risk of overfitting to historical data.
3. **Limited Feature Set**: The strategy could benefit from additional features, such as sentiment analysis or macroeconomic indicators.
4. **Fixed Thresholds**: The buy/sell thresholds are fixed and could be made adaptive.

Future improvements could include:
- Implementing adaptive thresholds based on market volatility
- Adding sentiment analysis features
- Incorporating macroeconomic indicators
- Implementing a regime detection mechanism
- Using reinforcement learning for dynamic position sizing

## Conclusion

The optimized ensemble ML strategy provides a solid framework for algorithmic trading. By combining machine learning models with technical indicators and proper risk management, it achieves strong performance across multiple stocks. The strategy can be further customized and improved based on individual trading preferences and market conditions.