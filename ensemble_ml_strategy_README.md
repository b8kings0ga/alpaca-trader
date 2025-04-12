# Ensemble ML Strategy for Alpaca Trader

This document provides an overview of the Ensemble Machine Learning (ML) strategy implemented for the Alpaca Trader bot.

## Overview

The Ensemble ML Strategy combines multiple machine learning models to generate more robust and accurate trading signals. By leveraging the strengths of different ML algorithms, the ensemble approach can reduce prediction errors and improve overall performance.

## How It Works

1. **Multiple Model Integration**: The strategy combines predictions from different ML models, including:
   - Random Forest
   - Gradient Boosting
   - (Optionally) Other models like Linear Regression

2. **Weighted Voting**: Each model's prediction is weighted based on:
   - Historical performance
   - Prediction confidence
   - Model reliability

3. **Signal Generation**: The weighted predictions are combined to generate a final trading signal:
   - Buy signal when the weighted prediction is strongly positive
   - Sell signal when the weighted prediction is strongly negative
   - Hold signal when the prediction is uncertain

4. **Confidence Scoring**: Each signal includes a confidence score (0.0-1.0) indicating the strategy's certainty in the prediction.

## Features

- **Robust Predictions**: Less susceptible to individual model errors or overfitting
- **Adaptable Weights**: Model weights can be adjusted based on performance
- **Confidence Metrics**: Provides confidence scores for each trading signal
- **Technical Indicator Integration**: Incorporates various technical indicators as features

## Implementation

The implementation consists of several key components:

1. **EnsembleMLStrategy Class**: Main strategy class that orchestrates the ensemble prediction process
2. **Model Loading**: Loads pre-trained ML models from the models directory
3. **Signal Generation**: Combines model predictions to generate trading signals
4. **Visualization**: Generates plots showing signals and price movements

## Usage

### Training Models

Before using the ensemble strategy, you need to train the individual models:

```bash
# Train Random Forest model
python train_ml_model.py --model-type random_forest --symbols AAPL MSFT GOOGL --period 1y

# Train Gradient Boosting model
python train_ml_model.py --model-type gradient_boosting --symbols AAPL MSFT GOOGL --period 1y
```

### Testing the Strategy

You can test the ensemble strategy on historical data:

```bash
python test_ensemble_strategy.py --symbols AAPL MSFT GOOGL --period 1mo --plot
```

### Running the Trading Bot

To run the trading bot with the ensemble ML strategy:

```bash
python run_ensemble_ml_bot.py --data-source yfinance --wait-for-market
```

## Configuration

The ensemble strategy can be configured in `config/config.py`:

```python
# Ensemble parameters
ENSEMBLE_WEIGHTS = {
    'gradient_boosting': 0.6,
    'random_forest': 0.3,
    'linear': 0.1
}
ENSEMBLE_VOTING = 'soft'  # Options: 'hard', 'soft'
ML_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence score to execute a trade
```

## Performance

The ensemble strategy typically outperforms individual models by:

1. Reducing variance in predictions
2. Mitigating the impact of outliers
3. Capturing different market patterns through diverse models
4. Providing more consistent performance across different market conditions

## Future Improvements

Potential enhancements to the ensemble strategy:

1. **Dynamic Weighting**: Automatically adjust model weights based on recent performance
2. **Additional Models**: Incorporate more model types (LSTM, CNN, etc.)
3. **Feature Expansion**: Add more technical indicators and market sentiment data
4. **Hyperparameter Optimization**: Automatically tune model parameters
5. **Reinforcement Learning**: Add RL components for adaptive trading strategies

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- alpaca-trade-api
- yfinance

## Files

- `src/ensemble_ml_strategy.py`: Main implementation of the ensemble strategy
- `test_ensemble_strategy.py`: Script for testing the strategy
- `run_ensemble_ml_bot.py`: Script for running the trading bot with the ensemble strategy
- `models/`: Directory containing trained ML models