# Machine Learning Models for Alpaca Trader

This directory contains trained machine learning models for the Alpaca Trading Bot.

## Available Models

The following models are available:

- `random_forest.joblib`: Random Forest model for predicting stock price movements
- `gradient_boosting.joblib`: Gradient Boosting model (XGBoost/LightGBM) for predicting stock price movements
- `ensemble.joblib`: Ensemble model combining multiple ML models

## Training Models

To train a new model, use the `train_ml_model.py` script:

```bash
python train_ml_model.py --model-type random_forest --symbols AAPL MSFT GOOGL --period 1y
```

### Command Line Arguments

- `--model-type`: Type of ML model to train (random_forest, gradient_boosting, ensemble)
- `--symbols`: Stock symbols to train on (default: symbols from config.py)
- `--period`: Period of historical data to use (default: 1y)
- `--lookback`: Number of days to look back for features (default: from config.py)
- `--force`: Force retraining even if model exists
- `--evaluate`: Evaluate model performance after training
- `--plot`: Plot feature importance and model performance

## Using Models

The models are automatically used by the `MLStrategy` class in `src/strategies.py`. To use a specific model, update the `ML_STRATEGY_TYPE` parameter in `config/config.py`.

## Model Performance

Model performance metrics are logged during training and evaluation. You can view these metrics in the log files.

Feature importance plots are saved to `logs/plots/` when using the `--plot` option.

## Model Features

The models use the following features:

- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Price data (open, high, low, close)
- Volume data
- Moving averages
- Volatility measures

## Customizing Models

To customize the models, you can modify the following files:

- `src/ml_models.py`: Implementation of ML models
- `src/feature_engineering.py`: Feature engineering for ML models
- `config/config.py`: Configuration parameters for ML models