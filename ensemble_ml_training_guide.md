# Training Guide for Ensemble ML Strategy

This guide explains how to train the machine learning models used in the Ensemble ML Strategy for the Alpaca Trader bot.

## Overview

The Ensemble ML Strategy combines predictions from multiple machine learning models to generate more accurate trading signals. By training different types of models and combining their predictions, we can achieve better performance than using a single model.

## Prerequisites

Before training the models, make sure you have:

1. Set up the Alpaca Trader environment
2. Installed all required dependencies (see requirements.txt)
3. Access to historical market data (the system will fetch this automatically)

## Training the Models

### Using the Training Script

The easiest way to train all models for the ensemble strategy is to use the `train_ensemble_models.py` script:

```bash
python train_ensemble_models.py
```

This will train all the default models (Random Forest and Gradient Boosting) using the default symbols from your config file and 1 year of historical data.

### Training Options

You can customize the training process with various command-line options:

```bash
python train_ensemble_models.py --model-types random_forest gradient_boosting --symbols AAPL MSFT GOOGL --period 2y --force --evaluate --plot
```

Available options:

- `--model-types`: List of model types to train (default: random_forest gradient_boosting)
- `--symbols`: Stock symbols to train on (default: symbols from config.py)
- `--period`: Period of historical data to use (default: 1y)
- `--lookback`: Number of days to look back for features (default: from config.py)
- `--force`: Force retraining even if models already exist
- `--evaluate`: Evaluate model performance after training
- `--plot`: Generate performance plots

### Training Individual Models

If you prefer to train models individually, you can use the `train_ml_model.py` script:

```bash
# Train Random Forest model
python train_ml_model.py --model-type random_forest --symbols AAPL MSFT GOOGL --period 1y

# Train Gradient Boosting model
python train_ml_model.py --model-type gradient_boosting --symbols AAPL MSFT GOOGL --period 1y
```

## Evaluating Model Performance

To evaluate the performance of your trained models:

```bash
python train_ensemble_models.py --evaluate --plot
```

This will:
1. Load the existing trained models
2. Evaluate their performance on recent data
3. Generate performance metrics (accuracy, precision, recall, F1 score)
4. Create performance plots in the logs/plots directory

## Model Storage

Trained models are saved in the `models/` directory:
- `models/random_forest.joblib`: Random Forest model
- `models/gradient_boosting.joblib`: Gradient Boosting model

## Adding New Model Types

To add a new model type to the ensemble:

1. Implement the model in `src/ml_models.py`
2. Update the `get_ml_model()` function to return your new model type
3. Add the model type to the `ENSEMBLE_WEIGHTS` in `config/config.py`
4. Train the new model using the training script

## Recommended Training Schedule

For optimal performance, consider retraining your models:
- Every 1-3 months to capture changing market conditions
- After significant market events
- When adding new symbols to your trading portfolio

## Troubleshooting

If you encounter issues during training:

1. Check the log files in the `logs/` directory
2. Ensure you have sufficient historical data for the requested period
3. Verify that all dependencies are correctly installed
4. Make sure you have enough disk space for model storage

## Next Steps

After training your models:

1. Test the ensemble strategy using `test_ensemble_strategy.py`
2. Run the trading bot with your trained models using `run_ensemble_ml_bot.py`
3. Monitor performance and retrain as needed