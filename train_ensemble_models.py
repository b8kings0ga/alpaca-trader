"""
Train Ensemble ML Models for the Alpaca Trading Bot.

This script trains multiple machine learning models for the ensemble strategy,
which combines predictions from different models to generate more accurate trading signals.
"""
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import config
from src.logger import get_logger
from src.ensemble_ml_strategy import train_ensemble_model
from src.feature_engineering import prepare_features_targets

logger = get_logger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Ensemble ML models for stock trading')
    parser.add_argument('--model-types', type=str, nargs='+', 
                        default=['random_forest', 'gradient_boosting'],
                        help='Types of ML models to train for the ensemble')
    parser.add_argument('--symbols', type=str, nargs='+', default=config.SYMBOLS,
                        help='Stock symbols to train on')
    parser.add_argument('--period', type=str, default='1y',
                        help='Period of historical data to use (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--lookback', type=int, default=config.ML_LOOKBACK_PERIOD,
                        help='Number of days to look back for features')
    parser.add_argument('--force', action='store_true',
                        help='Force retraining even if models exist')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model performance after training')
    parser.add_argument('--plot', action='store_true',
                        help='Plot feature importance and model performance')
    
    return parser.parse_args()

def evaluate_ensemble(models, args):
    """
    Evaluate the ensemble model performance.
    
    Args:
        models (dict): Dictionary of trained models
        args (Namespace): Command line arguments
    """
    logger.info("Evaluating ensemble model performance")
    
    # Import here to avoid circular imports
    from src.yfinance_data import YFinanceData
    
    # Get recent data for evaluation
    yf_data = YFinanceData()
    historical_data = yf_data.get_historical_data(args.symbols, period='1mo')
    
    if not historical_data:
        logger.error("Failed to fetch historical data for evaluation")
        return
    
    # Prepare features and targets
    prepared_data = prepare_features_targets(historical_data, lookback_period=args.lookback)
    
    if not prepared_data:
        logger.error("Failed to prepare features and targets for evaluation")
        return
    
    # Evaluate on each symbol
    ensemble_predictions = {}
    actual_values = {}
    
    for symbol, df in prepared_data.items():
        if df.empty:
            continue
        
        # Store actual values
        if 'target' in df.columns:
            actual_values[symbol] = df['target'].values
        else:
            logger.warning(f"No target column found for {symbol}")
            continue
        
        # Get predictions from each model
        symbol_predictions = {}
        
        for model_type, model in models.items():
            # Extract features and target
            X, y = model.preprocess_data(df)
            
            if X is None or y is None:
                continue
            
            # Get predictions
            predictions = model.predict(X)
            symbol_predictions[model_type] = predictions
        
        # Store predictions for this symbol
        ensemble_predictions[symbol] = symbol_predictions
    
    # Calculate ensemble metrics
    for symbol in ensemble_predictions:
        if symbol not in actual_values:
            continue
        
        # Get actual values
        y_true = actual_values[symbol]
        
        # Calculate weighted ensemble predictions
        ensemble_preds = np.zeros_like(y_true, dtype=float)
        
        for model_type, preds in ensemble_predictions[symbol].items():
            weight = config.ENSEMBLE_WEIGHTS.get(model_type, 1.0 / len(models))
            ensemble_preds += preds * weight
        
        # Convert to binary predictions (-1, 0, 1)
        binary_preds = np.zeros_like(y_true)
        binary_preds[ensemble_preds > config.ML_CONFIDENCE_THRESHOLD] = 1
        binary_preds[ensemble_preds < -config.ML_CONFIDENCE_THRESHOLD] = -1
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Filter out hold signals (0) for precision, recall, and f1
        mask = (y_true != 0) & (binary_preds != 0)
        
        if np.sum(mask) > 0:
            accuracy = accuracy_score(y_true, binary_preds)
            
            # For precision, recall, and f1, we only consider buy/sell signals
            precision = precision_score(y_true[mask], binary_preds[mask], average='weighted')
            recall = recall_score(y_true[mask], binary_preds[mask], average='weighted')
            f1 = f1_score(y_true[mask], binary_preds[mask], average='weighted')
            
            logger.info(f"Ensemble evaluation metrics for {symbol}:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
        else:
            logger.warning(f"Not enough non-zero predictions for {symbol} to calculate metrics")

def plot_ensemble_performance(models, args):
    """
    Plot ensemble model performance.
    
    Args:
        models (dict): Dictionary of trained models
        args (Namespace): Command line arguments
    """
    try:
        import matplotlib.pyplot as plt
        
        # Import here to avoid circular imports
        from src.yfinance_data import YFinanceData
        
        # Get recent data for evaluation
        logger.info(f"Fetching historical data for plotting: symbols={args.symbols}, period=1mo")
        yf_data = YFinanceData()
        historical_data = yf_data.get_historical_data(args.symbols, period='1mo')
        
        if not historical_data:
            logger.error("Failed to fetch historical data for plotting")
            return
        
        # Log the historical data shape
        for symbol, df in historical_data.items():
            logger.info(f"Historical data for {symbol}: {df.shape} rows, columns: {df.columns.tolist()}")
        
        # Prepare features and targets
        logger.info(f"Preparing features and targets with lookback period: {args.lookback}")
        try:
            prepared_data = prepare_features_targets(historical_data, lookback_period=args.lookback)
            
            if not prepared_data:
                logger.error("prepare_features_targets returned empty result")
                return
                
            # Log the prepared data shape
            for symbol, df in prepared_data.items():
                logger.info(f"Prepared data for {symbol}: {df.shape} rows, columns: {df.columns.tolist()}")
                
        except Exception as e:
            logger.error(f"Exception in prepare_features_targets: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return
        
        # Plot for each symbol
        for symbol, df in prepared_data.items():
            if df.empty or len(df) < 10:
                continue
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot price
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(df.index, df['close'], label='Close Price')
            ax1.set_title(f'{symbol} - Price and Ensemble Predictions')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot predictions
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            
            # Get predictions from each model
            for model_type, model in models.items():
                # Extract features and target
                X, y = model.preprocess_data(df)
                
                if X is None or y is None:
                    continue
                
                # Get predictions
                predictions = model.predict(X)
                
                # Plot predictions
                ax2.plot(df.index, predictions, label=f'{model_type} Predictions', alpha=0.7)
            
            # Plot actual target if available
            if 'target' in df.columns:
                ax2.plot(df.index, df['target'], label='Actual', color='black', linestyle='--')
            
            ax2.set_ylabel('Signal')
            ax2.set_xlabel('Date')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Save figure
            os.makedirs('logs/plots', exist_ok=True)
            plt.tight_layout()
            plt.savefig(f'logs/plots/ensemble_performance_{symbol}.png')
            plt.close()
            
            logger.info(f"Ensemble performance plot saved to logs/plots/ensemble_performance_{symbol}.png")
            
    except Exception as e:
        logger.error(f"Error plotting ensemble performance: {e}")

def main():
    """Main function."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Parse command line arguments
    args = parse_args()
    
    # Check if models already exist
    all_models_exist = True
    for model_type in args.model_types:
        model_path = f"models/{model_type}.joblib"
        if not os.path.exists(model_path):
            all_models_exist = False
            break
    
    if all_models_exist and not args.force:
        logger.info("All models already exist. Use --force to retrain.")
        
        # Load existing models for evaluation
        from src.ml_models import get_ml_model
        
        loaded_models = {}
        for model_type in args.model_types:
            model_path = f"models/{model_type}.joblib"
            model = get_ml_model(model_type)
            if model and model.load(model_path):
                loaded_models[model_type] = model
                logger.info(f"Loaded existing model from {model_path}")
        
        if args.evaluate and loaded_models:
            evaluate_ensemble(loaded_models, args)
            
            if args.plot:
                plot_ensemble_performance(loaded_models, args)
                
        return
    
    # Train ensemble models
    logger.info(f"Training ensemble models: {args.model_types} on {args.symbols} with {args.period} of data")
    trained_models = train_ensemble_model(args.model_types, args.symbols, args.period)
    
    if not trained_models:
        logger.error("Failed to train ensemble models")
        return
    
    logger.info(f"Successfully trained {len(trained_models)} models for ensemble strategy")
    
    # Evaluate ensemble
    if args.evaluate:
        evaluate_ensemble(trained_models, args)
        
        if args.plot:
            plot_ensemble_performance(trained_models, args)

if __name__ == '__main__':
    main()