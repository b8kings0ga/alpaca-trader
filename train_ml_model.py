"""
Train ML models for the Alpaca Trading Bot.

This script trains machine learning models for predicting stock price movements.
"""
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import config
from src.logger import get_logger
from src.yfinance_data import YFinanceData
from src.feature_engineering import prepare_features_targets
from src.ml_models import (
    get_ml_model, 
    GradientBoostingModel, 
    RandomForestModel
)

logger = get_logger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ML models for stock trading')
    parser.add_argument('--model-type', type=str, default=config.ML_STRATEGY_TYPE,
                        choices=['random_forest', 'gradient_boosting', 'ensemble'],
                        help='Type of ML model to train')
    parser.add_argument('--symbols', type=str, nargs='+', default=config.SYMBOLS,
                        help='Stock symbols to train on')
    parser.add_argument('--period', type=str, default='1y',
                        help='Period of historical data to use (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--lookback', type=int, default=config.ML_LOOKBACK_PERIOD,
                        help='Number of days to look back for features')
    parser.add_argument('--force', action='store_true',
                        help='Force retraining even if model exists')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model performance after training')
    parser.add_argument('--plot', action='store_true',
                        help='Plot feature importance and model performance')
    
    return parser.parse_args()

def train_ml_model(model_type):
    """Train an ML model for trading."""
    # Get model instance
    model = get_ml_model(model_type)
    
    if model is None:
        logger.error(f"Failed to create model of type '{model_type}'")
        return None
    
    # Get historical data
    yf_data = YFinanceData()
    historical_data = yf_data.get_historical_data(config.SYMBOLS, period='1y')
    
    # Prepare data
    all_X = []
    all_y = []
    
    for symbol, df in historical_data.items():
        X, y = model.preprocess_data(df)
        
        if X is not None and y is not None:
            all_X.append(X)
            all_y.append(y)
            
    if not all_X or not all_y:
        logger.error("No valid data for training")
        return None
        
    # Combine data
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    # Train model
    success = model.train(X, y)
    
    if not success:
        logger.error("Failed to train model")
        return None
        
    # Save model
    model_path = f"models/{model_type}.joblib"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model

def train_models(args):
    """Train ML models based on command line arguments."""
    logger.info(f"Training {args.model_type} model on {args.symbols} with {args.period} of historical data")
    
    # Check if model already exists
    model_path = f"models/{args.model_type}.joblib"
    if os.path.exists(model_path) and not args.force:
        logger.info(f"Model already exists at {model_path}. Use --force to retrain.")
        
        # Load the model to evaluate
        model = get_ml_model(args.model_type)
        if model and model.load(model_path):
            logger.info(f"Loaded existing model from {model_path}")
            
            if args.evaluate:
                evaluate_model(model, args)
                
            return model
        else:
            logger.warning(f"Failed to load model from {model_path}. Will retrain.")
    
    # Get historical data
    logger.info(f"Fetching historical data for {args.symbols} over {args.period}")
    yf_data = YFinanceData()
    historical_data = yf_data.get_historical_data(args.symbols, period=args.period)
    
    if not historical_data:
        logger.error("Failed to fetch historical data")
        return None
    
    # Prepare features and targets
    logger.info(f"Preparing features and targets with lookback period of {args.lookback}")
    prepared_data = prepare_features_targets(historical_data, lookback_period=args.lookback)
    
    if not prepared_data:
        logger.error("Failed to prepare features and targets")
        return None
    
    # Train model
    logger.info(f"Training {args.model_type} model")
    model = train_ml_model(args.model_type)
    
    if not model:
        logger.error("Failed to train model")
        return None
    
    logger.info(f"Model trained successfully and saved to {model_path}")
    
    # Evaluate model
    if args.evaluate:
        evaluate_model(model, args)
    
    return model

def evaluate_model(model, args):
    """Evaluate model performance."""
    logger.info("Evaluating model performance")
    
    # Get recent data for evaluation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Use last 30 days for evaluation
    
    # Get historical data
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
    for symbol, df in prepared_data.items():
        if df.empty:
            continue
            
        # Extract features and target
        X, y = model.preprocess_data(df)
        
        if X is None or y is None:
            continue
            
        # Evaluate model
        metrics = model.evaluate(X, y)
        
        logger.info(f"Evaluation metrics for {symbol}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Plot feature importance
    if args.plot:
        plot_feature_importance(model)

def plot_feature_importance(model):
    """Plot feature importance."""
    try:
        import matplotlib.pyplot as plt
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        if not feature_importance:
            logger.warning("No feature importance available")
            return
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features]
        importance = [f[1] for f in sorted_features]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.barh(features, importance)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance - {model.name}')
        plt.tight_layout()
        
        # Save figure
        os.makedirs('logs/plots', exist_ok=True)
        plt.savefig(f'logs/plots/feature_importance_{model.name}.png')
        plt.close()
        
        logger.info(f"Feature importance plot saved to logs/plots/feature_importance_{model.name}.png")
        
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")

def main():
    """Main function."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Parse command line arguments
    args = parse_args()
    
    # Train models
    model = train_models(args)
    
    if model:
        logger.info("Model training completed successfully")
    else:
        logger.error("Model training failed")

if __name__ == '__main__':
    main()