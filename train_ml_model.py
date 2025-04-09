#!/usr/bin/env python3
"""
Script to train and save ML models for the Alpaca Trading Bot.

This is a placeholder script that demonstrates how to train and save ML models
for future use with the Alpaca Trading Bot.
"""
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

from config import config
from src.logger import get_logger
from src.data import MarketData
from src.ml_models import get_ml_model, prepare_training_data

logger = get_logger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ML models for Alpaca Trading Bot')
    parser.add_argument('--model-type', type=str, default='ensemble',
                        choices=['supervised', 'reinforcement', 'nlp', 'ensemble'],
                        help='Type of ML model to train')
    parser.add_argument('--symbols', type=str, default=','.join(config.SYMBOLS),
                        help='Comma-separated list of symbols to train on')
    parser.add_argument('--days', type=int, default=365,
                        help='Number of days of historical data to use')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save trained models')
    return parser.parse_args()

def fetch_historical_data(symbols, days):
    """
    Fetch historical data for training.
    
    Args:
        symbols (list): List of symbols to fetch data for
        days (int): Number of days of historical data to fetch
        
    Returns:
        dict: Dictionary of DataFrames with historical data
    """
    logger.info(f"Fetching {days} days of historical data for {symbols}")
    
    # Initialize MarketData
    market_data = MarketData()
    
    # Fetch historical data
    data = market_data.get_bars(symbols, limit=days)
    
    if not data:
        logger.error("Failed to fetch historical data")
        return {}
        
    logger.info(f"Fetched data for {len(data)} symbols")
    return data

def train_and_save_model(model_type, data, output_dir):
    """
    Train and save an ML model.
    
    Args:
        model_type (str): Type of ML model to train
        data (dict): Dictionary of DataFrames with historical data
        output_dir (str): Directory to save trained models
        
    Returns:
        bool: True if training and saving was successful, False otherwise
    """
    logger.info(f"Training {model_type} model")
    
    # Prepare training data
    X_train, X_test, y_train, y_test = prepare_training_data(data)
    
    if X_train is None or y_train is None:
        logger.error("Failed to prepare training data")
        return False
        
    # Get ML model
    model = get_ml_model(model_type)
    
    if model is None:
        logger.error(f"Failed to get {model_type} model")
        return False
        
    # Train model
    try:
        logger.info("Training model...")
        success = model.train(X_train, y_train)
        
        if not success:
            logger.error("Failed to train model")
            return False
            
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        logger.info(f"Model evaluation metrics: {metrics}")
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_type}_model.pkl")
        
        logger.info(f"Saving model to {model_path}")
        success = model.save(model_path)
        
        if not success:
            logger.error(f"Failed to save model to {model_path}")
            return False
            
        logger.info(f"Successfully trained and saved {model_type} model")
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False

def main():
    """Main function."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_args()
    
    # Parse symbols
    symbols = args.symbols.split(',')
    
    # Fetch historical data
    data = fetch_historical_data(symbols, args.days)
    
    if not data:
        logger.error("No data to train on")
        return
        
    # Train and save model
    success = train_and_save_model(args.model_type, data, args.output_dir)
    
    if success:
        logger.info("Model training completed successfully")
    else:
        logger.error("Model training failed")

if __name__ == "__main__":
    main()