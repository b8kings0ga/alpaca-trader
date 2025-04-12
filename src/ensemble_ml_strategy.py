"""
Ensemble ML Strategy for the Alpaca Trading Bot.

This module implements an ensemble machine learning strategy that combines
multiple ML models to generate more accurate trading signals.
"""
import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from config import config
from src.logger import get_logger
from src.strategy_base import Strategy
from src.ml_models import get_ml_model, generate_ml_signals
from src.feature_engineering import add_technical_indicators, create_target_variable

logger = get_logger()

class EnsembleMLStrategy(Strategy):
    """
    Ensemble Machine Learning strategy.
    
    This strategy combines predictions from multiple ML models to generate
    more robust trading signals. It uses a weighted voting approach to
    combine the predictions from different models.
    """
    def __init__(self, model_types=None, weights=None):
        """
        Initialize the Ensemble ML strategy.
        
        Args:
            model_types (list): List of model types to use in the ensemble
            weights (dict): Dictionary of weights for each model type
        """
        super().__init__("Ensemble ML Strategy")
        
        # Set default model types if not provided
        self.model_types = model_types or ['random_forest', 'gradient_boosting']
        # Set default weights if not provided
        self.weights = weights or config.ENSEMBLE_WEIGHTS
        
        # Log the original weights
        logger.info(f"Original ensemble weights: {self.weights}")
        
        # Check if weights exist for models that aren't in model_types
        for weight_key in self.weights.keys():
            if weight_key not in self.model_types:
                logger.warning(f"Weight defined for model '{weight_key}' but this model is not in model_types: {self.model_types}")
        
        # Filter weights to only include models in model_types
        self.weights = {k: v for k, v in self.weights.items() if k in self.model_types}
        logger.info(f"Filtered ensemble weights (only for available models): {self.weights}")
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            for key in self.weights:
                self.weights[key] /= total_weight
            logger.info(f"Normalized ensemble weights (sum = 1.0): {self.weights}")
        
        
        # Initialize models dictionary
        self.models = {}
        
        # Load models
        self._load_models()
        
        logger.info(f"Ensemble ML Strategy initialized with models: {self.model_types}")
        logger.info(f"Model weights: {self.weights}")
        
    def _load_models(self):
        """
        Load pre-trained ML models.
        """
        for model_type in self.model_types:
            model_path = os.path.join("models", f"{model_type}.joblib")
            
            if os.path.exists(model_path):
                model = get_ml_model(model_type)
                if model and model.load(model_path):
                    self.models[model_type] = model
                    logger.info(f"Loaded pre-trained {model_type} model from {model_path}")
                else:
                    logger.warning(f"Failed to load model from {model_path}")
            else:
                logger.warning(f"No pre-trained model found at {model_path}")
        
        # Check if we have at least one model
        if not self.models:
            logger.warning("No models loaded for ensemble strategy")
            
    def generate_signals(self, data):
        """
        Generate trading signals based on ensemble of ML models.
        
        Args:
            data (dict): Dictionary of DataFrames with market data
            
        Returns:
            dict: Dictionary of signals for each symbol
        """
        if not self.models:
            logger.warning("No trained ML models available - returning 'hold' signals")
            signals = {}
            
            for symbol, df in data.items():
                if df.empty:
                    continue
                    
                signals[symbol] = {
                    'action': 'hold',
                    'signal': 0,
                    'signal_changed': False,
                    'price': df['close'].iloc[-1] if not df.empty else 0,
                    'timestamp': df['timestamp'].iloc[-1] if not df.empty else None,
                    'short_ma': df['sma_short'].iloc[-1] if not df.empty and 'sma_short' in df.columns else None,
                    'long_ma': df['sma_long'].iloc[-1] if not df.empty and 'sma_long' in df.columns else None,
                    'rsi': df['rsi'].iloc[-1] if not df.empty and 'rsi' in df.columns else None,
                    'ml_confidence': 0.5  # Default confidence score
                }
                
            return signals
        
        # Generate signals from each model
        all_model_signals = {}
        
        for model_type, model in self.models.items():
            try:
                model_signals = generate_ml_signals(model, data)
                all_model_signals[model_type] = model_signals
                logger.info(f"Generated signals using {model_type} model")
            except Exception as e:
                logger.error(f"Error generating signals with {model_type} model: {e}")
        
        # Combine signals using weighted voting
        ensemble_signals = {}
        
        for symbol in data.keys():
            if symbol not in ensemble_signals and not data[symbol].empty:
                ensemble_signals[symbol] = {
                    'action': 'hold',
                    'signal': 0,
                    'signal_changed': False,
                    'price': data[symbol]['close'].iloc[-1],
                    'timestamp': data[symbol]['timestamp'].iloc[-1],
                    'short_ma': data[symbol]['sma_short'].iloc[-1] if 'sma_short' in data[symbol].columns else None,
                    'long_ma': data[symbol]['sma_long'].iloc[-1] if 'sma_long' in data[symbol].columns else None,
                    'rsi': data[symbol]['rsi'].iloc[-1] if 'rsi' in data[symbol].columns else None,
                    'ml_confidence': 0.5  # Default confidence score
                }
            
            # Skip if the symbol is not in all model signals
            if any(symbol not in model_signals for model_signals in all_model_signals.values()):
                continue
            
            # Calculate weighted vote
            weighted_signal = 0
            total_confidence = 0
            
            for model_type, model_signals in all_model_signals.items():
                if symbol in model_signals:
                    signal = model_signals[symbol]['signal']
                    confidence = model_signals[symbol].get('ml_confidence', 0.5)
                    weight = self.weights.get(model_type, 1.0 / len(self.models))
                    
                    weighted_signal += signal * weight * confidence
                    total_confidence += weight * confidence
            
            # Normalize confidence
            if total_confidence > 0:
                normalized_confidence = min(abs(weighted_signal) / total_confidence, 1.0)
            else:
                normalized_confidence = 0.5
            
            # Determine final signal
            if weighted_signal > config.ML_CONFIDENCE_THRESHOLD:
                final_signal = 1
                action = 'buy'
            elif weighted_signal < -config.ML_CONFIDENCE_THRESHOLD:
                final_signal = -1
                action = 'sell'
            else:
                final_signal = 0
                action = 'hold'
            
            # Create ensemble signal
            ensemble_signals[symbol] = {
                'action': action,
                'signal': final_signal,
                'signal_changed': False,  # We don't track signal changes in this implementation
                'price': data[symbol]['close'].iloc[-1],
                'timestamp': data[symbol]['timestamp'].iloc[-1],
                'short_ma': data[symbol]['sma_short'].iloc[-1] if 'sma_short' in data[symbol].columns else None,
                'long_ma': data[symbol]['sma_long'].iloc[-1] if 'sma_long' in data[symbol].columns else None,
                'rsi': data[symbol]['rsi'].iloc[-1] if 'rsi' in data[symbol].columns else None,
                'ml_confidence': normalized_confidence
            }
            
            logger.info(f"Ensemble ML Strategy signal for {symbol}: {action} (confidence: {normalized_confidence:.2f})")
        
        return ensemble_signals

def train_ensemble_model(model_types=None, symbols=None, period='1y'):
    """
    Train all models used in the ensemble strategy.
    
    Args:
        model_types (list): List of model types to train
        symbols (list): List of symbols to train on
        period (str): Period of historical data to use for training
        
    Returns:
        dict: Dictionary of trained models
    """
    model_types = model_types or ['random_forest', 'gradient_boosting']
    symbols = symbols or config.SYMBOLS
    
    logger.info(f"Training ensemble models: {model_types} on {symbols} with {period} of data")
    
    # Import here to avoid circular imports
    from src.ml_models import train_ml_model
    
    trained_models = {}
    
    for model_type in model_types:
        try:
            model = train_ml_model(model_type)
            if model:
                trained_models[model_type] = model
                logger.info(f"Successfully trained {model_type} model")
            else:
                logger.error(f"Failed to train {model_type} model")
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
    
    return trained_models