"""
Machine Learning models for the Alpaca Trading Bot.

This module contains implementations of ML-based trading strategies.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from config import config
from src.logger import get_logger
from src.feature_engineering import add_technical_indicators, create_target_variable

logger = get_logger()

class MLModel:
    """Base class for ML models."""
    def __init__(self, name):
        self.name = name
        self.model = None
        self.is_trained = False
        self.scaler = StandardScaler()
        self.feature_names = []
        logger.info(f"ML Model '{name}' initialized")
        
    def preprocess_data(self, data):
        if data is None or data.empty:
            return None, None
            
        # Add technical indicators
        data = add_technical_indicators(data)
            
        # Create target variable
        data['target'] = create_target_variable(data)
            
        # Drop rows with NaN values
        data = data.dropna()
        
        if data.empty:
            return None, None
            
        # Select features
        features = config.ML_FEATURES
        self.feature_names = features
        
        # Check if all features exist in the dataframe
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            features = [f for f in features if f in data.columns]
            
        if not features:
            logger.error("No valid features found")
            return None, None
            
        # Extract features and target
        X = data[features].values
        y = data['target'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
        
    def train(self, X, y):
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            return False
            
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train model
            self._train_model(X_train, y_train)
            
            # Evaluate model
            metrics = self.evaluate(X_test, y_test)
            logger.info(f"Model metrics: {metrics}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
            
    def _train_model(self, X_train, y_train):
        raise NotImplementedError("Subclasses must implement _train_model()")
        
    def predict(self, X):
        if not self.is_trained or self.model is None:
            return np.zeros(len(X) if hasattr(X, '__len__') else 1)
            
        try:
            X_scaled = self.scaler.transform(X)
            return self._predict_model(X_scaled)
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.zeros(len(X) if hasattr(X, '__len__') else 1)
            
    def _predict_model(self, X_scaled):
        raise NotImplementedError("Subclasses must implement _predict_model()")
        
    def predict_proba(self, X):
        if not self.is_trained or self.model is None:
            return np.zeros((len(X) if hasattr(X, '__len__') else 1, 3))
            
        try:
            X_scaled = self.scaler.transform(X)
            return self._predict_proba_model(X_scaled)
            
        except Exception as e:
            logger.error(f"Error making probability predictions: {e}")
            return np.zeros((len(X) if hasattr(X, '__len__') else 1, 3))
            
    def _predict_proba_model(self, X_scaled):
        try:
            return self.model.predict_proba(X_scaled)
        except:
            # If model doesn't support predict_proba, return one-hot encoded predictions
            preds = self._predict_model(X_scaled)
            proba = np.zeros((len(X_scaled), 3))
            for i, p in enumerate(preds):
                if p == -1:
                    proba[i, 0] = 1.0
                elif p == 0:
                    proba[i, 1] = 1.0
                else:
                    proba[i, 2] = 1.0
            return proba
        
    def evaluate(self, X, y):
        if not self.is_trained or self.model is None:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
        try:
            X_scaled = self.scaler.transform(X)
            y_pred = self._predict_model(X_scaled)
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
    def save(self, path):
        if not self.is_trained or self.model is None:
            return False
            
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'name': self.name
            }
            
            joblib.dump(model_data, path)
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
            
    def load(self, path):
        try:
            if not os.path.exists(path):
                return False
                
            model_data = joblib.load(path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            self.name = model_data['name']
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
            
    def get_feature_importance(self):
        if not self.is_trained or self.model is None:
            return {}
            
        try:
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_[0])
            else:
                return {}
                
            # Create dictionary of feature importance
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                if i < len(importance):
                    feature_importance[feature] = float(importance[i])
                
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}


class RandomForestModel(MLModel):
    """Random Forest model for trading."""
    def __init__(self):
        super().__init__("RandomForest")
        
    def _train_model(self, X_train, y_train):
        self.model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=config.RF_RANDOM_STATE
        )
        
        self.model.fit(X_train, y_train)
        
    def _predict_model(self, X_scaled):
        return self.model.predict(X_scaled)


class GradientBoostingModel(MLModel):
    """Gradient Boosting model for trading."""
    def __init__(self, library='sklearn'):
        super().__init__(f"GradientBoosting-{library}")
        self.library = library
        
    def _train_model(self, X_train, y_train):
        self.model = GradientBoostingClassifier(
            learning_rate=config.GB_LEARNING_RATE,
            max_depth=config.GB_MAX_DEPTH,
            n_estimators=config.GB_N_ESTIMATORS,
            subsample=config.GB_SUBSAMPLE,
            random_state=config.GB_RANDOM_STATE
        )
            
        self.model.fit(X_train, y_train)
        
    def _predict_model(self, X_scaled):
        return self.model.predict(X_scaled)


def get_ml_model(model_type='random_forest'):
    """Get ML model instance by type."""
    models = {
        'gradient_boosting': GradientBoostingModel,
        'random_forest': RandomForestModel
    }
    
    logger.info(f"Requested ML model type: '{model_type}'")
    logger.info(f"Available model types: {list(models.keys())}")
    
    # Check if the model type is 'gradient_boosting_ensemble' and provide a helpful message
    if model_type.lower() == 'gradient_boosting_ensemble':
        logger.warning("'gradient_boosting_ensemble' is not a valid model type. Did you mean to use the EnsembleMLStrategy instead?")
        logger.warning("For ML models, use 'gradient_boosting' or 'random_forest'. For ensemble strategy, use 'ensemble_ml' as the strategy name.")
    
    model_class = models.get(model_type.lower())
    if not model_class:
        logger.error(f"ML model type '{model_type}' not found")
        return None
        
    return model_class()


def train_ml_model(model_type=None):
    """Train an ML model for trading."""
    model_type = model_type or config.ML_STRATEGY_TYPE
    
    # Get model instance
    model = get_ml_model(model_type)
    
    if model is None:
        logger.error(f"Failed to create model of type '{model_type}'")
        return None
    
    # Get historical data
    from src.yfinance_data import YFinanceData
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


def generate_ml_signals(model, data):
    """Generate trading signals using an ML model."""
    if model is None or not model.is_trained:
        return {}
        
    signals = {}
    
    for symbol, df in data.items():
        if df.empty:
            continue
            
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if df.empty:
            continue
            
        # Extract features
        features = [f for f in config.ML_FEATURES if f in df.columns]
        
        if not features:
            continue
            
        X = df[features].values[-1:] # Get the most recent data point
        
        # Scale features
        X_scaled = model.scaler.transform(X)
        
        # Make predictions
        prediction = model._predict_model(X_scaled)[0]
        
        # Get prediction probabilities if available
        try:
            proba = model._predict_proba_model(X_scaled)[0]
            confidence = max(proba)
        except:
            confidence = 0.5
        
        # Determine action
        if prediction == 1:
            action = 'buy'
            signal = 1
        elif prediction == -1:
            action = 'sell'
            signal = -1
        else:
            action = 'hold'
            signal = 0
            
        signals[symbol] = {
            'action': action,
            'signal': signal,
            'price': df['close'].iloc[-1],
            'timestamp': df['timestamp'].iloc[-1],
            'ml_confidence': confidence
        }
        
    return signals
