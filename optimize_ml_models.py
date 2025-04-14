#!/usr/bin/env python
"""
Script to fine-tune ML models and optimize ensemble weights for trading strategies.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from src.yfinance_data import YFinanceData
from src.feature_engineering import add_technical_indicators
from src.ml_models import MLModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def load_and_prepare_data(symbols, period='1y'):
    """
    Load and prepare data for model training.
    
    Args:
        symbols (list): List of stock symbols
        period (str): Time period to fetch data for
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    logger.info(f"Loading data for {symbols} over {period} period")
    
    # Initialize YFinanceData
    data_source = YFinanceData()
    
    # Get historical data
    historical_data = data_source.get_historical_data(symbols, period=period)
    
    # Prepare data for ML
    X_all = []
    y_all = []
    
    for symbol, data in historical_data.items():
        logger.info(f"Preparing data for {symbol} with {len(data)} data points")
        
        # Add features
        df = add_technical_indicators(data.copy())
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        # Create target variable (1 for price increase, 0 for decrease or no change)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop the last row since we don't have a target for it
        df = df[:-1]
        
        # Select features
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_std',
            'atr', 'adx', 'cci', 'stoch_k', 'stoch_d',
            'obv', 'roc', 'willr', 'mom', 'ppo', 'dx'
        ]
        
        # Add lag features
        for feature in ['close', 'volume', 'rsi', 'macd']:
            for lag in [1, 2, 3, 5]:
                col_name = f"{feature}_lag_{lag}"
                df[col_name] = df[feature].shift(lag)
                features.append(col_name)
        
        # Drop NaN values again after adding lag features
        df.dropna(inplace=True)
        
        # Add to combined dataset
        X_all.append(df[features])
        y_all.append(df['target'])
    
    # Combine data from all symbols
    X = pd.concat(X_all)
    y = pd.concat(y_all)
    
    # Split data into train and test sets (time series split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    logger.info(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test

def optimize_random_forest(X_train, X_test, y_train, y_test):
    """
    Optimize RandomForest model using grid search.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        
    Returns:
        tuple: Best model, best parameters, performance metrics
    """
    logger.info("Optimizing RandomForest model")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=42)
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_rf.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    logger.info(f"Best RandomForest parameters: {grid_search.best_params_}")
    logger.info(f"RandomForest metrics: {metrics}")
    
    # Save model
    joblib.dump(best_rf, 'models/random_forest_optimized.joblib')
    
    return best_rf, grid_search.best_params_, metrics

def optimize_gradient_boosting(X_train, X_test, y_train, y_test):
    """
    Optimize GradientBoosting model using grid search.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        
    Returns:
        tuple: Best model, best parameters, performance metrics
    """
    logger.info("Optimizing GradientBoosting model")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Initialize model
    gb = GradientBoostingClassifier(random_state=42)
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_gb = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_gb.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    logger.info(f"Best GradientBoosting parameters: {grid_search.best_params_}")
    logger.info(f"GradientBoosting metrics: {metrics}")
    
    # Save model
    joblib.dump(best_gb, 'models/gradient_boosting_optimized.joblib')
    
    return best_gb, grid_search.best_params_, metrics

def optimize_ensemble_weights(X_test, y_test, rf_model, gb_model):
    """
    Optimize ensemble weights using grid search.
    
    Args:
        X_test, y_test: Test data
        rf_model: RandomForest model
        gb_model: GradientBoosting model
        
    Returns:
        dict: Optimal weights
    """
    logger.info("Optimizing ensemble weights")
    
    # Get predictions from each model
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
    
    # Try different weight combinations
    weights = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
    
    best_f1 = 0
    best_weights = {'gradient_boosting': 0.5, 'random_forest': 0.5}
    
    results = []
    
    for w_gb in weights:
        w_rf = 1 - w_gb
        
        # Combine predictions
        ensemble_pred_proba = w_gb * gb_pred_proba + w_rf * rf_pred_proba
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        f1 = f1_score(y_test, ensemble_pred)
        accuracy = accuracy_score(y_test, ensemble_pred)
        precision = precision_score(y_test, ensemble_pred)
        recall = recall_score(y_test, ensemble_pred)
        
        results.append({
            'gb_weight': w_gb,
            'rf_weight': w_rf,
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = {'gradient_boosting': w_gb, 'random_forest': w_rf}
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['gb_weight'], results_df['f1'], marker='o', label='F1 Score')
    plt.plot(results_df['gb_weight'], results_df['accuracy'], marker='s', label='Accuracy')
    plt.plot(results_df['gb_weight'], results_df['precision'], marker='^', label='Precision')
    plt.plot(results_df['gb_weight'], results_df['recall'], marker='d', label='Recall')
    plt.axvline(x=best_weights['gradient_boosting'], color='r', linestyle='--', label=f'Best GB Weight: {best_weights["gradient_boosting"]:.2f}')
    plt.xlabel('GradientBoosting Weight')
    plt.ylabel('Score')
    plt.title('Ensemble Weight Optimization')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/ensemble_weight_optimization.png')
    
    logger.info(f"Best ensemble weights: {best_weights}")
    logger.info(f"Best F1 score: {best_f1:.4f}")
    
    # Save best weights to a file
    with open('models/ensemble_weights_optimized.txt', 'w') as f:
        f.write(f"gradient_boosting: {best_weights['gradient_boosting']}\n")
        f.write(f"random_forest: {best_weights['random_forest']}\n")
    
    return best_weights

def main():
    """Main function to optimize ML models and ensemble weights."""
    # Define symbols
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA']
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(symbols, period='1y')
    
    # Optimize RandomForest model
    rf_model, rf_params, rf_metrics = optimize_random_forest(X_train, X_test, y_train, y_test)
    
    # Optimize GradientBoosting model
    gb_model, gb_params, gb_metrics = optimize_gradient_boosting(X_train, X_test, y_train, y_test)
    
    # Optimize ensemble weights
    best_weights = optimize_ensemble_weights(X_test, y_test, rf_model, gb_model)
    
    # Print summary
    logger.info("Optimization complete!")
    logger.info(f"RandomForest metrics: {rf_metrics}")
    logger.info(f"GradientBoosting metrics: {gb_metrics}")
    logger.info(f"Best ensemble weights: {best_weights}")
    
    # Create optimized config file
    with open('config/optimized_config.py', 'w') as f:
        f.write("# Optimized configuration for ML models and ensemble weights\n\n")
        f.write("# RandomForest parameters\n")
        f.write(f"RF_PARAMS = {rf_params}\n\n")
        f.write("# GradientBoosting parameters\n")
        f.write(f"GB_PARAMS = {gb_params}\n\n")
        f.write("# Ensemble weights\n")
        f.write(f"ENSEMBLE_WEIGHTS = {best_weights}\n")
    
    logger.info("Optimized config saved to config/optimized_config.py")

if __name__ == "__main__":
    main()