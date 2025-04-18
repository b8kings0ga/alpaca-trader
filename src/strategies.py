"""
Trading strategies for the Alpaca Trading Bot.
"""
import pandas as pd
import numpy as np
import os.path
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from config import config
from src.logger import get_logger
from src.ml_models import get_ml_model, generate_ml_signals
from src.strategy_base import Strategy

logger = get_logger()

class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover strategy.
    
    Buy when short MA crosses above long MA.
    Sell when short MA crosses below long MA.
    """
    def __init__(self):
        """
        Initialize the Moving Average Crossover strategy.
        """
        super().__init__("Moving Average Crossover")
        # Reload the config to ensure we have the latest values
        import importlib
        importlib.reload(config)
        
        # Log the strategy parameters
        logger.info(f"Initializing Moving Average Crossover strategy with parameters:")
        logger.info(f"SHORT_WINDOW: {config.SHORT_WINDOW}")
        logger.info(f"LONG_WINDOW: {config.LONG_WINDOW}")
        logger.info(f"RSI_PERIOD: {config.RSI_PERIOD}")
        logger.info(f"RSI_OVERSOLD: {config.RSI_OVERSOLD}")
        logger.info(f"RSI_OVERBOUGHT: {config.RSI_OVERBOUGHT}")
        
        self.short_window = config.SHORT_WINDOW
        self.long_window = config.LONG_WINDOW
        
    def generate_signals(self, data):
        """
        Generate trading signals based on Moving Average Crossover.
        
        Args:
            data (dict): Dictionary of DataFrames with market data
            
        Returns:
            dict: Dictionary of signals for each symbol
        """
        signals = {}
        
        for symbol, df in data.items():
            logger.info(f"Analyzing data for {symbol}: {len(df)} data points available")
            logger.info(f"Strategy requires at least {self.long_window} data points")
            
            if df.empty:
                logger.warning(f"DataFrame is empty for {symbol}")
                continue
                
            if len(df) < self.long_window:
                logger.warning(f"Not enough data for {symbol} to generate signals. Need {self.long_window}, got {len(df)}")
                continue
                
            # Create a copy of the DataFrame
            df = df.copy()
            
            # Create signal column (1 = buy, -1 = sell, 0 = hold)
            df['signal'] = 0
            
            # Generate signals
            # Buy signal: short MA crosses above long MA
            df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
            buy_count = (df['signal'] == 1).sum()
            logger.info(f"Generated {buy_count} buy signals for {symbol}")
            
            # Sell signal: short MA crosses below long MA
            df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1
            sell_count = (df['signal'] == -1).sum()
            logger.info(f"Generated {sell_count} sell signals for {symbol}")
            logger.info(f"Last 5 rows of data for {symbol}:")
            for i in range(min(5, len(df))):
                idx = len(df) - 5 + i
                if idx >= 0:
                    row = df.iloc[idx]
                    logger.info(f"  Row {idx}: sma_short={row['sma_short']:.2f}, sma_long={row['sma_long']:.2f}, signal={row['signal']}, close={row['close']:.2f}")
            
            # Get the latest signal
            latest_signal = df['signal'].iloc[-1]
            logger.info(f"Latest signal for {symbol}: {latest_signal} ({self._get_action(latest_signal)})")
            
            # Check for crossover (signal change)
            signal_changed = False
            if len(df) >= 2:
                prev_signal = df['signal'].iloc[-2]
                signal_changed = prev_signal != latest_signal and latest_signal != 0
                if signal_changed:
                    logger.info(f"Signal changed for {symbol}: {prev_signal} -> {latest_signal} ({self._get_action(prev_signal)} -> {self._get_action(latest_signal)})")
            
            signals[symbol] = {
                'action': self._get_action(latest_signal),
                'signal': latest_signal,
                'signal_changed': signal_changed,
                'price': df['close'].iloc[-1],
                'timestamp': df['timestamp'].iloc[-1],
                'short_ma': df['sma_short'].iloc[-1],
                'long_ma': df['sma_long'].iloc[-1],
                'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else None
            }
            
        return signals
    
    def optimize(self, historical_data, target_metric='profit', test_period='1mo'):
        """
        Optimize strategy parameters based on historical performance.
        
        This method tests different combinations of short and long window parameters
        to find the optimal settings that maximize the target metric.
        
        Args:
            historical_data (dict): Dictionary of DataFrames with historical market data
            target_metric (str): Metric to optimize for ('profit', 'sharpe', 'win_rate', etc.)
            test_period (str): Period to use for testing optimization results
            
        Returns:
            dict: Dictionary containing optimization results and new parameters
        """
        logger.info(f"Optimizing MovingAverageCrossover strategy for {target_metric}...")
        
        # Define parameter ranges to test
        short_windows = range(5, 51, 5)  # 5, 10, 15, ..., 50
        long_windows = range(20, 201, 20)  # 20, 40, 60, ..., 200
        
        # Store results for each parameter combination
        results = []
        
        # Test each parameter combination
        for short_window in short_windows:
            for long_window in long_windows:
                # Skip invalid combinations (short window must be less than long window)
                if short_window >= long_window:
                    continue
                
                logger.info(f"Testing parameters: short_window={short_window}, long_window={long_window}")
                
                # Create a copy of the strategy with these parameters
                strategy_copy = MovingAverageCrossover()
                strategy_copy.short_window = short_window
                strategy_copy.long_window = long_window
                
                # Prepare data with these parameters
                processed_data = {}
                for symbol, df in historical_data.items():
                    if df.empty:
                        continue
                    
                    # Create a copy of the DataFrame
                    df_copy = df.copy()
                    
                    # Calculate moving averages with these parameters
                    df_copy['sma_short'] = df_copy['close'].rolling(window=short_window).mean()
                    df_copy['sma_long'] = df_copy['close'].rolling(window=long_window).mean()
                    
                    # Drop NaN values
                    df_copy = df_copy.dropna()
                    
                    processed_data[symbol] = df_copy
                
                # Run backtest with these parameters
                backtest_results = strategy_copy.backtest(processed_data)
                
                # Calculate average performance across all symbols
                avg_performance = {}
                for metric in ['returns', 'strategy_returns', 'cumulative_returns',
                              'strategy_cumulative_returns', 'sharpe_ratio',
                              'max_drawdown', 'win_rate', 'profit']:
                    values = [result[metric] for symbol, result in backtest_results.items()]
                    avg_performance[metric] = sum(values) / len(values) if values else 0
                
                # Store results
                results.append({
                    'short_window': short_window,
                    'long_window': long_window,
                    'performance': avg_performance
                })
        
        # Find the best parameter combination based on the target metric
        if not results:
            logger.warning("No valid parameter combinations found during optimization")
            return {
                'success': False,
                'message': 'No valid parameter combinations found',
                'original_params': {
                    'short_window': self.short_window,
                    'long_window': self.long_window
                }
            }
        
        # Sort results by the target metric
        sorted_results = sorted(results,
                               key=lambda x: x['performance'][target_metric],
                               reverse=True)
        
        best_params = sorted_results[0]
        
        logger.info(f"Optimization complete. Best parameters: "
                   f"short_window={best_params['short_window']}, "
                   f"long_window={best_params['long_window']}")
        logger.info(f"Performance improvement: "
                   f"{target_metric} increased from "
                   f"{results[0]['performance'][target_metric]:.4f} to "
                   f"{best_params['performance'][target_metric]:.4f}")
        
        # Update strategy parameters
        original_short = self.short_window
        original_long = self.long_window
        self.short_window = best_params['short_window']
        self.long_window = best_params['long_window']
        
        # Return optimization results
        return {
            'success': True,
            'original_params': {
                'short_window': original_short,
                'long_window': original_long
            },
            'new_params': {
                'short_window': self.short_window,
                'long_window': self.long_window
            },
            'performance_improvement': {
                target_metric: best_params['performance'][target_metric] - results[0]['performance'][target_metric]
            },
            'all_results': sorted_results[:5]  # Return top 5 parameter combinations
        }
    
    def _get_action(self, signal):
        """
        Convert signal to action string.
        
        Args:
            signal (int): Signal value (1, -1, or 0)
            
        Returns:
            str: Action string ('buy', 'sell', or 'hold')
        """
        if signal == 1:
            return 'buy'
        elif signal == -1:
            return 'sell'
        else:
            return 'hold'


class RSIStrategy(Strategy):
    """
    Relative Strength Index (RSI) strategy.
    
    Buy when RSI crosses below oversold threshold.
    Sell when RSI crosses above overbought threshold.
    """
    def __init__(self):
        """
        Initialize the RSI strategy.
        """
        super().__init__("RSI Strategy")
        self.period = config.RSI_PERIOD
        self.oversold = config.RSI_OVERSOLD
        self.overbought = config.RSI_OVERBOUGHT
        
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI.
        
        Args:
            data (dict): Dictionary of DataFrames with market data
            
        Returns:
            dict: Dictionary of signals for each symbol
        """
        signals = {}
        
        for symbol, df in data.items():
            if df.empty or len(df) < self.period:
                logger.warning(f"Not enough data for {symbol} to generate signals")
                continue
                
            # Create a copy of the DataFrame
            df = df.copy()
            
            # Create signal column (1 = buy, -1 = sell, 0 = hold)
            df['signal'] = 0
            
            # Generate signals
            # Buy signal: RSI crosses below oversold threshold
            df.loc[df['rsi'] < self.oversold, 'signal'] = 1
            
            # Sell signal: RSI crosses above overbought threshold
            df.loc[df['rsi'] > self.overbought, 'signal'] = -1
            
            # Get the latest signal
            latest_signal = df['signal'].iloc[-1]
            
            # Check for crossover (signal change)
            signal_changed = False
            if len(df) >= 2:
                prev_signal = df['signal'].iloc[-2]
                signal_changed = prev_signal != latest_signal and latest_signal != 0
            
            signals[symbol] = {
                'action': self._get_action(latest_signal),
                'signal': latest_signal,
                'signal_changed': signal_changed,
                'price': df['close'].iloc[-1],
                'rsi': df['rsi'].iloc[-1],
                'short_ma': df['sma_short'].iloc[-1] if 'sma_short' in df.columns else None,
                'long_ma': df['sma_long'].iloc[-1] if 'sma_long' in df.columns else None,
                'timestamp': df['timestamp'].iloc[-1]
            }
            
        return signals
    
    def optimize(self, historical_data, target_metric='profit', test_period='1mo'):
        """
        Optimize RSI strategy parameters based on historical performance.
        
        This method tests different combinations of RSI period, oversold, and overbought
        parameters to find the optimal settings that maximize the target metric.
        
        Args:
            historical_data (dict): Dictionary of DataFrames with historical market data
            target_metric (str): Metric to optimize for ('profit', 'sharpe', 'win_rate', etc.)
            test_period (str): Period to use for testing optimization results
            
        Returns:
            dict: Dictionary containing optimization results and new parameters
        """
        logger.info(f"Optimizing RSIStrategy for {target_metric}...")
        
        # Define parameter ranges to test
        periods = range(5, 31, 5)  # 5, 10, 15, 20, 25, 30
        oversold_values = range(20, 41, 5)  # 20, 25, 30, 35, 40
        overbought_values = range(60, 81, 5)  # 60, 65, 70, 75, 80
        
        # Store results for each parameter combination
        results = []
        
        # Test each parameter combination
        for period in periods:
            for oversold in oversold_values:
                for overbought in overbought_values:
                    # Skip invalid combinations (oversold must be less than overbought)
                    if oversold >= overbought:
                        continue
                    
                    logger.info(f"Testing parameters: period={period}, oversold={oversold}, overbought={overbought}")
                    
                    # Create a copy of the strategy with these parameters
                    strategy_copy = RSIStrategy()
                    strategy_copy.period = period
                    strategy_copy.oversold = oversold
                    strategy_copy.overbought = overbought
                    
                    # Prepare data with these parameters
                    processed_data = {}
                    for symbol, df in historical_data.items():
                        if df.empty:
                            continue
                        
                        # Create a copy of the DataFrame
                        df_copy = df.copy()
                        
                        # Calculate RSI with these parameters
                        delta = df_copy['close'].diff()
                        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
                        
                        rs = gain / loss
                        df_copy['rsi'] = 100 - (100 / (1 + rs))
                        
                        # Drop NaN values
                        df_copy = df_copy.dropna()
                        
                        processed_data[symbol] = df_copy
                    
                    # Run backtest with these parameters
                    backtest_results = strategy_copy.backtest(processed_data)
                    
                    # Calculate average performance across all symbols
                    avg_performance = {}
                    for metric in ['returns', 'strategy_returns', 'cumulative_returns',
                                  'strategy_cumulative_returns', 'sharpe_ratio',
                                  'max_drawdown', 'win_rate', 'profit']:
                        values = [result[metric] for symbol, result in backtest_results.items()]
                        avg_performance[metric] = sum(values) / len(values) if values else 0
                    
                    # Store results
                    results.append({
                        'period': period,
                        'oversold': oversold,
                        'overbought': overbought,
                        'performance': avg_performance
                    })
        
        # Find the best parameter combination based on the target metric
        if not results:
            logger.warning("No valid parameter combinations found during optimization")
            return {
                'success': False,
                'message': 'No valid parameter combinations found',
                'original_params': {
                    'period': self.period,
                    'oversold': self.oversold,
                    'overbought': self.overbought
                }
            }
        
        # Sort results by the target metric
        sorted_results = sorted(results,
                               key=lambda x: x['performance'][target_metric],
                               reverse=True)
        
        best_params = sorted_results[0]
        
        logger.info(f"Optimization complete. Best parameters: "
                   f"period={best_params['period']}, "
                   f"oversold={best_params['oversold']}, "
                   f"overbought={best_params['overbought']}")
        logger.info(f"Performance improvement: "
                   f"{target_metric} increased from "
                   f"{results[0]['performance'][target_metric]:.4f} to "
                   f"{best_params['performance'][target_metric]:.4f}")
        
        # Update strategy parameters
        original_period = self.period
        original_oversold = self.oversold
        original_overbought = self.overbought
        
        self.period = best_params['period']
        self.oversold = best_params['oversold']
        self.overbought = best_params['overbought']
        
        # Return optimization results
        return {
            'success': True,
            'original_params': {
                'period': original_period,
                'oversold': original_oversold,
                'overbought': original_overbought
            },
            'new_params': {
                'period': self.period,
                'oversold': self.oversold,
                'overbought': self.overbought
            },
            'performance_improvement': {
                target_metric: best_params['performance'][target_metric] - results[0]['performance'][target_metric]
            },
            'all_results': sorted_results[:5]  # Return top 5 parameter combinations
        }
    
    def _get_action(self, signal):
        """
        Convert signal to action string.
        
        Args:
            signal (int): Signal value (1, -1, or 0)
            
        Returns:
            str: Action string ('buy', 'sell', or 'hold')
        """
        if signal == 1:
            return 'buy'
        elif signal == -1:
            return 'sell'
        else:
            return 'hold'

class MLStrategy(Strategy):
    """
    Machine Learning based strategy.
    
    This strategy uses machine learning models to predict price movements and generate trading signals.
    Currently supports the following models:
    - Random Forest
    - Gradient Boosting (XGBoost/LightGBM)
    
    The models are trained on historical price data and technical indicators to predict
    future price movements. The predictions are then converted to trading signals.
    """
    def __init__(self, model_type=None):
        """
        Initialize the ML strategy.
        
        Args:
            model_type (str): Type of ML model to use ('random_forest', 'gradient_boosting')
        """
        super().__init__("ML Strategy")
        self.model_type = model_type or config.ML_STRATEGY_TYPE
        self.model = None
        self.model_path = os.path.join("models", f"{self.model_type}.joblib")
        
        # Try to load a pre-trained model if it exists
        self._load_model()
        
        logger.info(f"ML Strategy initialized with model type: {self.model_type}")
        
    def _load_model(self):
        """
        Load a pre-trained ML model if it exists.
        """
        try:
            if os.path.exists(self.model_path):
                self.model = get_ml_model(self.model_type)
                if self.model and self.model.load(self.model_path):
                    logger.info(f"Loaded pre-trained {self.model_type} model from {self.model_path}")
                else:
                    logger.warning(f"Failed to load model from {self.model_path}")
            else:
                logger.info(f"No pre-trained model found at {self.model_path}")
                self.model = get_ml_model(self.model_type)
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self.model = get_ml_model(self.model_type)
        
    def generate_signals(self, data):
        """
        Generate trading signals based on ML models.
        
        Args:
            data (dict): Dictionary of DataFrames with market data
            
        Returns:
            dict: Dictionary of signals for each symbol
        """
        if not self.model or not self.model.is_trained:
            logger.warning("No trained ML model available - returning 'hold' signals")
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
        
        # Use the ml_models module to generate signals
        try:
            ml_signals = generate_ml_signals(self.model, data)
            
            # Process signals to add signal_changed flag
            signals = {}
            for symbol, signal_data in ml_signals.items():
                if symbol not in data or data[symbol].empty:
                    continue
                    
                df = data[symbol]
                
                # Check for signal change
                signal_changed = False
                current_signal = signal_data['signal']
                
                # Create a new signal entry with additional information
                signals[symbol] = {
                    'action': signal_data['action'],
                    'signal': current_signal,
                    'signal_changed': signal_changed,
                    'price': signal_data['price'],
                    'timestamp': signal_data['timestamp'],
                    'short_ma': df['sma_short'].iloc[-1] if 'sma_short' in df.columns else None,
                    'long_ma': df['sma_long'].iloc[-1] if 'sma_long' in df.columns else None,
                    'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else None,
                    'ml_confidence': signal_data.get('ml_confidence', 0.5)
                }
                
                logger.info(f"ML Strategy signal for {symbol}: {signals[symbol]['action']} (confidence: {signals[symbol]['ml_confidence']:.2f})")
                
            return signals
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
            return {}
    
    def optimize(self, historical_data, target_metric='profit', test_period='1mo'):
        """
        Optimize ML strategy hyperparameters based on historical performance.
        
        This method tests different hyperparameter combinations for the ML model
        to find the optimal settings that maximize the target metric.
        
        Args:
            historical_data (dict): Dictionary of DataFrames with historical market data
            target_metric (str): Metric to optimize for ('profit', 'sharpe', 'win_rate', etc.)
            test_period (str): Period to use for testing optimization results
            
        Returns:
            dict: Dictionary containing optimization results and new hyperparameters
        """
        logger.info(f"Optimizing MLStrategy for {target_metric}...")
        
        # Check if model is available
        if not self.model:
            logger.warning("No ML model available for optimization")
            return {
                'success': False,
                'message': 'No ML model available for optimization',
                'original_params': {}
            }
        
        # Get hyperparameter grid based on model type
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:
            logger.warning(f"Unknown model type: {self.model_type}")
            return {
                'success': False,
                'message': f'Unknown model type: {self.model_type}',
                'original_params': {}
            }
        
        # Store original hyperparameters
        original_params = self.model.get_params()
        
        # Generate all hyperparameter combinations
        import itertools
        param_combinations = []
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        for combination in itertools.product(*values):
            param_dict = {keys[i]: combination[i] for i in range(len(keys))}
            param_combinations.append(param_dict)
        
        logger.info(f"Testing {len(param_combinations)} hyperparameter combinations")
        
        # Store results for each hyperparameter combination
        results = []
        
        # Test each hyperparameter combination
        for params in param_combinations:
            logger.info(f"Testing hyperparameters: {params}")
            
            # Create a copy of the strategy with these hyperparameters
            strategy_copy = MLStrategy(model_type=self.model_type)
            
            # Set hyperparameters for the model
            if strategy_copy.model:
                strategy_copy.model.set_params(**params)
                
                # Train the model on historical data
                try:
                    # Prepare features and target
                    features, target = self._prepare_training_data(historical_data)
                    
                    # Train the model
                    strategy_copy.model.train(features, target)
                    
                    # Run backtest with the trained model
                    backtest_results = strategy_copy.backtest(historical_data)
                    
                    # Calculate average performance across all symbols
                    avg_performance = {}
                    for metric in ['returns', 'strategy_returns', 'cumulative_returns',
                                  'strategy_cumulative_returns', 'sharpe_ratio',
                                  'max_drawdown', 'win_rate', 'profit']:
                        values = [result[metric] for symbol, result in backtest_results.items()]
                        avg_performance[metric] = sum(values) / len(values) if values else 0
                    
                    # Store results
                    results.append({
                        'params': params,
                        'performance': avg_performance
                    })
                    
                except Exception as e:
                    logger.error(f"Error training model with hyperparameters {params}: {e}")
        
        # Find the best hyperparameter combination based on the target metric
        if not results:
            logger.warning("No valid hyperparameter combinations found during optimization")
            return {
                'success': False,
                'message': 'No valid hyperparameter combinations found',
                'original_params': original_params
            }
        
        # Sort results by the target metric
        sorted_results = sorted(results,
                               key=lambda x: x['performance'][target_metric],
                               reverse=True)
        
        best_params = sorted_results[0]['params']
        
        logger.info(f"Optimization complete. Best hyperparameters: {best_params}")
        logger.info(f"Performance improvement: "
                   f"{target_metric} increased from "
                   f"{results[0]['performance'][target_metric]:.4f} to "
                   f"{sorted_results[0]['performance'][target_metric]:.4f}")
        
        # Update model hyperparameters
        if self.model:
            self.model.set_params(**best_params)
            
            # Retrain the model with the best hyperparameters
            try:
                # Prepare features and target
                features, target = self._prepare_training_data(historical_data)
                
                # Train the model
                self.model.train(features, target)
                
                # Save the optimized model
                self.model.save(self.model_path)
                logger.info(f"Saved optimized model to {self.model_path}")
            except Exception as e:
                logger.error(f"Error retraining model with best hyperparameters: {e}")
        
        # Return optimization results
        return {
            'success': True,
            'original_params': original_params,
            'new_params': best_params,
            'performance_improvement': {
                target_metric: sorted_results[0]['performance'][target_metric] - results[0]['performance'][target_metric]
            },
            'all_results': sorted_results[:5]  # Return top 5 hyperparameter combinations
        }
    
    def _prepare_training_data(self, historical_data):
        """
        Prepare training data for ML model.
        
        Args:
            historical_data (dict): Dictionary of DataFrames with historical market data
            
        Returns:
            tuple: (features, target) for training the ML model
        """
        features = []
        target = []
        
        for symbol, df in historical_data.items():
            if df.empty:
                continue
                
            # Create a copy of the DataFrame
            df = df.copy()
            
            # Prepare features (use all available technical indicators)
            symbol_features = df.drop(['timestamp', 'open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
            
            # Create target (1 for price increase, -1 for price decrease, 0 for no change)
            df['target'] = np.sign(df['close'].shift(-1) - df['close'])
            
            # Drop NaN values
            df = df.dropna()
            
            # Add to features and target lists
            features.append(symbol_features)
            target.append(df['target'])
        
        # Concatenate features and target from all symbols
        if features and target:
            return pd.concat(features), pd.concat(target)
        else:
            return pd.DataFrame(), pd.Series()


class DualMovingAverageYF(Strategy):
    """
    Dual Moving Average strategy using yfinance data.
    
    Buy when short MA crosses above long MA.
    Sell when short MA crosses below long MA.
    
    This strategy uses yfinance to fetch data instead of Alpaca API.
    """
    def __init__(self):
        """
        Initialize the Dual Moving Average strategy.
        """
        super().__init__("Dual Moving Average YF")
        self.short_window = config.SHORT_WINDOW
        self.long_window = config.LONG_WINDOW
        
    def fetch_data(self, symbols, period="1mo", interval="1d"):
        """
        Fetch data from yfinance.
        
        Args:
            symbols (list): List of stock symbols
            period (str): Period to fetch data for (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Interval between data points (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            dict: Dictionary of DataFrames with market data
        """
        data = {}
        
        for symbol in symbols:
            try:
                # Fetch data from yfinance
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Reset index to make Date a column
                df = df.reset_index()
                
                # Rename columns to match Alpaca API format
                df = df.rename(columns={
                    'Date': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Add technical indicators
                df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
                df['sma_long'] = df['close'].rolling(window=self.long_window).mean()
                
                # Calculate RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=config.RSI_PERIOD).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=config.RSI_PERIOD).mean()
                
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                data[symbol] = df
                
                # Generate and save a plot of the dual moving averages
                self._generate_plot(df, symbol)
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                
        return data
    
    def _generate_plot(self, df, symbol):
        """
        Generate and save a plot of the dual moving averages.
        
        Args:
            df (DataFrame): DataFrame with market data
            symbol (str): Stock symbol
        """
        try:
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the closing price
            ax.plot(df['timestamp'], df['close'], label='Close Price', alpha=0.5)
            
            # Plot the short and long moving averages
            ax.plot(df['timestamp'], df['sma_short'], label=f'SMA {self.short_window}', linewidth=1.5)
            ax.plot(df['timestamp'], df['sma_long'], label=f'SMA {self.long_window}', linewidth=1.5)
            
            # Add buy/sell signals
            buy_signals = df[(df['sma_short'] > df['sma_long']) & (df['sma_short'].shift(1) <= df['sma_long'].shift(1))]
            sell_signals = df[(df['sma_short'] < df['sma_long']) & (df['sma_short'].shift(1) >= df['sma_long'].shift(1))]
            
            ax.scatter(buy_signals['timestamp'], buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
            ax.scatter(sell_signals['timestamp'], sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
            
            # Set title and labels
            ax.set_title(f'{symbol} - Dual Moving Average Strategy')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save the figure
            os.makedirs('logs/plots', exist_ok=True)
            plt.savefig(f'logs/plots/{symbol}_dual_ma.png')
            plt.close(fig)
            
            logger.info(f"Generated plot for {symbol}")
            
        except Exception as e:
            logger.error(f"Error generating plot for {symbol}: {e}")
    
    def generate_signals(self, data=None):
        """
        Generate trading signals based on Dual Moving Average.
        
        Args:
            data (dict, optional): Dictionary of DataFrames with market data.
                                  If None, data will be fetched from yfinance.
            
        Returns:
            dict: Dictionary of signals for each symbol
        """
        if data is None:
            # Fetch data from yfinance
            data = self.fetch_data(config.SYMBOLS)
        
        signals = {}
        
        for symbol, df in data.items():
            if df.empty or len(df) < self.long_window:
                logger.warning(f"Not enough data for {symbol} to generate signals")
                continue
                
            # Create a copy of the DataFrame
            df = df.copy()
            
            # Create signal column (1 = buy, -1 = sell, 0 = hold)
            df['signal'] = 0
            
            # Generate signals
            # Buy signal: short MA crosses above long MA
            df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
            buy_count = (df['signal'] == 1).sum()
            logger.info(f"DualMA: Generated {buy_count} buy signals for {symbol}")
            
            # Sell signal: short MA crosses below long MA
            df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1
            sell_count = (df['signal'] == -1).sum()
            logger.info(f"DualMA: Generated {sell_count} sell signals for {symbol}")
            
            # Get the latest signal
            latest_signal = df['signal'].iloc[-1]
            
            # Check for crossover (signal change)
            signal_changed = False
            if len(df) >= 2:
                prev_signal = df['signal'].iloc[-2]
                signal_changed = prev_signal != latest_signal and latest_signal != 0
            
            signals[symbol] = {
                'action': self._get_action(latest_signal),
                'signal': latest_signal,
                'signal_changed': signal_changed,
                'price': df['close'].iloc[-1],
                'timestamp': df['timestamp'].iloc[-1],
                'short_ma': df['sma_short'].iloc[-1],
                'long_ma': df['sma_long'].iloc[-1],
                'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else None
            }
            
        return signals
    
    def _get_action(self, signal):
        """
        Convert signal to action string.
        
        Args:
            signal (int): Signal value (1, -1, or 0)
            
        Returns:
            str: Action string ('buy', 'sell', or 'hold')
        """
        if signal == 1:
            return 'buy'
        elif signal == -1:
            return 'sell'
        else:
            return 'hold'


# Factory function to get strategy by name
def get_strategy(strategy_name):
    """
    Get strategy instance by name.
    
    Args:
        strategy_name (str): Strategy name
        
    Returns:
        Strategy: Strategy instance
    """
    strategies = {
        'moving_average_crossover': MovingAverageCrossover,
        'rsi': RSIStrategy,
        'ml': MLStrategy,  # Added ML strategy placeholder
        'dual_ma_yf': DualMovingAverageYF,  # Added Dual Moving Average with yfinance
    }
    
    # Import EnsembleMLStrategy here to avoid circular imports
    if strategy_name.lower() == 'ensemble_ml':
        from src.ensemble_ml_strategy import EnsembleMLStrategy
        return EnsembleMLStrategy()
    
    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        logger.error(f"Strategy '{strategy_name}' not found")
        return None
        
    return strategy_class()
