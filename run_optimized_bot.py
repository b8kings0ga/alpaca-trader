#!/usr/bin/env python
"""
Script to run a trading bot using the optimized ensemble ML strategy.
"""
import os
import time
import logging
import argparse
import pandas as pd
import joblib
from datetime import datetime, timedelta
from src.trader import Trader
from src.yfinance_data import YFinanceData
from src.feature_engineering import add_technical_indicators
from src.notifications import NotificationSystem
from config.config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL
from config.optimized_config import ENSEMBLE_WEIGHTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/optimized_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedEnsembleBot:
    """
    Trading bot that uses an optimized ensemble of ML models to make trading decisions.
    """
    
    def __init__(self, symbols, cash_allocation=0.9, max_positions=5, paper=True, ignore_market_hours=False):
        """
        Initialize the bot.
        
        Args:
            symbols (list): List of stock symbols to trade
            cash_allocation (float): Percentage of cash to allocate for trading
            max_positions (int): Maximum number of positions to hold
            paper (bool): Whether to use paper trading
        """
        self.symbols = symbols
        self.cash_allocation = cash_allocation
        self.max_positions = max_positions
        self.paper = paper
        self.ignore_market_hours = ignore_market_hours
        # Initialize trader
        self.trader = Trader()
        
        # Initialize data source
        self.data_source = YFinanceData()
        
        # Initialize notification system
        self.notifier = NotificationSystem()
        
        # Load ML models
        self.load_models()
        
        logger.info(f"Initialized OptimizedEnsembleBot with {len(symbols)} symbols")
        logger.info(f"Cash allocation: {cash_allocation * 100}%")
        logger.info(f"Max positions: {max_positions}")
        logger.info(f"Paper trading: {paper}")
        logger.info(f"Ignore market hours: {ignore_market_hours}")
    
    def load_models(self):
        """
        Load the ML models.
        """
        try:
            self.rf_model = joblib.load('models/random_forest_optimized.joblib')
            self.gb_model = joblib.load('models/gradient_boosting_optimized.joblib')
            logger.info("Loaded optimized ML models")
            
            # Log ensemble weights
            logger.info(f"Ensemble weights: {ENSEMBLE_WEIGHTS}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def get_market_data(self):
        """
        Get market data for all symbols.
        
        Returns:
            dict: Dictionary of DataFrames with market data
        """
        try:
            # Get historical data
            historical_data = self.data_source.get_historical_data(
                self.symbols, period='1mo'
            )
            
            # Add technical indicators
            for symbol in historical_data:
                historical_data[symbol] = add_technical_indicators(historical_data[symbol])
            
            return historical_data
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
    
    def generate_signals(self, market_data):
        """
        Generate trading signals for all symbols.
        
        Args:
            market_data (dict): Dictionary of DataFrames with market data
            
        Returns:
            dict: Dictionary of trading signals
        """
        signals = {}
        
        for symbol, data in market_data.items():
            try:
                # Drop NaN values
                df = data.dropna()
                
                if df.empty:
                    logger.warning(f"Empty DataFrame for {symbol} after dropping NaN values")
                    continue
                
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
                
                # Get the latest data point
                X = df[features].iloc[-1:].values
                
                # Generate predictions
                rf_pred_proba = self.rf_model.predict_proba(X)[0, 1]
                gb_pred_proba = self.gb_model.predict_proba(X)[0, 1]
                
                # Combine predictions
                ensemble_pred_proba = (
                    ENSEMBLE_WEIGHTS['gradient_boosting'] * gb_pred_proba + 
                    ENSEMBLE_WEIGHTS['random_forest'] * rf_pred_proba
                )
                
                # Define thresholds for buy/sell signals
                buy_threshold = 0.6  # Higher threshold for buy signals
                sell_threshold = 0.4  # Lower threshold for sell signals
                
                # Determine signal
                if ensemble_pred_proba > buy_threshold:
                    signal = 'buy'
                elif ensemble_pred_proba < sell_threshold:
                    signal = 'sell'
                else:
                    signal = 'hold'
                
                # Store signal
                signals[symbol] = {
                    'signal': signal,
                    'probability': ensemble_pred_proba,
                    'price': df.iloc[-1]['close'],
                    'timestamp': df.iloc[-1]['timestamp']
                }
                
                logger.info(f"Generated signal for {symbol}: {signal} (probability: {ensemble_pred_proba:.4f})")
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def execute_trades(self, signals):
        """
        Execute trades based on signals.
        
        Args:
            signals (dict): Dictionary of trading signals
            
        Returns:
            list: List of executed trades
        """
        executed_trades = []
        
        try:
            # Get account information
            logger.info("Attempting to get account information")
            try:
                account = self.trader.api.get_account()
                logger.info(f"Successfully retrieved account information: {account}")
            except Exception as e:
                logger.error(f"Error getting account information: {e}")
                logger.error(f"Trader API object: {self.trader.api}")
                raise
            cash = float(account.cash)
            equity = float(account.equity)
            
            # Get current positions
            positions = self.trader.api.list_positions()
            position_symbols = [p.symbol for p in positions]
            
            # Calculate cash available for trading
            cash_available = cash * self.cash_allocation
            
            # Sort symbols by signal probability
            buy_candidates = []
            for symbol, signal_data in signals.items():
                if signal_data['signal'] == 'buy' and symbol not in position_symbols:
                    buy_candidates.append((symbol, signal_data['probability']))
            
            buy_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Execute sell orders first
            for symbol, signal_data in signals.items():
                if signal_data['signal'] == 'sell' and symbol in position_symbols:
                    # Get position
                    position = next((p for p in positions if p.symbol == symbol), None)
                    
                    if position:
                        # Sell all shares
                        qty = int(position.qty)
                        
                        if qty > 0:
                            logger.info(f"Selling {qty} shares of {symbol}")
                            
                            # Execute sell order
                            order = self.trader.api.submit_order(
                                symbol=symbol,
                                qty=qty,
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )
                            
                            # Record trade
                            executed_trades.append({
                                'symbol': symbol,
                                'action': 'sell',
                                'qty': qty,
                                'price': signal_data['price'],
                                'timestamp': datetime.now().isoformat(),
                                'order_id': order.id if order else None
                            })
                            
                            # Send notification
                            self.notifier.send_notification(
                                f"SELL: {qty} shares of {symbol} at ${signal_data['price']:.2f}"
                            )
            
            # Execute buy orders
            positions_count = len(position_symbols)
            for symbol, probability in buy_candidates:
                # Check if we've reached the maximum number of positions
                if positions_count >= self.max_positions:
                    logger.info(f"Maximum positions reached ({self.max_positions})")
                    break
                
                # Calculate position size
                position_size = cash_available / (self.max_positions - positions_count)
                
                # Calculate number of shares to buy
                price = signals[symbol]['price']
                qty = int(position_size / price)
                
                if qty > 0:
                    logger.info(f"Buying {qty} shares of {symbol}")
                    
                    # Execute buy order
                    order = self.trader.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    # Record trade
                    executed_trades.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'qty': qty,
                        'price': price,
                        'timestamp': datetime.now().isoformat(),
                        'order_id': order.id if order else None
                    })
                    
                    # Send notification
                    self.notifier.send_notification(
                        f"BUY: {qty} shares of {symbol} at ${price:.2f}"
                    )
                    
                    # Increment positions count
                    positions_count += 1
                    
                    # Reduce cash available
                    cash_available -= qty * price
        
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
        
        return executed_trades
    
    def run_once(self):
        """
        Run the bot once.
        
        Returns:
            list: List of executed trades
        """
        logger.info("Running bot...")
        
        # Get market data
        market_data = self.get_market_data()
        
        if not market_data:
            logger.error("No market data available")
            return []
        
        # Generate signals
        signals = self.generate_signals(market_data)
        
        if not signals:
            logger.error("No signals generated")
            return []
        
        # Execute trades
        executed_trades = self.execute_trades(signals)
        
        # Log summary
        logger.info(f"Executed {len(executed_trades)} trades")
        
        return executed_trades
    
    def run_continuously(self, interval_minutes=15):
        """
        Run the bot continuously.
        
        Args:
            interval_minutes (int): Interval between runs in minutes
        """
        logger.info(f"Running bot continuously with {interval_minutes} minute intervals")
        
        while True:
            try:
                # Check if market is open
                clock = self.trader.api.get_clock()
                
                # Log market status for debugging
                logger.info(f"Market status check: is_open={clock.is_open}, next_open={clock.next_open}, next_close={clock.next_close}")
                
                if clock.is_open or self.ignore_market_hours:
                    if not clock.is_open:
                        logger.info("Market is closed, but running anyway due to ignore_market_hours=True")
                    else:
                        logger.info("Market is open")
                    
                    # Run the bot
                    executed_trades = self.run_once()
                    
                    # Log trades
                    for trade in executed_trades:
                        logger.info(f"Executed trade: {trade}")
                else:
                    logger.info("Market is closed, skipping bot run")
                
                # Sleep until next run
                next_run = datetime.now() + timedelta(minutes=interval_minutes)
                logger.info(f"Next run at {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                
                time.sleep(interval_minutes * 60)
            
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            
            except Exception as e:
                logger.error(f"Error in bot loop: {e}")
                time.sleep(60)  # Sleep for 1 minute before retrying

def main():
    """Main function to run the bot."""
    parser = argparse.ArgumentParser(description='Run the optimized ensemble ML trading bot')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA'],
                        help='List of stock symbols to trade')
    parser.add_argument('--cash-allocation', type=float, default=0.9,
                        help='Percentage of cash to allocate for trading')
    parser.add_argument('--max-positions', type=int, default=5,
                        help='Maximum number of positions to hold')
    parser.add_argument('--paper', action='store_true', default=True,
                        help='Use paper trading')
    parser.add_argument('--interval', type=int, default=15,
                        help='Interval between runs in minutes')
    parser.add_argument('--once', action='store_true',
                        help='Run the bot once and exit')
    parser.add_argument('--ignore-market-hours', action='store_true',
                        help='Run the bot even when the market is closed (for backtesting/optimization)')
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Initialize bot
    bot = OptimizedEnsembleBot(
        symbols=args.symbols,
        cash_allocation=args.cash_allocation,
        max_positions=args.max_positions,
        paper=args.paper,
        ignore_market_hours=args.ignore_market_hours
    )
    
    # Run the bot
    if args.once:
        bot.run_once()
    else:
        bot.run_continuously(interval_minutes=args.interval)

if __name__ == "__main__":
    main()