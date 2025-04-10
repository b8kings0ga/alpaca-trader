"""
Order execution for the Alpaca Trading Bot.
"""
import alpaca_trade_api as tradeapi
from config import config
from src.logger import get_logger

logger = get_logger()

class Trader:
    """
    Class for executing trades with Alpaca.
    """
    def __init__(self):
        """
        Initialize the Trader class with Alpaca API.
        """
        # Log the API URLs for debugging
        logger.info(f"Initializing Trader with base URL: {config.ALPACA_BASE_URL}")
        
        # Remove trailing /v2 from base URL if present to avoid duplicate version in path
        base_url = config.ALPACA_BASE_URL
        if base_url.endswith('/v2'):
            base_url = base_url[:-3]
            logger.info(f"Adjusted base URL to avoid duplicate version: {base_url}")
        
        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_API_SECRET,
            base_url,
            api_version='v2'
        )
        logger.info("Trader initialized")
        
    def execute_signals(self, signals, account_info, force_initial_positions=False):
        """
        Execute trades based on signals.
        
        Args:
            signals (dict): Dictionary of signals for each symbol
            account_info (dict): Account information
            
        Returns:
            list: List of executed orders
        """
        executed_orders = []
        
        # If force_initial_positions is True and we have no positions, create initial positions
        current_positions = {p.symbol: p for p in self.api.list_positions()}
        
        # Get portfolio value from account info
        portfolio_value = float(account_info.get('portfolio_value', 100000.0))
        
        if force_initial_positions and len(current_positions) == 0:
            logger.info("No current positions found. Creating initial positions based on current signals.")
            logger.info(f"Signals available: {list(signals.keys())}")
            
            # Filter for buy signals or create balanced portfolio if no buy signals
            buy_signals = {symbol: data for symbol, data in signals.items()
                          if data['action'] == 'buy' or data['signal'] > 0}
            
            logger.info(f"Buy signals found: {list(buy_signals.keys())}")
            
            # If no buy signals, use all symbols with a balanced approach
            if not buy_signals:
                logger.info("No buy signals found. Creating balanced initial positions.")
                # Use all symbols with equal weight
                symbols_to_buy = list(signals.keys())
                
                logger.info(f"Symbols to buy (all signals): {symbols_to_buy}")
                
                # Check if symbols_to_buy is empty to avoid division by zero
                if not symbols_to_buy:
                    logger.warning("No symbols available for initial positions")
                    logger.warning(f"Signals dictionary: {signals}")
                    logger.warning("This is likely due to insufficient data for signal generation")
                    return executed_orders
                    
                # Use default symbols if none are available in signals
                if len(symbols_to_buy) == 0:
                    logger.info("Using default symbols for initial positions")
                    symbols_to_buy = config.SYMBOLS
                    
                # Log the symbols we're going to buy
                logger.info(f"Creating initial positions for: {symbols_to_buy}")
                
                # Calculate position size per symbol
                position_size_per_symbol = portfolio_value * config.POSITION_SIZE / len(symbols_to_buy)
                
                for symbol in symbols_to_buy:
                    if symbol in current_positions:
                        logger.info(f"Already have position in {symbol}, skipping")
                        continue
                        
                    # Check if price exists and is valid
                    if 'price' not in signals[symbol] or signals[symbol]['price'] <= 0:
                        logger.warning(f"Invalid or missing price for {symbol}")
                        # Use a default price based on historical data or a fixed value
                        try:
                            # Try to get the latest price from Alpaca
                            latest_bar = self.api.get_latest_bar(symbol)
                            price = latest_bar.c  # Close price
                            logger.info(f"Using latest price from Alpaca for {symbol}: ${price:.2f}")
                        except Exception as e:
                            logger.error(f"Error getting latest price for {symbol}: {e}")
                            # Use a default price as fallback
                            price = 100.0  # Default price
                            logger.info(f"Using default price for {symbol}: ${price:.2f}")
                    else:
                        price = signals[symbol]['price']
                    
                    # Calculate quantity based on position size and price
                    qty = int(position_size_per_symbol / price)
                    
                    # Ensure minimum quantity
                    if qty <= 0:
                        qty = 1  # Minimum quantity
                        logger.warning(f"Adjusted quantity for {symbol} to minimum: {qty}")
                    
                    logger.info(f"Calculated quantity for {symbol}: {qty} shares at ${price:.2f}")
                        
                    try:
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        logger.info(f"Initial position: Buy order placed for {qty} shares of {symbol} at ~${price:.2f}")
                        executed_orders.append(self._format_order(order))
                    except Exception as e:
                        logger.error(f"Error creating initial position for {symbol}: {e}")
            else:
                # Use buy signals
                # Check if buy_signals is empty to avoid division by zero
                if not buy_signals:
                    logger.warning("No buy signals available for initial positions")
                    return executed_orders
                    
                # Log the symbols we're going to buy
                logger.info(f"Creating initial positions based on buy signals for: {list(buy_signals.keys())}")
                
                # Calculate position size per symbol
                position_size_per_symbol = portfolio_value * config.POSITION_SIZE / len(buy_signals)
                
                for symbol, signal_data in buy_signals.items():
                    if symbol in current_positions:
                        continue
                        
                    price = signal_data['price']
                    qty = int(position_size_per_symbol / price)
                    
                    if qty <= 0 or price <= 0:
                        logger.warning(f"Invalid quantity ({qty}) or price ({price}) for {symbol}")
                        continue
                        
                    try:
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        logger.info(f"Initial position: Buy order placed for {qty} shares of {symbol} at ~${price:.2f}")
                        executed_orders.append(self._format_order(order))
                    except Exception as e:
                        logger.error(f"Error creating initial position for {symbol}: {e}")
        
        # Check if we have enough buying power
        buying_power = float(account_info.get('buying_power', 0))
        if buying_power <= 0:
            logger.warning(f"Not enough buying power: {buying_power}")
            return executed_orders
            
        # Calculate position size
        portfolio_value = float(account_info.get('portfolio_value', 0))
        position_value = portfolio_value * config.POSITION_SIZE
        
        # Get current positions (refresh after potential initial positions)
        if force_initial_positions and executed_orders:
            # Wait a moment for orders to process
            import time
            time.sleep(2)
            
        # Refresh positions
        current_positions = {p.symbol: p for p in self.api.list_positions()}
        
        # Process signals
        for symbol, signal_data in signals.items():
            action = signal_data['action']
            price = signal_data['price']
            signal_changed = signal_data.get('signal_changed', False)
            
            # Skip if no action or signal hasn't changed
            if action == 'hold' or not signal_changed:
                continue
                
            try:
                if action == 'buy':
                    # Skip if we already have a position
                    if symbol in current_positions:
                        logger.info(f"Already have position in {symbol}, skipping buy")
                        continue
                        
                    # Calculate quantity based on position size and current price
                    qty = int(position_value / price)
                    
                    # Skip if quantity is too small
                    if qty <= 0:
                        logger.warning(f"Calculated quantity for {symbol} is {qty}, skipping buy")
                        continue
                        
                    # Place buy order
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    logger.info(f"Buy order placed for {qty} shares of {symbol} at ~${price:.2f}")
                    executed_orders.append(self._format_order(order))
                    
                elif action == 'sell':
                    # Skip if we don't have a position
                    if symbol not in current_positions:
                        logger.info(f"No position in {symbol}, skipping sell")
                        continue
                        
                    # Get position quantity
                    position = current_positions[symbol]
                    qty = abs(int(position.qty))
                    
                    # Place sell order
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    
                    logger.info(f"Sell order placed for {qty} shares of {symbol} at ~${price:.2f}")
                    executed_orders.append(self._format_order(order))
            
            except Exception as e:
                logger.error(f"Error executing {action} order for {symbol}: {e}")
                
        return executed_orders
        
    def _format_order(self, order):
        """
        Format order information.
        
        Args:
            order: Alpaca order object
            
        Returns:
            dict: Formatted order information
        """
        return {
            'id': order.id,
            'symbol': order.symbol,
            'qty': order.qty,
            'side': order.side,
            'type': order.type,
            'status': order.status,
            'created_at': order.created_at
        }
        
    def cancel_all_orders(self):
        """
        Cancel all open orders.
        
        Returns:
            int: Number of orders canceled
        """
        try:
            canceled = self.api.cancel_all_orders()
            logger.info(f"Canceled {len(canceled)} orders")
            return len(canceled)
        except Exception as e:
            logger.error(f"Error canceling orders: {e}")
            return 0
            
    def liquidate_all_positions(self):
        """
        Liquidate all positions.
        
        Returns:
            int: Number of positions liquidated
        """
        try:
            positions = self.api.list_positions()
            for position in positions:
                self.api.submit_order(
                    symbol=position.symbol,
                    qty=position.qty,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"Liquidated position in {position.symbol}")
            
            return len(positions)
        except Exception as e:
            logger.error(f"Error liquidating positions: {e}")
            return 0