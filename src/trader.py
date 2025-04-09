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
        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_API_SECRET,
            config.ALPACA_BASE_URL,
            api_version='v2'
        )
        logger.info("Trader initialized")
        
    def execute_signals(self, signals, account_info):
        """
        Execute trades based on signals.
        
        Args:
            signals (dict): Dictionary of signals for each symbol
            account_info (dict): Account information
            
        Returns:
            list: List of executed orders
        """
        executed_orders = []
        
        # Check if we have enough buying power
        buying_power = float(account_info.get('buying_power', 0))
        if buying_power <= 0:
            logger.warning(f"Not enough buying power: {buying_power}")
            return executed_orders
            
        # Calculate position size
        portfolio_value = float(account_info.get('portfolio_value', 0))
        position_value = portfolio_value * config.POSITION_SIZE
        
        # Get current positions
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