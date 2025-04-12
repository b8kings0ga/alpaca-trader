"""
Base strategy class for the Alpaca Trading Bot.
"""
from src.logger import get_logger

logger = get_logger()

class Strategy:
    """
    Base strategy class that all strategies should inherit from.
    """
    def __init__(self, name):
        """
        Initialize the strategy.
        
        Args:
            name (str): Strategy name
        """
        self.name = name
        logger.info(f"Strategy '{name}' initialized")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the strategy.
        
        Args:
            data (dict): Dictionary of DataFrames with market data
            
        Returns:
            dict: Dictionary of signals for each symbol
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")