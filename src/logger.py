"""
Logging functionality for the Alpaca Trading Bot.
"""
import os
import sys
from loguru import logger
from config import config

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

# Configure logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level=config.LOG_LEVEL)  # Add stderr handler
logger.add(
    config.LOG_FILE,
    rotation="10 MB",
    retention="1 week",
    level=config.LOG_LEVEL,
    backtrace=True,
    diagnose=True,
)

# Create a function to get the logger
def get_logger():
    """
    Returns the configured logger instance.
    """
    return logger