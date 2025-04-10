"""
Configuration settings for the Alpaca Trading Bot.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Determine if running in Docker
IN_DOCKER = os.environ.get('PYTHONPATH', '').startswith('/app')

# Alpaca API credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
ALPACA_DATA_URL = os.getenv('ALPACA_DATA_URL', 'https://data.alpaca.markets')

# Trading parameters
SYMBOLS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']  # Stocks to trade
MAX_POSITIONS = 5  # Maximum number of positions to hold
POSITION_SIZE = 0.2  # Percentage of portfolio per position

# Strategy parameters - Moving Average Crossover
SHORT_WINDOW = 20  # Short moving average window
LONG_WINDOW = 50  # Long moving average window
RSI_PERIOD = 14  # RSI period
RSI_OVERSOLD = 30  # RSI oversold threshold
RSI_OVERBOUGHT = 70  # RSI overbought threshold

# ML Strategy parameters (placeholder for future implementation)
ML_STRATEGY_TYPE = 'ensemble'  # Options: 'ensemble', 'neural_network', 'reinforcement', 'nlp'
ML_LOOKBACK_PERIOD = 60  # Number of days of historical data to use for ML features
ML_FEATURES = [  # Features to use for ML model
    'sma_short', 'sma_long', 'rsi',
    'volume', 'open', 'high', 'low', 'close'
]
ML_TRAIN_TEST_SPLIT = 0.8  # Percentage of data to use for training
ML_RETRAIN_FREQUENCY = 7  # Days between model retraining
ML_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence score to execute a trade

# Schedule settings
MARKET_OPEN_TIME = '09:30'  # Market open time (Eastern Time)
MARKET_CLOSE_TIME = '16:00'  # Market close time (Eastern Time)
TIMEZONE = 'America/New_York'  # Timezone for scheduling
RUN_FREQUENCY = 'minutes'  # Options: 'daily', 'hourly', 'minutes'
RUN_INTERVAL = 15  # Run every X minutes (when RUN_FREQUENCY is 'minutes')
RUN_TIME = '09:30'  # Time to run the bot (when RUN_FREQUENCY is 'daily')

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FILE = '/app/logs/alpaca_trader.log' if IN_DOCKER else 'logs/alpaca_trader.log'

# Notification settings (optional)
ENABLE_NOTIFICATIONS = False
NOTIFICATION_TYPE = 'email'  # email, telegram, webhook
NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL', '')
SMTP_SERVER = os.getenv('SMTP_SERVER', '')
SMTP_PORT = os.getenv('SMTP_PORT', 587)
SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')