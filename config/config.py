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
SYMBOLS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA']  # Stocks to trade
MAX_POSITIONS = 5  # Maximum number of positions to hold
POSITION_SIZE = 0.2  # Percentage of portfolio per position

# Strategy parameters - Moving Average Crossover
SHORT_WINDOW = 3  # Short moving average window (reduced from 5 for even faster response)
LONG_WINDOW = 10  # Long moving average window (reduced from 20 for faster response)
RSI_PERIOD = 3  # RSI period (reduced from 5 for faster response)
RSI_OVERSOLD = 35  # RSI oversold threshold (decreased from 40 for more buy signals)
RSI_OVERBOUGHT = 65  # RSI overbought threshold (increased from 60 for more sell signals)

# ML Strategy parameters
ML_STRATEGY_TYPE = 'gradient_boosting_ensemble'  # Options: 'gradient_boosting', 'random_forest', 'ensemble', 'neural_network', 'reinforcement', 'nlp'
ML_LOOKBACK_PERIOD = 20  # Number of days of historical data to use for ML features (reduced from 60 to match available data)
ML_FEATURES = [  # Features to use for ML model
    'sma_short', 'sma_long', 'rsi',
    'volume', 'open', 'high', 'low', 'close'
]
ML_TRAIN_TEST_SPLIT = 0.8  # Percentage of data to use for training
ML_RETRAIN_FREQUENCY = 7  # Days between model retraining
ML_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence score to execute a trade

# Gradient Boosting parameters
GB_LEARNING_RATE = 0.1
GB_MAX_DEPTH = 5
GB_N_ESTIMATORS = 100
GB_SUBSAMPLE = 0.8
GB_COLSAMPLE_BYTREE = 0.8
GB_RANDOM_STATE = 42

# Random Forest parameters
RF_MAX_DEPTH = 10
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42

# Ensemble parameters
ENSEMBLE_WEIGHTS = {
    'gradient_boosting': 0.7,  # Increased from 0.6
    'random_forest': 0.3
    # Removed 'linear' model weight as it's not being used
}
ENSEMBLE_VOTING = 'soft'  # Options: 'hard', 'soft'

# Feature Engineering parameters
TECHNICAL_INDICATORS = [
    'rsi',           # Relative Strength Index
    'macd',          # Moving Average Convergence Divergence
    'bollinger',     # Bollinger Bands
    'atr',           # Average True Range
    'adx',           # Average Directional Index
    'obv',           # On-Balance Volume
    'stoch',         # Stochastic Oscillator
    'williams_r',    # Williams %R
    'cci',           # Commodity Channel Index
    'mfi'            # Money Flow Index
]

# Schedule settings
MARKET_OPEN_TIME = '09:30'  # Market open time (Eastern Time)
MARKET_CLOSE_TIME = '16:00'  # Market close time (Eastern Time)
TIMEZONE = 'America/New_York'  # Timezone for scheduling
RUN_FREQUENCY = 'minutes'  # Options: 'daily', 'hourly', 'minutes'
# Increased from 0.5 (30 seconds) to 2 minutes to prevent overlapping runs
# This is a temporary change for debugging - adjust based on actual run time
RUN_INTERVAL = 2  # Run every 2 minutes (when RUN_FREQUENCY is 'minutes')
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

# Position settings
USE_REAL_POSITIONS = os.getenv('USE_REAL_POSITIONS', 'True').lower() == 'true'  # Whether to use real positions from Alpaca API