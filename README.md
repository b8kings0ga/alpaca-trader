# Alpaca Trader

An automated trading bot built with Python and Alpaca SDK that runs in a Docker container.

## Features

- Implements a simple trading strategy (Moving Average Crossover)
- Connects to Alpaca's Paper Trading API
- Automatically fetches market data and executes trades
- Logs trades and portfolio status
- Runs on a configurable schedule (every 15 minutes by default)
- Containerized with Docker for easy deployment and fast builds
- Uses uv for fast Python package management
- Includes placeholder for ML-based strategy upgrades

## Future Enhancements

The trading bot includes a placeholder for future machine learning-based strategy implementations. Potential ML approaches include:

- Supervised learning models (Random Forest, SVM, Neural Networks)
- Reinforcement learning for dynamic strategy optimization
- Deep learning for pattern recognition in price data
- Natural Language Processing for sentiment analysis of news/social media
- Ensemble methods combining multiple ML models

### ML Strategy Implementation

The project includes a framework for implementing ML-based trading strategies:

- `src/ml_models.py`: Contains placeholder classes for different types of ML models
- `models/`: Directory for storing trained ML models
- `train_ml_model.py`: Script for training and saving ML models

To train a new ML model:

```bash
# Train an ensemble model on the default symbols
./train_ml_model.py --model-type ensemble

# Train a supervised model on specific symbols with 2 years of data
./train_ml_model.py --model-type supervised --symbols AAPL,MSFT,GOOGL --days 730
```

To use an ML-based strategy, update the strategy name in `config/config.py` or set it when initializing the bot:

```python
from src.bot import AlpacaBot

# Initialize the bot with the ML strategy
bot = AlpacaBot(strategy_name='ml')
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Alpaca API credentials

### Configuration

Create a `.env` file in the root directory with your Alpaca API credentials:

```
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Running the Bot

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f
```

## Configuration Options

Edit the `config/config.py` file to customize:

- Trading strategy parameters
- Stocks to trade
- Schedule settings (frequency, interval)
- Notification settings

## Trading Frequency

The bot is configured to trade every 15 minutes during market hours by default. You can adjust this in the `.env` file:

```
# Trading frequency settings
RUN_FREQUENCY=minutes  # Options: daily, hourly, minutes
RUN_INTERVAL=15        # Run every X minutes when using 'minutes' frequency
```

For daily trading, use:
```
RUN_FREQUENCY=daily
RUN_TIME=09:30         # Time to run (market open)
```

## Local Development

For local development without Docker, you can use the provided setup script:

```bash
# On Linux/macOS:
./setup.sh

# On Windows:
setup.bat

# Run the bot
python main.py
```

The setup script will:
1. Install uv if it's not already installed
2. Create a virtual environment and install dependencies
3. Create a logs directory
4. Copy .env.example to .env if it doesn't exist

After running the setup script, you can verify your installation:

```bash
python test_setup.py
```

This will check that all dependencies are installed correctly and that your environment is properly configured.

Alternatively, you can manually set up your environment:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# Run the bot
python main.py
```

## Project Structure

```
alpaca-trader/
├── config/
│   ├── config.py         # Configuration settings
├── src/
│   ├── bot.py            # Main bot logic
│   ├── strategies.py     # Trading strategies
│   ├── data.py           # Market data handling
│   ├── trader.py         # Order execution
│   ├── logger.py         # Logging functionality
│   ├── scheduler.py      # Scheduling functionality
│   └── notifications.py  # Optional notification system
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── pyproject.toml        # Python project metadata and dependencies
├── requirements.txt      # Python dependencies (legacy format)
├── .env.example          # Example environment variables
├── main.py               # Entry point script
└── README.md             # This file
```

## License

MIT