#!/bin/bash

# Script to update Docker containers with the latest code changes

echo "Updating Docker containers..."

# Copy the updated files to the Docker container
echo "Copying updated trader.py to container..."
docker cp src/trader.py alpaca-trader:/app/src/trader.py

echo "Copying dashboard_config.py to container..."
docker cp src/dashboard_config.py alpaca-trader:/app/src/dashboard_config.py

echo "Copying updated dashboard.py to container..."
docker cp src/dashboard.py alpaca-trader:/app/src/dashboard.py

echo "Copying updated run_optimized_bot.py to container..."
docker cp run_optimized_bot.py alpaca-optimized-bot:/app/run_optimized_bot.py

echo "Copying updated run_optimized_bot_market_hours.py to container..."
docker cp run_optimized_bot_market_hours.py alpaca-optimized-bot:/app/run_optimized_bot_market_hours.py

# Restart the containers
echo "Restarting containers..."
docker-compose restart

echo "Waiting for services to start..."
sleep 5

# Check if the containers are running
echo "Checking container status..."
docker-compose ps

echo "Update complete! The trading system should now process sell signals first and preserve buy signals when there's insufficient buying power."
echo "The dashboard now uses dynamic configuration values from dashboard_config.py."
echo "The optimized bot can now run during closed market hours for backtesting and optimization."