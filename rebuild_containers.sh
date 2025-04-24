#!/bin/bash

echo "Rebuilding Docker containers with TA-Lib support..."

# Stop any running containers
docker-compose down

# Rebuild the containers
docker-compose build --no-cache

# Start the containers
docker-compose up -d

echo "Containers rebuilt and started. Check logs for any errors."