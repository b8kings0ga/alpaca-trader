#!/bin/bash
# Script to update Docker containers for the Alpaca Trading System

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}   Updating Alpaca Trading System        ${NC}"
echo -e "${GREEN}=========================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running!${NC}"
    echo -e "${YELLOW}Please start Docker and try again.${NC}"
    exit 1
fi

# Pull latest code if in a git repository
if [ -d ".git" ]; then
    echo -e "${YELLOW}Pulling latest code from repository...${NC}"
    git pull
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Warning: Failed to pull latest code.${NC}"
        echo -e "${YELLOW}Continuing with local code...${NC}"
    else
        echo -e "${GREEN}Successfully pulled latest code.${NC}"
    fi
fi

# Stop running containers
echo -e "${YELLOW}Stopping running containers...${NC}"
docker-compose down

# Remove old images
echo -e "${YELLOW}Removing old images...${NC}"
docker-compose rm -f

# Pull latest base images
echo -e "${YELLOW}Pulling latest base images...${NC}"
docker pull python:3.11-slim

# Build new images
echo -e "${YELLOW}Building new images...${NC}"
echo -e "${YELLOW}Building YFinance DB image...${NC}"
docker-compose build --no-cache yfinance-db
echo -e "${YELLOW}Building Alpaca Trader image...${NC}"
docker-compose build --no-cache alpaca-trader
echo -e "${YELLOW}Building Optimized Bot image...${NC}"
docker-compose build --no-cache optimized-bot
echo -e "${YELLOW}Building Dashboard image...${NC}"
docker-compose build --no-cache dashboard

# Check if models need to be updated
echo -e "${YELLOW}Checking if models need to be updated...${NC}"
if [ ! -f "models/random_forest_optimized.joblib" ] || [ ! -f "models/gradient_boosting_optimized.joblib" ] || [ ! -f "models/ensemble_weights_optimized.txt" ]; then
    echo -e "${YELLOW}Models need to be updated. Running model optimization...${NC}"
    python optimize_ml_models_simple.py
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Model optimization failed!${NC}"
        echo -e "${YELLOW}Please run 'python optimize_ml_models_simple.py' manually to troubleshoot.${NC}"
    else
        echo -e "${GREEN}Model optimization completed successfully.${NC}"
    fi
else
    # Check if models are older than 7 days
    MODELS_AGE=$(find models/random_forest_optimized.joblib -mtime +7 2>/dev/null)
    if [ ! -z "$MODELS_AGE" ]; then
        echo -e "${YELLOW}Models are older than 7 days. Consider re-optimizing them.${NC}"
        read -p "Do you want to re-optimize the models? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Running model optimization...${NC}"
            python optimize_ml_models_simple.py
            
            if [ $? -ne 0 ]; then
                echo -e "${RED}Error: Model optimization failed!${NC}"
                echo -e "${YELLOW}Please run 'python optimize_ml_models_simple.py' manually to troubleshoot.${NC}"
            else
                echo -e "${GREEN}Model optimization completed successfully.${NC}"
            fi
        fi
    else
        echo -e "${GREEN}Models are up to date.${NC}"
    fi
fi

# Start the services
echo -e "${YELLOW}Starting updated services...${NC}"
docker-compose up -d

# Check if services are running
echo -e "${YELLOW}Checking service status...${NC}"
docker-compose ps

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}   Alpaca Trading System Updated         ${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "${YELLOW}Dashboard URL: http://localhost:8501${NC}"
echo -e "${YELLOW}YFinance DB API: http://localhost:8001${NC}"
echo -e "${YELLOW}To view logs: docker-compose logs -f${NC}"
echo -e "${YELLOW}To stop the system: docker-compose down${NC}"