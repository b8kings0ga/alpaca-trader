#!/bin/bash
# Script to start the entire trading system using Docker Compose

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}   Starting Alpaca Trading System        ${NC}"
echo -e "${GREEN}=========================================${NC}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo -e "${YELLOW}Creating a sample .env file from .env.example...${NC}"
    
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}Created .env file from .env.example${NC}"
        echo -e "${YELLOW}Please edit the .env file with your Alpaca API credentials before continuing.${NC}"
        exit 1
    else
        echo -e "${RED}Error: .env.example file not found!${NC}"
        echo -e "${YELLOW}Creating a basic .env file...${NC}"
        
        cat > .env << EOL
# Alpaca API credentials
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets

# YFinance DB settings
MAX_DB_SIZE_GB=10
FETCH_INTERVAL_MINUTES=60
SYMBOLS=AAPL,MSFT,AMZN,GOOGL,META,TSLA,NVDA

# Notification settings
ENABLE_NOTIFICATIONS=false
NOTIFICATION_TYPE=email
NOTIFICATION_EMAIL=your_email@example.com
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your_username
SMTP_PASSWORD=your_password
EOL
        
        echo -e "${GREEN}Created basic .env file${NC}"
        echo -e "${YELLOW}Please edit the .env file with your Alpaca API credentials before continuing.${NC}"
        exit 1
    fi
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo -e "${YELLOW}Creating models directory...${NC}"
    mkdir -p models
fi

# Check if logs directory exists
if [ ! -d "logs" ]; then
    echo -e "${YELLOW}Creating logs directory...${NC}"
    mkdir -p logs
fi

# Check if data directory exists
if [ ! -d "data" ]; then
    echo -e "${YELLOW}Creating data directory...${NC}"
    mkdir -p data
fi

# Check if optimized models exist
if [ ! -f "models/random_forest_optimized.joblib" ] || [ ! -f "models/gradient_boosting_optimized.joblib" ]; then
    echo -e "${YELLOW}Optimized models not found. Running model optimization...${NC}"
    python optimize_ml_models_simple.py
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Model optimization failed!${NC}"
        echo -e "${YELLOW}Please run 'python optimize_ml_models_simple.py' manually to troubleshoot.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Model optimization completed successfully.${NC}"
fi

# Pull latest Docker images
echo -e "${YELLOW}Pulling latest Docker images...${NC}"
docker-compose pull

# Build Docker images
echo -e "${YELLOW}Building Docker images...${NC}"
echo -e "${YELLOW}Building YFinance DB image...${NC}"
docker-compose build yfinance-db
echo -e "${YELLOW}Building Alpaca Trader image...${NC}"
docker-compose build alpaca-trader
echo -e "${YELLOW}Building Optimized Bot image...${NC}"
docker-compose build optimized-bot
echo -e "${YELLOW}Building Dashboard image...${NC}"
docker-compose build dashboard

# Start the services
echo -e "${YELLOW}Starting services...${NC}"
docker-compose up -d

# Check if services are running
echo -e "${YELLOW}Checking service status...${NC}"
docker-compose ps

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}   Alpaca Trading System Started         ${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "${YELLOW}Dashboard URL: http://localhost:8501${NC}"
echo -e "${YELLOW}YFinance DB API: http://localhost:8001${NC}"
echo -e "${YELLOW}To view logs: docker-compose logs -f${NC}"
echo -e "${YELLOW}To stop the system: docker-compose down${NC}"