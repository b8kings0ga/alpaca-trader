#!/bin/bash
# Setup script for local development with uv

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    echo "uv installed successfully!"
else
    echo "uv is already installed."
fi

# Create virtual environment and install dependencies
echo "Creating virtual environment and installing dependencies..."
uv venv
source .venv/bin/activate
uv pip install -e .

# Create logs directory
mkdir -p logs

# Copy .env.example to .env if .env doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please edit .env file with your Alpaca API credentials."
fi

echo ""
echo "Running setup verification..."
python test_setup.py

echo ""
echo "Setup complete! You can now run the bot with:"
echo "source .venv/bin/activate  # If not already activated"
echo "python main.py"
echo ""
echo "For Docker deployment:"
echo "docker-compose up -d"