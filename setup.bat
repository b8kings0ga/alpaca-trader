@echo off
:: Setup script for local development with uv on Windows

echo Checking if uv is installed...
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo uv is not installed. Installing uv...
    powershell -Command "Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -OutFile install.ps1; .\install.ps1"
    
    echo uv installed successfully!
) else (
    echo uv is already installed.
)

:: Create virtual environment and install dependencies
echo Creating virtual environment and installing dependencies...
uv venv
call .venv\Scripts\activate.bat
uv pip install -e .

:: Create logs directory
if not exist logs mkdir logs

:: Copy .env.example to .env if .env doesn't exist
if not exist .env (
    echo Creating .env file from .env.example...
    copy .env.example .env
    echo Please edit .env file with your Alpaca API credentials.
)

echo.
echo Setup complete! You can now run the bot with:
echo call .venv\Scripts\activate.bat  :: If not already activated
echo python main.py
echo.
echo For Docker deployment:
echo docker-compose up -d