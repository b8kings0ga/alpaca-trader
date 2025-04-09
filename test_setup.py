#!/usr/bin/env python3
"""
Test script to verify that the Alpaca Trading Bot is set up correctly.
This script checks for required dependencies and configuration.
"""
import os
import sys
import importlib.util
import traceback

def check_module(module_name):
    """Check if a module is installed."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    """Run tests to verify the setup."""
    print("Testing Alpaca Trading Bot setup...")
    
    # Check for required modules
    required_modules = [
        "alpaca_trade_api",
        "pandas",
        "numpy",
        "dotenv",
        "apscheduler",
        "pytz",
        "loguru",
        "matplotlib"
    ]
    
    missing_modules = []
    for module in required_modules:
        if not check_module(module):
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required modules: {', '.join(missing_modules)}")
        print("Please run the setup script or install dependencies manually.")
        return False
    else:
        print("✅ All required modules are installed.")
    
    # Check for .env file
    if not os.path.exists(".env"):
        print("❌ .env file not found.")
        print("Please create a .env file with your Alpaca API credentials.")
        return False
    else:
        print("✅ .env file exists.")
    
    # Check for logs directory
    if not os.path.exists("logs"):
        print("❌ logs directory not found.")
        print("Creating logs directory...")
        os.makedirs("logs", exist_ok=True)
    
    print("✅ logs directory exists.")
    
    # Try to import the bot module
    try:
        from src.bot import AlpacaBot
        print("✅ Bot module imported successfully.")
    except Exception as e:
        print(f"❌ Error importing bot module: {e}")
        traceback.print_exc()
        return False
    
    print("\nSetup verification complete!")
    print("You can now run the bot with: python main.py")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)