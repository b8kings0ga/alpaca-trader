"""
Streamlit application entry point for the Alpaca Trading Bot Dashboard.
"""
import streamlit as st
from dashboard import Dashboard
from bot import setup_logging

def main():
    """
    Main entry point for the Streamlit dashboard.
    """
    # Set up logging
    setup_logging()
    
    # Initialize and run the dashboard
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()