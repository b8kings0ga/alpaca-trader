"""
Streamlit dashboard for the Alpaca Trading Bot.
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import requests
from config import config
from src.bot import AlpacaBot
from src.data import MarketData
from src.strategies import get_strategy
from src.logger import get_logger

logger = get_logger()

class Dashboard:
    """
    Streamlit dashboard for the Alpaca Trading Bot.
    """
    def __init__(self):
        """
        Initialize the dashboard.
        """
        self.market_data = MarketData()
        
        # Set up a background thread to periodically check the connection
        self.connection_check_interval = 60  # seconds
        self._setup_connection_check_thread()
        self.strategies = {
            'moving_average_crossover': 'Moving Average Crossover',
            'rsi': 'RSI Strategy',
            'ml': 'ML Strategy',
            'dual_ma_yf': 'Dual Moving Average (YFinance)'
        }
        self.current_strategy = None
        self.data = {}
        # YFinance DB service URL
        self.yfinance_db_host = os.getenv("YFINANCE_DB_HOST", "localhost")
        self.yfinance_db_port = int(os.getenv("YFINANCE_DB_PORT", "8001"))
        self.yfinance_db_url = f"http://{self.yfinance_db_host}:{self.yfinance_db_port}"
        
        # Initialize connection status
        self.yfinance_db_connected = False
        self.check_yfinance_db_connection()
        
        # Initialize account and positions data
        self.account_info = None
        self.positions = None
        self.refresh_account_data()
        
        logger.info(f"Dashboard initialized with YFinance DB URL: {self.yfinance_db_url}")
        logger.info("Dashboard initialized")
        
    def get_trading_signals(self, symbol, limit=100):
        """
        Get trading signals for a symbol from the YFinance DB service.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of signals to return
            
        Returns:
            DataFrame: DataFrame with trading signals
        """
        # Check if YFinance DB service is connected
        if not self.yfinance_db_connected:
            logger.warning(f"YFinance DB service is not connected. Attempting to reconnect...")
            if not self.check_yfinance_db_connection():
                logger.error(f"Failed to connect to YFinance DB service. Cannot retrieve trading signals.")
                return pd.DataFrame()
        
        try:
            url = f"{self.yfinance_db_url}/signals/{symbol}"
            params = {"limit": limit}
            
            logger.info(f"Requesting trading signals from: {url} with params: {params}")
            logger.info(f"YFinance DB connection status: {self.yfinance_db_connected}")
            
            # Use a session for connection pooling and retry mechanism
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                max_retries=3,  # Retry up to 3 times
                pool_connections=10,
                pool_maxsize=10
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            try:
                response = session.get(url, params=params, timeout=10)
                
                logger.info(f"Signals response status code: {response.status_code}")
                logger.info(f"Response headers: {response.headers}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        logger.info(f"Received JSON data: {data[:5] if data else 'Empty'}")
                        if data:
                            df = pd.DataFrame(data)
                            # Convert timestamp strings to datetime
                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                            logger.info(f"Retrieved {len(df)} trading signals for {symbol}")
                            logger.info(f"DataFrame columns: {df.columns.tolist()}")
                            logger.info(f"First few rows: {df.head(2).to_dict('records')}")
                            return df
                        else:
                            logger.warning(f"No trading signals found for {symbol} (empty response)")
                            return pd.DataFrame()
                    except ValueError as json_e:
                        logger.error(f"Error parsing JSON response for trading signals: {json_e}")
                        logger.error(f"Response content: {response.text[:200]}...")
                        return pd.DataFrame()
                else:
                    # Handle 404 responses more gracefully
                    if response.status_code == 404:
                        logger.info(f"No trading signals found for {symbol}: {response.status_code}")
                        try:
                            error_data = response.json()
                            logger.info(f"Error message: {error_data.get('message', 'No message')}")
                        except ValueError:
                            logger.info(f"Could not parse error response as JSON")
                            logger.info(f"Response content: {response.text[:200]}...")
                    else:
                        logger.error(f"Error retrieving trading signals for {symbol}: {response.status_code}")
                        if hasattr(response, 'text'):
                            logger.error(f"Response content: {response.text[:200]}...")
                    return pd.DataFrame()
            except requests.exceptions.ConnectionError as conn_e:
                logger.error(f"Connection error retrieving trading signals for {symbol}: {conn_e}")
                logger.info("This could indicate that the yfinance-db service is not running or not accessible")
                # Mark the service as disconnected
                self.yfinance_db_connected = False
                return pd.DataFrame()
            except requests.exceptions.Timeout as timeout_e:
                logger.error(f"Timeout retrieving trading signals for {symbol}: {timeout_e}")
                logger.info("This could indicate that the yfinance-db service is running but not responding")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error connecting to YFinance DB service for signals: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
            
    def get_executed_trades(self, symbol, limit=100):
        """
        Get executed trades for a symbol from the YFinance DB service.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of trades to return
            
        Returns:
            DataFrame: DataFrame with executed trades
        """
        # Check if YFinance DB service is connected
        if not self.yfinance_db_connected:
            logger.warning(f"YFinance DB service is not connected. Attempting to reconnect...")
            if not self.check_yfinance_db_connection():
                logger.error(f"Failed to connect to YFinance DB service. Cannot retrieve executed trades.")
                return pd.DataFrame()
                
        def _display_connection_status(self):
            """
            Display the connection status to the YFinance DB service.
            """
            # Create a container for the connection status
            conn_container = st.container()
            
            with conn_container:
                # Check connection status
                if self.yfinance_db_connected:
                    st.success("✅ YFinance DB Service: Connected")
                else:
                    st.error("❌ YFinance DB Service: Disconnected")
                    st.info("Attempting to reconnect... Check logs for details.")
                    
                    # Add a button to manually retry connection
                    if st.button("Retry Connection"):
                        with st.spinner("Connecting to YFinance DB Service..."):
                            if self.check_yfinance_db_connection():
                                st.success("✅ Connection successful!")
                            else:
                                st.error("❌ Connection failed. Check logs for details.")
        def _setup_connection_check_thread(self):
            """
            Set up a background thread to periodically check the connection to the YFinance DB service.
            """
            import threading
            
            def check_connection_periodically():
                """Background thread function to check connection periodically."""
                while True:
                    try:
                        # Sleep first to allow initial connection attempt to complete
                        import time
                        time.sleep(self.connection_check_interval)
                        
                        # Check connection
                        logger.info(f"Performing periodic connection check to YFinance DB service")
                        was_connected = self.yfinance_db_connected
                        is_connected = self.check_yfinance_db_connection()
                        
                        if not was_connected and is_connected:
                            logger.info(f"YFinance DB service connection restored")
                        elif was_connected and not is_connected:
                            logger.warning(f"YFinance DB service connection lost")
                    except Exception as e:
                        logger.error(f"Error in connection check thread: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Start the background thread
            connection_thread = threading.Thread(
                target=check_connection_periodically,
                daemon=True  # Make thread a daemon so it exits when the main thread exits
            )
            connection_thread.start()
            logger.info(f"Started background thread for periodic connection checks every {self.connection_check_interval} seconds")
        try:
            url = f"{self.yfinance_db_url}/trades/{symbol}"
            params = {"limit": limit}
            
            logger.info(f"Requesting executed trades from: {url} with params: {params}")
            
            # Use a session for connection pooling and retry mechanism
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                max_retries=3,  # Retry up to 3 times
                pool_connections=10,
                pool_maxsize=10
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            try:
                response = session.get(url, params=params, timeout=10)
                
                logger.info(f"Trades response status code: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data:
                            df = pd.DataFrame(data)
                            # Convert timestamp strings to datetime
                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                            logger.info(f"Retrieved {len(df)} executed trades for {symbol}")
                            return df
                        else:
                            logger.warning(f"No executed trades found for {symbol} (empty response)")
                            return pd.DataFrame()
                    except ValueError as json_e:
                        logger.error(f"Error parsing JSON response for executed trades: {json_e}")
                        logger.error(f"Response content: {response.text[:200]}...")
                        return pd.DataFrame()
                else:
                    # Handle 404 responses more gracefully
                    if response.status_code == 404:
                        logger.info(f"No executed trades found for {symbol}: {response.status_code}")
                        try:
                            error_data = response.json()
                            logger.info(f"Error message: {error_data.get('message', 'No message')}")
                        except ValueError:
                            logger.info(f"Response content: {response.text[:200]}...")
                    else:
                        logger.error(f"Error retrieving executed trades for {symbol}: {response.status_code}")
                        if hasattr(response, 'text'):
                            logger.error(f"Response content: {response.text[:200]}...")
                    return pd.DataFrame()
            except requests.exceptions.ConnectionError as conn_e:
                logger.error(f"Connection error retrieving executed trades for {symbol}: {conn_e}")
                logger.info("This could indicate that the yfinance-db service is not running or not accessible")
                # Mark the service as disconnected
                self.yfinance_db_connected = False
                return pd.DataFrame()
            except requests.exceptions.Timeout as timeout_e:
                logger.error(f"Timeout retrieving executed trades for {symbol}: {timeout_e}")
                logger.info("This could indicate that the yfinance-db service is running but not responding")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error connecting to YFinance DB service for trades: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
            
    def check_yfinance_db_connection(self):
        """
        Check if the YFinance DB service is available.
        
        Returns:
            bool: True if connected, False otherwise
        """
        try:
            # Log connection details for debugging
            logger.info(f"Attempting to connect to YFinance DB service at {self.yfinance_db_url}")
            logger.info(f"YFinance DB host: {self.yfinance_db_host}, port: {self.yfinance_db_port}")
            
            # Log environment variables
            import os
            logger.info(f"Environment variables:")
            logger.info(f"YFINANCE_DB_HOST: {os.getenv('YFINANCE_DB_HOST', 'not set')}")
            logger.info(f"YFINANCE_DB_PORT: {os.getenv('YFINANCE_DB_PORT', 'not set')}")
            
            # Check if the service is running in Docker
            is_docker = os.path.exists('/.dockerenv')
            logger.info(f"Running in Docker container: {is_docker}")
            
            # First try the root endpoint to see if the service is running at all
            root_url = f"{self.yfinance_db_url}/"
            logger.info(f"Checking root endpoint: {root_url}")
            
            # Use a session for connection pooling and retry mechanism
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                max_retries=3,  # Retry up to 3 times
                pool_connections=10,
                pool_maxsize=10
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            try:
                root_response = session.get(root_url, timeout=5)
                logger.info(f"Root endpoint response: {root_response.status_code}")
                if root_response.status_code == 200:
                    logger.info(f"Root endpoint content: {root_response.text[:200]}...")
                    # Check if the response is valid JSON
                    try:
                        root_data = root_response.json()
                        logger.info(f"Root endpoint returned valid JSON: {root_data}")
                    except ValueError:
                        logger.warning(f"Root endpoint did not return valid JSON: {root_response.text[:200]}")
            except requests.exceptions.ConnectionError as conn_e:
                logger.error(f"Connection error to root endpoint: {conn_e}")
                logger.info("This could indicate that the yfinance-db service is not running or not accessible")
            except requests.exceptions.Timeout as timeout_e:
                logger.error(f"Timeout connecting to root endpoint: {timeout_e}")
                logger.info("This could indicate that the yfinance-db service is running but not responding")
            except Exception as root_e:
                logger.error(f"Error connecting to root endpoint: {root_e}")
            
            # Now try the health endpoint
            url = f"{self.yfinance_db_url}/health"
            logger.info(f"Checking health endpoint: {url}")
            
            try:
                response = session.get(url, timeout=5)
                
                if response.status_code == 200:
                    self.yfinance_db_connected = True
                    logger.info(f"Successfully connected to YFinance DB service health endpoint")
                    
                    # Check if the response is valid JSON
                    try:
                        health_data = response.json()
                        logger.info(f"Health endpoint status: {health_data.get('status', 'unknown')}")
                        
                        # Check database status if available
                        if 'database' in health_data:
                            db_info = health_data['database']
                            logger.info(f"Database tables: {db_info.get('tables', [])}")
                            logger.info(f"Record counts: {db_info.get('record_counts', {})}")
                            
                            # Check specifically for trading_signals table
                            record_counts = db_info.get('record_counts', {})
                            if 'trading_signals' in record_counts:
                                logger.info(f"Trading signals count: {record_counts['trading_signals']}")
                                if record_counts['trading_signals'] == 0:
                                    logger.warning("Trading signals table exists but has no records!")
                    except ValueError:
                        logger.warning(f"Health endpoint did not return valid JSON: {response.text[:200]}")
                    
                    return True
                else:
                    self.yfinance_db_connected = False
                    logger.warning(f"YFinance DB service health endpoint returned status code: {response.status_code}")
                    if hasattr(response, 'text'):
                        logger.warning(f"Response content: {response.text[:200]}...")
                    return False
            except requests.exceptions.ConnectionError as conn_e:
                self.yfinance_db_connected = False
                logger.error(f"Connection error to health endpoint: {conn_e}")
                logger.info("This could indicate that the yfinance-db service is not running or not accessible")
                return False
            except requests.exceptions.Timeout as timeout_e:
                self.yfinance_db_connected = False
                logger.error(f"Timeout connecting to health endpoint: {timeout_e}")
                logger.info("This could indicate that the yfinance-db service is running but not responding")
                return False
                
        except Exception as e:
            self.yfinance_db_connected = False
            logger.error(f"Error connecting to YFinance DB service health endpoint: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
            
    def refresh_account_data(self):
        """
        Refresh account and positions data from the market data service.
        """
        try:
            self.account_info = self.market_data.get_account()
            self.positions = self.market_data.get_positions()
            logger.info(f"Account data refreshed successfully: equity=${float(self.account_info.get('equity', 0)):.2f}")
            logger.info(f"Retrieved {len(self.positions)} positions")
            
            # Log position details for debugging
            for position in self.positions:
                symbol = position.symbol if hasattr(position, 'symbol') else 'Unknown'
                qty = position.qty if hasattr(position, 'qty') else 0
                value = position.market_value if hasattr(position, 'market_value') else 0
                logger.info(f"Position: {symbol}, Qty: {qty}, Value: ${float(value):.2f}")
        except Exception as e:
            logger.error(f"Error refreshing account data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def run(self):
        """
        Run the Streamlit dashboard.
        """
        st.set_page_config(
            page_title="Alpaca Trading Bot Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Sidebar
        self._build_sidebar()

        # Main content
        st.title("Alpaca Trading Bot Dashboard")
        
        # Display connection status
        self._display_connection_status()
        
        # Display market status
        self._display_market_status()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "Portfolio Overview", 
            "Market Data & Signals", 
            "Strategy Performance", 
            "Trading History"
        ])
        
        with tab1:
            self._build_portfolio_tab()
            
        with tab2:
            self._build_market_data_tab()
            
        with tab3:
            self._build_strategy_tab()
            
        with tab4:
            self._build_trading_history_tab()

    def _build_sidebar(self):
        """
        Build the sidebar with controls and settings.
        """
        st.sidebar.title("Controls")
        
        # Strategy selection
        selected_strategy = st.sidebar.selectbox(
            "Select Strategy",
            list(self.strategies.keys()),
            format_func=lambda x: self.strategies[x]
        )
        
        if selected_strategy != self.current_strategy:
            self.current_strategy = selected_strategy
            self.strategy = get_strategy(selected_strategy)
            st.sidebar.success(f"Strategy changed to {self.strategies[selected_strategy]}")
        
        # Symbol selection
        selected_symbols = st.sidebar.multiselect(
            "Select Symbols",
            config.SYMBOLS,
            default=config.SYMBOLS[:3]
        )
        
        # Timeframe selection
        timeframe = st.sidebar.selectbox(
            "Timeframe",
            ["1D", "1H", "15Min"],
            index=0
        )
        
        # Data period selection
        data_period = st.sidebar.slider(
            "Data Period (days)",
            min_value=7,
            max_value=100,
            value=30,
            step=1
        )
        
        # Fetch data button
        if st.sidebar.button("Fetch Data"):
            with st.sidebar:
                with st.spinner("Fetching data..."):
                    if hasattr(self.strategy, 'fetch_data') and callable(getattr(self.strategy, 'fetch_data')):
                        self.data = self.strategy.fetch_data(selected_symbols)
                        st.success(f"Data fetched for {len(self.data)} symbols using YFinance")
                    else:
                        self.data = self.market_data.get_bars(selected_symbols, timeframe, data_period)
                        st.success(f"Data fetched for {len(self.data)} symbols using Alpaca API")
        
        # Account refresh button
        if st.sidebar.button("Refresh Account Data"):
            with st.sidebar:
                with st.spinner("Refreshing account data..."):
                    self.account_info = self.market_data.get_account()
                    self.positions = self.market_data.get_positions()
                    st.success("Account data refreshed")
        
        # Display bot status
        st.sidebar.subheader("Bot Status")
        is_market_open = self.market_data.is_market_open()
        
        # Create two columns for status metrics
        col1, col2 = st.sidebar.columns(2)
        
        col1.metric(
            "Market Status",
            "Open" if is_market_open else "Closed",
            delta=None
        )
        
        # Display YFinance DB connection status
        yf_status = "Connected" if self.yfinance_db_connected else "Disconnected"
        yf_status_color = "green" if self.yfinance_db_connected else "red"
        col2.markdown(f"**YFinance DB:** <span style='color:{yf_status_color}'>{yf_status}</span>", unsafe_allow_html=True)
        
        # Display API connection status
        try:
            account = self.market_data.get_account()
            api_status = "Connected" if account else "Error"
            api_status_color = "green" if account else "red"
        except:
            api_status = "Error"
            api_status_color = "red"
            
        col1.markdown(f"**API Status:** <span style='color:{api_status_color}'>{api_status}</span>", unsafe_allow_html=True)
        
        # Add a refresh button for YFinance DB connection
        if st.sidebar.button("Check YFinance DB Connection"):
            with st.sidebar:
                with st.spinner("Checking connection..."):
                    if self.check_yfinance_db_connection():
                        st.success("Connected to YFinance DB service")
                    else:
                        st.error("Failed to connect to YFinance DB service")
        
        # Display environment info
        st.sidebar.subheader("Environment")
        st.sidebar.info(
            f"API Mode: {'Paper' if 'paper' in config.ALPACA_BASE_URL else 'Live'}\n"
            f"Timezone: {config.TIMEZONE}\n"
            f"Run Frequency: Every {config.RUN_INTERVAL} {config.RUN_FREQUENCY}"
        )

    def _display_market_status(self):
        """
        Display market status information.
        """
        # Create three columns
        col1, col2, col3 = st.columns(3)
        
        # Market status
        is_market_open = self.market_data.is_market_open()
        col1.metric(
            "Market Status",
            "Open" if is_market_open else "Closed"
        )
        
        # Market hours
        try:
            market_hours = self.market_data.get_market_hours()
            if market_hours[0]:
                open_time = market_hours[0].strftime("%H:%M:%S")
                close_time = market_hours[1].strftime("%H:%M:%S")
                col2.metric("Market Hours", f"{open_time} - {close_time}")
            else:
                col2.metric("Market Hours", "Unknown")
        except:
            col2.metric("Market Hours", "Error fetching")
        
        # Current time
        now = datetime.now(pytz.timezone(config.TIMEZONE))
        col3.metric("Current Time", now.strftime("%Y-%m-%d %H:%M:%S"))
        
    def _build_portfolio_tab(self):
        """
        Build the portfolio overview tab with enhanced visualizations.
        """
        st.header("Portfolio Overview")
        
        try:
            # Use cached account data if available, otherwise fetch it
            account_info = self.account_info if self.account_info else self.market_data.get_account()
            positions = self.positions if self.positions else self.market_data.get_positions()
            
            if not account_info:
                st.warning("Could not fetch account information. Please check your API connection.")
                # Add a retry button
                if st.button("Retry Fetching Account Data"):
                    with st.spinner("Fetching account data..."):
                        self.refresh_account_data()
                        st.experimental_rerun()
                return
                
            # Account metrics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "Portfolio Value",
                f"${float(account_info.get('portfolio_value', 0)):.2f}"
            )
            
            col2.metric(
                "Cash",
                f"${float(account_info.get('cash', 0)):.2f}"
            )
            
            col3.metric(
                "Buying Power",
                f"${float(account_info.get('buying_power', 0)):.2f}"
            )
            
            # Calculate daily P&L based on positions
            daily_pl = 0
            if positions:
                for position in positions:
                    if 'unrealized_pl' in position:
                        daily_pl += float(position.get('unrealized_pl', 0))
            
            # Determine if P&L is positive or negative for the delta color
            delta_color = "normal"
            if daily_pl > 0:
                delta_color = "green"
            elif daily_pl < 0:
                delta_color = "red"
                
            col4.metric(
                "Unrealized P&L",
                f"${daily_pl:.2f}",
                delta=f"{daily_pl:.2f}"
            )
            
            # Positions table
            st.subheader(f"Current Positions ({len(positions)})")
            
            if positions:
                # Create a DataFrame for positions
                positions_df = pd.DataFrame(positions)
                
                # Calculate additional metrics
                positions_df['current_value'] = positions_df['qty'] * positions_df['current_price']
                positions_df['cost_basis'] = positions_df['qty'] * positions_df['avg_entry_price']
                positions_df['profit_loss'] = positions_df['current_value'] - positions_df['cost_basis']
                positions_df['profit_loss_pct'] = (positions_df['profit_loss'] / positions_df['cost_basis']) * 100
                
                # Format the DataFrame for display
                display_df = positions_df[['symbol', 'qty', 'avg_entry_price', 'current_price',
                                          'market_value', 'unrealized_pl', 'profit_loss_pct']]
                display_df = display_df.rename(columns={
                    'symbol': 'Symbol',
                    'qty': 'Quantity',
                    'avg_entry_price': 'Entry Price',
                    'current_price': 'Current Price',
                    'market_value': 'Market Value',
                    'unrealized_pl': 'Unrealized P&L',
                    'profit_loss_pct': 'P&L %'
                })
                
                # Format numeric columns
                for col in ['Entry Price', 'Current Price', 'Market Value', 'Unrealized P&L']:
                    display_df[col] = display_df[col].map('${:,.2f}'.format)
                    
                display_df['P&L %'] = display_df['P&L %'].map('{:,.2f}%'.format)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Portfolio allocation pie chart with enhanced visualization
                st.subheader("Portfolio Allocation")
                
                # Create two columns for the charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Pie chart for allocation by symbol
                    fig1 = px.pie(
                        positions_df,
                        values='market_value',
                        names='symbol',
                        title='Allocation by Symbol',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    fig1.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig1, use_container_width=True)
                
                with chart_col2:
                    # Bar chart for position performance
                    fig2 = px.bar(
                        positions_df,
                        x='symbol',
                        y='profit_loss_pct',
                        title='Position Performance (%)',
                        color='profit_loss_pct',
                        color_continuous_scale=['red', 'yellow', 'green'],
                        text='profit_loss_pct'
                    )
                    fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                    fig2.update_layout(xaxis_title="Symbol", yaxis_title="Profit/Loss (%)")
                    st.plotly_chart(fig2, use_container_width=True)
                
            else:
                st.info("No positions currently held.")
                
        except Exception as e:
            st.error(f"Error building portfolio tab: {e}")
            
    def _build_market_data_tab(self):
        """
        Build the market data and signals tab.
        """
        st.header("Market Data & Signals")
        
        if not self.data:
            st.info("No data available. Please fetch data using the sidebar controls.")
            return
            
        # Symbol selection for charts
        symbols = list(self.data.keys())
        if not symbols:
            st.warning("No data available for any symbols.")
            return
            
        selected_symbol = st.selectbox("Select Symbol", symbols)
        
        if selected_symbol not in self.data:
            st.warning(f"No data available for {selected_symbol}.")
            return
            
        df = self.data[selected_symbol]
        
        # Generate signals for the selected symbol
        signals = {}
        try:
            signals = self.strategy.generate_signals({selected_symbol: df})
        except Exception as e:
            st.error(f"Error generating signals: {e}")
        
        # Display current signal
        if selected_symbol in signals:
            signal_data = signals[selected_symbol]
            
            cols = st.columns(4)
            
            # Action (buy/sell/hold)
            action = signal_data.get('action', 'unknown')
            action_color = {
                'buy': 'green',
                'sell': 'red',
                'hold': 'blue'
            }.get(action, 'gray')
            
            cols[0].markdown(f"**Current Signal:** <span style='color:{action_color};font-weight:bold;font-size:1.2em;'>{action.upper()}</span>", unsafe_allow_html=True)
            
            # Price
            price = signal_data.get('price', 0)
            cols[1].metric("Current Price", f"${price:.2f}")
            
            # Signal changed
            signal_changed = signal_data.get('signal_changed', False)
            cols[2].metric("Signal Changed", "Yes" if signal_changed else "No")
            
            # Timestamp
            timestamp = signal_data.get('timestamp', None)
            if timestamp:
                if isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.to_pydatetime()
                cols[3].metric("Signal Time", timestamp.strftime("%Y-%m-%d %H:%M"))
        
        # Price chart with indicators
        st.subheader(f"{selected_symbol} Price Chart with Indicators")
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price"
            )
        )
        
        # Add moving averages if available
        if 'sma_short' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['sma_short'],
                    name=f"SMA {config.SHORT_WINDOW}",
                    line=dict(color='blue', width=1)
                )
            )
            
        if 'sma_long' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['sma_long'],
                    name=f"SMA {config.LONG_WINDOW}",
                    line=dict(color='orange', width=1)
                )
            )
        
        # Add RSI on secondary y-axis if available
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['rsi'],
                    name="RSI",
                    line=dict(color='purple', width=1),
                    yaxis="y2"
                )
            )
            
            # Add RSI overbought/oversold lines
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=[config.RSI_OVERBOUGHT] * len(df),
                    name="Overbought",
                    line=dict(color='red', width=1, dash='dash'),
                    yaxis="y2"
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=[config.RSI_OVERSOLD] * len(df),
                    name="Oversold",
                    line=dict(color='green', width=1, dash='dash'),
                    yaxis="y2"
                )
            )
        
        # Update layout with secondary y-axis
        fig.update_layout(
            title=f"{selected_symbol} Price and Indicators",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2=dict(
                title="RSI",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        st.subheader(f"{selected_symbol} Volume")
        
        volume_fig = px.bar(
            df,
            x='timestamp',
            y='volume',
            title=f"{selected_symbol} Trading Volume"
        )
        
        st.plotly_chart(volume_fig, use_container_width=True)
        
    def _build_strategy_tab(self):
        """
        Build the strategy performance tab.
        """
        st.header("Strategy Performance")
        
        if not hasattr(self, 'strategy') or not self.strategy:
            st.info("Please select a strategy from the sidebar first.")
            return
            
        # Display strategy information
        st.subheader(f"Strategy: {self.strategy.name}")
        
        # Strategy description based on type
        strategy_descriptions = {
            'Moving Average Crossover': """
                This strategy generates buy signals when the short-term moving average crosses above
                the long-term moving average, and sell signals when it crosses below.
                
                **Parameters:**
                - Short Window: {short} days
                - Long Window: {long} days
            """.format(short=config.SHORT_WINDOW, long=config.LONG_WINDOW),
            
            'RSI Strategy': """
                This strategy generates buy signals when the RSI indicator crosses below the oversold threshold,
                and sell signals when it crosses above the overbought threshold.
                
                **Parameters:**
                - RSI Period: {period} days
                - Oversold Threshold: {oversold}
                - Overbought Threshold: {overbought}
            """.format(period=config.RSI_PERIOD, oversold=config.RSI_OVERSOLD, overbought=config.RSI_OVERBOUGHT),
            
            'ML Strategy': """
                This strategy uses machine learning models to predict price movements and generate trading signals.
                
                **Parameters:**
                - Model Type: {model_type}
                - Features: {features}
                - Confidence Threshold: {threshold}
            """.format(
                model_type=config.ML_STRATEGY_TYPE,
                features=", ".join(config.ML_FEATURES[:3]) + "...",
                threshold=config.ML_CONFIDENCE_THRESHOLD
            ),
            
            'Dual Moving Average YF': """
                This strategy is similar to the Moving Average Crossover strategy but uses data from Yahoo Finance.
                
                **Parameters:**
                - Short Window: {short} days
                - Long Window: {long} days
            """.format(short=config.SHORT_WINDOW, long=config.LONG_WINDOW)
        }
        
        # Display strategy description
        if self.strategy.name in strategy_descriptions:
            st.markdown(strategy_descriptions[self.strategy.name])
        else:
            st.markdown("No detailed description available for this strategy.")
        
        # Strategy performance metrics (placeholder)
        st.subheader("Performance Metrics")
        
        # In a real implementation, you would calculate these metrics based on historical trades
        # For now, we'll use placeholder values
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Win Rate", "65%")
        col2.metric("Profit Factor", "1.8")
        col3.metric("Sharpe Ratio", "1.2")
        col4.metric("Max Drawdown", "-12%")
        
        # Strategy backtest chart (placeholder)
        st.subheader("Backtest Results")
        
        # Generate some placeholder data for the backtest chart
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        baseline = np.linspace(100, 130, 100) + np.random.normal(0, 3, 100).cumsum()
        strategy = np.linspace(100, 150, 100) + np.random.normal(0, 4, 100).cumsum()
        
        backtest_df = pd.DataFrame({
            'date': dates,
            'Baseline': baseline,
            'Strategy': strategy
        })
        
        # Create a line chart for the backtest results
        backtest_fig = px.line(
            backtest_df,
            x='date',
            y=['Baseline', 'Strategy'],
            title='Strategy vs. Baseline Performance',
            labels={'value': 'Portfolio Value', 'date': 'Date'}
        )
        
        st.plotly_chart(backtest_fig, use_container_width=True)
        
        # Strategy parameters
        st.subheader("Strategy Parameters")
        
        # Display different parameters based on strategy type
        if isinstance(self.strategy, type) and hasattr(self.strategy, '__name__'):
            strategy_type = self.strategy.__name__
        else:
            strategy_type = type(self.strategy).__name__
            
        if strategy_type == 'MovingAverageCrossover' or strategy_type == 'DualMovingAverageYF':
            col1, col2 = st.columns(2)
            col1.number_input("Short Window", min_value=5, max_value=50, value=config.SHORT_WINDOW, key="short_window")
            col2.number_input("Long Window", min_value=20, max_value=200, value=config.LONG_WINDOW, key="long_window")
            
        elif strategy_type == 'RSIStrategy':
            col1, col2, col3 = st.columns(3)
            col1.number_input("RSI Period", min_value=5, max_value=30, value=config.RSI_PERIOD, key="rsi_period")
            col2.number_input("Oversold Threshold", min_value=10, max_value=40, value=config.RSI_OVERSOLD, key="rsi_oversold")
            col3.number_input("Overbought Threshold", min_value=60, max_value=90, value=config.RSI_OVERBOUGHT, key="rsi_overbought")
            
        elif strategy_type == 'MLStrategy':
            col1, col2 = st.columns(2)
            col1.selectbox("Model Type", ["ensemble", "neural_network", "reinforcement", "nlp"], index=0, key="ml_model_type")
            col2.number_input("Confidence Threshold", min_value=0.5, max_value=0.95, value=config.ML_CONFIDENCE_THRESHOLD, key="ml_confidence")
            
        # Note: In a real implementation, you would save these parameters and apply them to the strategy
        
    def _build_trading_history_tab(self):
        """
        Build the trading history tab with real trading signals and executed trades.
        """
        st.header("Trading History")
        
        # Symbol selection for history
        selected_symbol = st.selectbox(
            "Select Symbol for Trading History",
            config.SYMBOLS,
            key="history_symbol"
        )
        
        # Create tabs for signals and trades
        signal_tab, trade_tab = st.tabs(["Trading Signals", "Executed Trades"])
        
        with signal_tab:
            st.subheader(f"Trading Signals for {selected_symbol}")
            
            # Fetch trading signals
            signals_df = self.get_trading_signals(selected_symbol)
            
            if signals_df.empty:
                st.info(f"No trading signals found for {selected_symbol}.")
            else:
                # Format the DataFrame for display
                display_df = signals_df.copy()
                
                # Format timestamp
                if 'timestamp' in display_df.columns:
                    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                
                # Format price and other numeric columns
                numeric_cols = ['price', 'short_ma', 'long_ma', 'rsi']
                for col in numeric_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
                
                # Rename columns for display
                display_df = display_df.rename(columns={
                    'timestamp': 'Date/Time',
                    'action': 'Action',
                    'signal': 'Signal Value',
                    'signal_changed': 'Signal Changed',
                    'price': 'Price',
                    'short_ma': 'Short MA',
                    'long_ma': 'Long MA',
                    'rsi': 'RSI',
                    'strategy': 'Strategy'
                })
                
                # Display the signals
                st.dataframe(display_df, use_container_width=True)
                
                # Create a signal visualization
                st.subheader(f"Signal Visualization for {selected_symbol}")
                
                # Prepare data for visualization
                if 'timestamp' in signals_df.columns and 'price' in signals_df.columns:
                    # Create a figure
                    fig = go.Figure()
                    
                    # Add price line
                    fig.add_trace(go.Scatter(
                        x=signals_df['timestamp'],
                        y=signals_df['price'],
                        mode='lines',
                        name='Price'
                    ))
                    
                    # Add buy signals
                    buy_signals = signals_df[signals_df['action'] == 'buy']
                    if not buy_signals.empty:
                        fig.add_trace(go.Scatter(
                            x=buy_signals['timestamp'],
                            y=buy_signals['price'],
                            mode='markers',
                            marker=dict(color='green', size=10, symbol='triangle-up'),
                            name='Buy Signal'
                        ))
                    
                    # Add sell signals
                    sell_signals = signals_df[signals_df['action'] == 'sell']
                    if not sell_signals.empty:
                        fig.add_trace(go.Scatter(
                            x=sell_signals['timestamp'],
                            y=sell_signals['price'],
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='triangle-down'),
                            name='Sell Signal'
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{selected_symbol} Price and Signals",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Insufficient data for signal visualization.")
        
        with trade_tab:
            st.subheader(f"Executed Trades for {selected_symbol}")
            
            # Fetch executed trades
            trades_df = self.get_executed_trades(selected_symbol)
            
            if trades_df.empty:
                st.info(f"No executed trades found for {selected_symbol}.")
            else:
                # Format the DataFrame for display
                display_df = trades_df.copy()
                
                # Format timestamp
                if 'timestamp' in display_df.columns:
                    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                
                # Format price and quantity
                if 'price' in display_df.columns:
                    display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
                
                if 'quantity' in display_df.columns and 'price' in trades_df.columns:
                    display_df['total_value'] = trades_df['quantity'] * trades_df['price']
                    display_df['total_value'] = display_df['total_value'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
                
                # Rename columns for display
                display_df = display_df.rename(columns={
                    'timestamp': 'Date/Time',
                    'side': 'Side',
                    'quantity': 'Quantity',
                    'price': 'Price',
                    'status': 'Status',
                    'total_value': 'Total Value'
                })
                
                # Display the trades
                st.dataframe(display_df, use_container_width=True)
                
                # Performance metrics
                st.subheader("Performance Metrics")
                
                # Calculate performance metrics
                total_trades = len(trades_df)
                buy_trades = len(trades_df[trades_df['side'] == 'buy'])
                sell_trades = len(trades_df[trades_df['side'] == 'sell'])
                
                # Calculate total value if price and quantity are available
                total_value = 0
                if 'price' in trades_df.columns and 'quantity' in trades_df.columns:
                    total_value = (trades_df['price'] * trades_df['quantity']).sum()
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Total Trades", total_trades)
                col2.metric("Buy Trades", buy_trades)
                col3.metric("Sell Trades", sell_trades)
                col4.metric("Total Value", f"${total_value:,.2f}")
                
                # Trading performance chart (if we have enough data)
                if len(trades_df) > 1:
                    st.subheader("Trading Performance")
                    
                    # Sort trades by timestamp
                    sorted_trades = trades_df.sort_values('timestamp')
                    
                    # Create a performance chart
                    fig = go.Figure()
                    
                    # Add a trace for each trade
                    fig.add_trace(go.Scatter(
                        x=sorted_trades['timestamp'],
                        y=sorted_trades['price'],
                        mode='markers+lines',
                        name='Trade Price',
                        marker=dict(
                            color=sorted_trades['side'].apply(lambda x: 'green' if x == 'buy' else 'red'),
                            size=10,
                            symbol=sorted_trades['side'].apply(lambda x: 'triangle-up' if x == 'buy' else 'triangle-down')
                        )
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{selected_symbol} Trading Performance",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        # Note: The performance_fig is already plotted above, no need to plot it again
