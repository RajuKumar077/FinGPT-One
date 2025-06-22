import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
import requests # Import requests for API calls
import json # Import json for parsing API responses

# Import functions from your separate modules
# This has been updated from 'yahoo_autocomplete' to 'fmp_autocomplete'
import pages.fmp_autocomplete as fmp_autocomplete
import pages.stock_summary as stock_summary
import pages.financials as financials
import pages.probabilistic_stock_model as probabilistic_stock_model
import pages.forecast_module as forecast_module
import pages.news_sentiment as news_sentiment

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# --- GLOBAL CONFIGURATIONS AND INITIAL STREAMLIT SETUP ---
st.set_page_config(
    page_title="Intelligent Stock Insights",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': None
    },
    page_icon="📈"
)

# Custom CSS to hide the Streamlit hamburger menu icon (sidebar toggle) and the header
st.markdown("""
    <style>
        /* Hide the Streamlit hamburger menu icon */
        button[data-testid="stSidebarToggle"] {
            display: none !important;
        }
        /* Hide the entire Streamlit header which often contains the sidebar toggle and other default elements */
        header {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# NewsAPI.com API Key - IMPORTANT: Replace with your actual key
NEWS_API_KEY = "874ba654bdcd4aa7b68f7367a907cc2f"

# Financial Modeling Prep API Key - Your provided key
FMP_API_KEY = "5C9DnMCAzYam2ZPjNpOxKLFxUiGhrJDD"

# --- Custom CSS and Font Loading ---
def load_css(file_path):
    """Loads custom CSS from a file."""
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: CSS file not found at {file_path}. Please ensure 'assets/style.css' exists.")

st.markdown(
    "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap' rel='stylesheet'>",
    unsafe_allow_html=True)
load_css("assets/style.css")


# --- Common Data Loading & Feature Engineering Functions ---
@st.cache_data(ttl=3600, show_spinner=False)  # Cache historical data for 1 hour
def load_historical_data(ticker_symbol, retries=5, initial_delay=0.75):
    """
    Loads historical stock data from Financial Modeling Prep (FMP) with retry logic.
    FMP provides daily data.
    """
    if not ticker_symbol:
        return pd.DataFrame()

    # FMP API endpoint for historical daily prices
    base_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_symbol}"
    params = {
        "apikey": FMP_API_KEY
    }

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                print(f"Retrying historical data fetch for {ticker_symbol} (attempt {attempt}/{retries}). Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                time.sleep(initial_delay) # Initial delay before first API call

            response = requests.get(base_url, params=params, timeout=20) # Increased timeout
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            # FMP returns data in 'historical' key
            if "historical" not in data or not data["historical"]:
                print(f"No historical data found for {ticker_symbol}. Response: {data}")
                if "Error Message" in data: # Check for FMP-specific error messages
                    st.error(f"❌ FMP API Error for {ticker_symbol}: {data['Error Message']}. Please check the ticker symbol or API key.")
                elif "limit" in str(data).lower(): # Generic check for rate limit messages
                     st.warning(f"⚠️ FMP API rate limit might be hit for {ticker_symbol}. Please wait and try again or check your API key usage.")
                else:
                    st.error(f"❌ No historical data found for {ticker_symbol}. Please check the ticker symbol or the API response structure.")
                if attempt == retries:
                    return pd.DataFrame()
                continue # Retry if not last attempt

            # Convert to DataFrame
            df = pd.DataFrame(data["historical"])
            df['date'] = pd.to_datetime(df['date'])
            df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adjClose': 'Adjusted_Close' # FMP often provides adjClose directly
            }, inplace=True)

            # Ensure expected columns are present
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Missing required columns in FMP historical data for {ticker_symbol}.")
                return pd.DataFrame()

            df.sort_values(by='Date', ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df['Date'] = df['Date'].dt.date # Keep only date part

            if df.empty:
                if attempt == retries:
                    st.error(f"❌ No historical data found for {ticker_symbol} after processing. Please check the ticker symbol.")
                continue

            return df

        except requests.exceptions.RequestException as req_err:
            print(f"Attempt {attempt}/{retries}: Network or API request error for {ticker_symbol}: {req_err}")
            if attempt == retries:
                st.error(f"⚠️ Network error or API issue for {ticker_symbol}. Please check your internet connection or try again later.")
                return pd.DataFrame()
        except json.JSONDecodeError as json_err:
            print(f"Attempt {attempt}/{retries}: JSON Decode Error for {ticker_symbol}: {json_err}. Response content starts with: {response.text[:200]}...")
            if attempt == retries:
                st.error(f"⚠️ Received invalid data from API for {ticker_symbol}. Please try again later.")
                return pd.DataFrame()
        except Exception as e:
            print(f"Attempt {attempt}/{retries}: An unexpected error occurred while fetching historical data for {ticker_symbol}: {e}")
            if attempt == retries:
                st.error(f"⚠️ An unexpected error occurred while fetching historical data for {ticker_symbol}. Please try again later.")
                return pd.DataFrame()
    return pd.DataFrame() # Fallback if all retries fail


@st.cache_data(show_spinner=False)
def add_common_features(hist):
    """Adds various technical indicators as features common to multiple modules."""
    df = hist.copy()
    # Ensure 'Close' column is numeric for calculations
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')

    df['Return_1d'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['High_Low_Diff'] = (df['High'] - df['Low']) / df['Close']
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # 1 = up, 0 = down
    df.dropna(inplace=True)
    return df


@st.cache_data(show_spinner=False)
def compute_rsi(series, period=14):
    """Computes the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    # Handle division by zero for rs calculation
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Avoid division by zero: if avg_loss is 0, rs becomes infinity, RSI becomes 100
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    
    return 100 - (100 / (1 + rs))

# --- Streamlit Application Main Layout ---
def main():
    st.markdown("""
        <div class="main-header">
            <h1>📈 Intelligent Stock Insights</h1>
            <p>Your Comprehensive AI-Powered Stock Analysis Dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Centralized Stock Ticker Input and Autocomplete ---
    st.markdown("<h3>Enter a Stock Ticker to Begin Analysis</h3>", unsafe_allow_html=True)

    # Initialize session state for ticker if not already present
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = ""
    if 'analyze_triggered' not in st.session_state:
        st.session_state.analyze_triggered = False

    # Text input for ticker search
    # Updated help text to reflect FMP's common ticker formats
    ticker_input = st.text_input(
        "Search Stock Ticker (e.g., AAPL, RELIANCE.NS, TCS.BO)",
        value=st.session_state.current_ticker,
        key="main_ticker_search",
        help="Type a few letters to see suggestions. Press Enter to analyze. For Indian stocks, use .NS for NSE (e.g., RELIANCE.NS) and .BO for BSE (e.g., TCS.BO)."
    )

    # Autocomplete suggestions
    suggestions = []
    if ticker_input:
        # Use the robust fetch_fmp_suggestions from fmp_autocomplete module
        suggestions = fmp_autocomplete.fetch_fmp_suggestions(ticker_input, api_key=FMP_API_KEY)

    # --- Only display columns and buttons if suggestions exist ---
    if suggestions:
        st.markdown("<h5>Suggestions:</h5>", unsafe_allow_html=True)
        num_columns_to_create = min(len(suggestions), 5)

        if num_columns_to_create > 0: # Ensure we create at least one column if suggestions are present
            cols = st.columns(num_columns_to_create)
            for i, suggestion in enumerate(suggestions):
                if i < len(cols): # Defensive check to prevent index out of range
                    with cols[i]:
                        # Extract just the ticker symbol, assuming format is "SYMBOL - Name"
                        suggested_ticker = suggestion.split(' - ')[0] 
                        if st.button(suggestion, key=f"suggestion_{i}"):
                            st.session_state.current_ticker = suggested_ticker
                            st.session_state.analyze_triggered = True
                            st.rerun()  # Rerun to update the main input and trigger analysis
    # --- END FIX ---

    # Analyze Button
    if st.button("🚀 Analyze Stock", key="analyze_button", type="primary"):
        if ticker_input:
            # Ensure just ticker symbol is saved, clean input if it contains description
            st.session_state.current_ticker = ticker_input.split(' - ')[0]
            st.session_state.analyze_triggered = True
        else:
            st.warning("Please enter a stock ticker to analyze.")
            st.session_state.analyze_triggered = False
        st.rerun()  # Rerun the app to process the analysis

    # --- Display Analysis Results in Tabs ---
    if st.session_state.analyze_triggered and st.session_state.current_ticker:
        ticker_to_analyze = st.session_state.current_ticker

        st.markdown(f"<h2 class='analysis-header'>Comprehensive Analysis for {ticker_to_analyze.upper()}</h2>",
                    unsafe_allow_html=True)

        # Tabs for different analysis sections
        tab_summary, tab_financials, tab_probabilistic, tab_forecast, tab_news = st.tabs([
            "Company Overview", "Financials", "Probabilistic Models", "Forecasting", "News Sentiment"
        ])

        # Use st.session_state to cache historical data
        # Only load data if ticker changes or not loaded yet
        if 'historical_data' not in st.session_state or st.session_state.historical_data is None or st.session_state.historical_data.empty or (hasattr(st.session_state.historical_data, 'name') and st.session_state.historical_data.name != ticker_to_analyze):
            with st.spinner(f"Loading historical data for {ticker_to_analyze.upper()}..."):
                # Call the load_historical_data function with retry logic
                st.session_state.historical_data = load_historical_data(ticker_to_analyze)
                if not st.session_state.historical_data.empty: # Only store name if data was actually loaded
                    st.session_state.historical_data.name = ticker_to_analyze  # Store ticker name with data for caching logic

        hist_data_for_tabs = st.session_state.historical_data

        if hist_data_for_tabs.empty: # Check for empty DataFrame instead of None
            # Error message is already shown by load_historical_data
            st.session_state.analyze_triggered = False  # Reset trigger
            return  # Exit function if no data

        with tab_summary:
            # Pass the FMP API key to the stock_summary module
            stock_summary.display_stock_summary(ticker_to_analyze, api_key=FMP_API_KEY)

        with tab_financials:
            # Pass the FMP API key to the financials module
            financials.display_financials(ticker_to_analyze, api_key=FMP_API_KEY)

        with tab_probabilistic:
            probabilistic_stock_model.display_probabilistic_models(hist_data_for_tabs)

        with tab_news:
            # Pass the FMP API key to news_sentiment for company name lookup
            news_sentiment.display_news_sentiment(ticker_to_analyze, news_api_key=NEWS_API_KEY, fmp_api_key=FMP_API_KEY)

        with tab_forecast:
            forecast_module.display_forecasting(hist_data_for_tabs)


if __name__ == "__main__":
    main()

