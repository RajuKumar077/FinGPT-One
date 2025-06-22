import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
import requests
import json
import yfinance as yf # Import yfinance for historical data

# Import functions from your separate modules
import pages.fmp_autocomplete as fmp_autocomplete # Retaining FMP for autocomplete
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

# API Keys
NEWS_API_KEY = "874ba654bdcd4aa7b68f7367a907cc2f"
# FMP API Key will now ONLY be used for symbol search/autocomplete and company name for news
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
def load_historical_data(ticker_symbol):
    """
    Loads historical stock data using yfinance.
    This is generally more reliable for historical OHLCV data across many exchanges.
    """
    if not ticker_symbol:
        return pd.DataFrame()

    try:
        # yfinance automatically handles common ticker formats like TCS.NS, RELIANCE.BO, AAPL
        ticker = yf.Ticker(ticker_symbol)
        # Fetch max historical data available
        hist_df = ticker.history(period="max") 

        if hist_df.empty:
            st.warning(f"⚠️ No historical data found for {ticker_symbol} via yfinance. Please check the ticker symbol and its exchange suffix (e.g., .NS for NSE, .BO for BSE for Indian stocks).")
            return pd.DataFrame()
        
        # Reset index to make 'Date' a column
        hist_df.reset_index(inplace=True)
        
        # Rename columns to match expected format
        hist_df.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
            'Adj Close': 'Adjusted_Close'
        }, inplace=True)

        # Ensure 'Date' is datetime and then convert to date object for consistency
        hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.date
        
        # Keep only required columns and sort
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        hist_df = hist_df[required_cols]
        hist_df.sort_values(by='Date', ascending=True, inplace=True)
        hist_df.reset_index(drop=True, inplace=True)

        return hist_df

    except Exception as e:
        st.error(f"❌ Error loading historical data for {ticker_symbol} from yfinance: {e}. Please check the ticker symbol or try again later.")
        return pd.DataFrame()

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
    # Handle division by zero: if avg_loss is 0, rs becomes infinity, RSI becomes 100
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
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

    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = ""
    if 'analyze_triggered' not in st.session_state:
        st.session_state.analyze_triggered = False

    ticker_input = st.text_input(
        "Search Stock Ticker (e.g., AAPL, RELIANCE.NS, TCS.BO)",
        value=st.session_state.current_ticker,
        key="main_ticker_search",
        help="Type a few letters to see suggestions. Press Enter to analyze. For Indian stocks, use .NS for NSE (e.g., RELIANCE.NS) and .BO for BSE (e.g., TCS.BO)."
    )

    suggestions = []
    if ticker_input:
        # Use FMP for autocomplete with your key
        suggestions = fmp_autocomplete.fetch_fmp_suggestions(ticker_input, api_key=FMP_API_KEY)

    if suggestions:
        st.markdown("<h5>Suggestions:</h5>", unsafe_allow_html=True)
        num_columns_to_create = min(len(suggestions), 5)

        if num_columns_to_create > 0:
            cols = st.columns(num_columns_to_create)
            for i, suggestion in enumerate(suggestions):
                if i < len(cols):
                    with cols[i]:
                        suggested_ticker = suggestion.split(' - ')[0]
                        if st.button(suggestion, key=f"suggestion_{i}"):
                            st.session_state.current_ticker = suggested_ticker
                            st.session_state.analyze_triggered = True
                            st.rerun()

    if st.button("🚀 Analyze Stock", key="analyze_button", type="primary"):
        if ticker_input:
            st.session_state.current_ticker = ticker_input.split(' - ')[0]
            st.session_state.analyze_triggered = True
        else:
            st.warning("Please enter a stock ticker to analyze.")
            st.session_state.analyze_triggered = False
        st.rerun()

    if st.session_state.analyze_triggered and st.session_state.current_ticker:
        ticker_to_analyze = st.session_state.current_ticker

        st.markdown(f"<h2 class='analysis-header'>Comprehensive Analysis for {ticker_to_analyze.upper()}</h2>",
                    unsafe_allow_html=True)

        tab_summary, tab_financials, tab_probabilistic, tab_forecast, tab_news = st.tabs([
            "Company Overview", "Financials", "Probabilistic Models", "Forecasting", "News Sentiment"
        ])

        if 'historical_data' not in st.session_state or st.session_state.historical_data is None or st.session_state.historical_data.empty or (hasattr(st.session_state.historical_data, 'name') and st.session_state.historical_data.name != ticker_to_analyze):
            with st.spinner(f"Loading historical data for {ticker_to_analyze.upper()}..."):
                # Call load_historical_data without FMP API key, as it uses yfinance now
                st.session_state.historical_data = load_historical_data(ticker_to_analyze)
                if not st.session_state.historical_data.empty:
                    st.session_state.historical_data.name = ticker_to_analyze

        hist_data_for_tabs = st.session_state.historical_data

        if hist_data_for_tabs.empty:
            st.error(f"❌ Failed to load historical data for {ticker_to_analyze}. Some analysis tabs might not function.")
            st.session_state.analyze_triggered = False
            return

        with tab_summary:
            # Pass FMP API key for basic company info via FMP
            stock_summary.display_stock_summary(ticker_to_analyze, api_key=FMP_API_KEY)

        with tab_financials:
            # Pass FMP API key for financials via FMP
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

