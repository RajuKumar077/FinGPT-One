import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
import requests
import json
import yfinance as yf # Primary for historical data

# Import functions from your separate modules
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

# API Keys (IMPORTANT: REPLACE "YOUR_KEY_HERE" WITH YOUR ACTUAL KEYS)
NEWS_API_KEY = "874ba654bdcd4aa7b68f7367a907cc2f" # Get your free key from newsapi.com
FMP_API_KEY = "5C9DnMCAzYam2ZPjNpOxKLFxUiGhrJDD"     # Get your free key from financialmodelingprep.com
GEMINI_API_KEY = "AIzaSyAK8BevJ1wIrwMoYDsnCLQXdZlFglF92WE" # IMPORTANT: Get your key from Google Cloud Console (enable Generative Language API)

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


# --- Historical Data Loading (yfinance) - ENHANCED ROBUSTNESS ---
@st.cache_data(ttl=3600, show_spinner=False)  # Cache historical data for 1 hour
def load_historical_data(ticker_symbol):
    """
    Loads historical stock data using yfinance.
    Attempts to fetch max historical data, with fallbacks for shorter periods if max fails.
    """
    if not ticker_symbol:
        return pd.DataFrame()

    periods_to_try = ["max", "5y", "2y", "1y", "6mo", "3mo", "1mo"] # Ordered from longest to shortest

    for period in periods_to_try:
        try:
            # Use st.info within spinner to show progress to user
            with st.spinner(f"Attempting to load historical data for {ticker_symbol} (period: {period})..."):
                ticker = yf.Ticker(ticker_symbol)
                # auto_adjust=True automatically adjusts prices for splits and dividends
                hist_df = ticker.history(period=period, auto_adjust=True, timeout=15)

            if hist_df.empty:
                print(f"yfinance returned empty data for {ticker_symbol} with period '{period}'. Trying next period.")
                continue # Try the next shorter period

            # If data is found, process it
            hist_df.reset_index(inplace=True)
            hist_df.rename(columns={
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close', # 'Close' will be adjusted due to auto_adjust=True
                'Volume': 'Volume'
            }, inplace=True)

            hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.date
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            hist_df = hist_df[required_cols]
            hist_df.sort_values(by='Date', ascending=True, inplace=True)
            hist_df.reset_index(drop=True, inplace=True)
            st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using period '{period}'.")
            return hist_df # Return on first successful load

        except requests.exceptions.RequestException as req_err:
            # This catches network-related errors from 'requests' (used by yfinance internally)
            st.warning(f"⚠️ Network error while fetching {ticker_symbol} for period '{period}': {req_err}. Trying next period.")
            print(f"DEBUG: Network error for {ticker_symbol} ({period}): {req_err}")
            time.sleep(1) # Small delay before retrying period in case of temporary network issue
            continue # Try next period on network errors
        except Exception as e:
            # This catches the 'Expecting value' (JSON parsing) or 'No timezone' (data format)
            # and other unexpected errors from yfinance/underlying data.
            st.warning(f"⚠️ Failed to load historical data for {ticker_symbol} (period: {period}): {e}. This often indicates an issue with the data source's response, an invalid ticker, or temporary data unavailability. Trying next period.")
            print(f"DEBUG: Generic yfinance error for {ticker_symbol} ({period}): {e}")
            time.sleep(1) # Small delay before retrying period
            continue # Try next period on other exceptions

    # If all periods fail
    st.error(f"❌ Failed to load historical data for {ticker_symbol} after multiple attempts and periods. Please double-check the ticker symbol (e.g., AAPL for US, RELIANCE.NS for NSE, TCS.BO for BSE for Indian stocks) or try again later. The data source might be temporarily unavailable for this symbol.")
    return pd.DataFrame()


# --- Streamlit Application Main Layout ---
def main():
    st.markdown("""
        <div class="main-header">
            <h1>📈 Intelligent Stock Insights</h1>
            <p>Your Comprehensive AI-Powered Stock Analysis Dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Enter a Stock Ticker to Begin Analysis</h3>", unsafe_allow_html=True)

    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = ""
    if 'analyze_triggered' not in st.session_state:
        st.session_state.analyze_triggered = False

    ticker_input_key = "main_ticker_search_input"
    ticker_input_value = st.session_state.current_ticker

    ticker_input = st.text_input(
        "Search Stock Ticker (e.g., AAPL, RELIANCE.NS, TCS.BO)",
        value=ticker_input_value,
        key=ticker_input_key,
        help="Type a few letters to see suggestions. Press Enter to analyze. For Indian stocks, use .NS for NSE (e.g., RELIANCE.NS) and .BO for BSE (e.g., TCS.BO)."
    )

    suggestions = []
    if ticker_input:
        if FMP_API_KEY == "YOUR_FMP_KEY":
            st.warning("⚠️ FMP_API_KEY is not set. Autocomplete suggestions may be limited or unavailable. Please update `app.py`.")
        else:
            suggestions = fmp_autocomplete.fetch_fmp_suggestions(ticker_input, api_key=FMP_API_KEY)

    if suggestions:
        st.markdown("<h5>Suggestions:</h5>", unsafe_allow_html=True)
        num_columns_to_create = min(len(suggestions), 5)

        if num_columns_to_create > 0:
            cols = st.columns(num_columns_to_create)
            for i, suggestion in enumerate(suggestions):
                if i < len(cols):
                    with cols[i]:
                        suggested_ticker = suggestion.split(' - ')[0].strip().upper()
                        if st.button(suggestion, key=f"suggestion_{i}"):
                            st.session_state.current_ticker = suggested_ticker
                            st.session_state.analyze_triggered = True
                            st.rerun()

    if st.button("🚀 Analyze Stock", key="analyze_button", type="primary"):
        if ticker_input:
            st.session_state.current_ticker = ticker_input.split(' - ')[0].strip().upper()
            st.session_state.analyze_triggered = True
        else:
            st.warning("Please enter a stock ticker to analyze.")
            st.session_state.analyze_triggered = False
        st.rerun()

    if st.session_state.analyze_triggered and st.session_state.current_ticker:
        ticker_to_analyze = st.session_state.current_ticker

        st.markdown(f"<h2 class='analysis-header'>Comprehensive Analysis for {ticker_to_analyze}</h2>",
                    unsafe_allow_html=True)

        # Load historical data first, as it's a prerequisite for multiple tabs
        # This block now uses the enhanced load_historical_data
        if 'historical_data' not in st.session_state or st.session_state.historical_data is None or \
           st.session_state.historical_data.empty or \
           (hasattr(st.session_state.historical_data, 'name') and st.session_state.historical_data.name != ticker_to_analyze):
            # The spinner is now handled inside load_historical_data for each period attempt
            st.session_state.historical_data = load_historical_data(ticker_to_analyze)
            if not st.session_state.historical_data.empty:
                st.session_state.historical_data.name = ticker_to_analyze # Store ticker with data

        hist_data_for_tabs = st.session_state.historical_data

        if hist_data_for_tabs.empty:
            # If historical data is still empty after all attempts, show a final error and stop analysis
            st.error(f"❌ Analysis cannot proceed for {ticker_to_analyze}: Historical data could not be retrieved. Please check the ticker symbol and try again.")
            st.session_state.analyze_triggered = False # Reset trigger if data is missing
            return

        tab_summary, tab_financials, tab_probabilistic, tab_forecast, tab_news = st.tabs([
            "Company Overview", "Financials", "Probabilistic Models", "Forecasting", "News Sentiment"
        ])

        # --- Pass relevant data and API keys to each module ---
        with tab_summary:
            if FMP_API_KEY == "YOUR_FMP_KEY":
                st.warning("⚠️ FMP_API_KEY is not set. Company overview might be incomplete (relying solely on yfinance) and financial data/news company name lookup will be unavailable.")
            if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
                st.warning("⚠️ GEMINI_API_KEY is not set. AI-powered company insights will be unavailable. Please update `app.py`.")
            stock_summary.display_stock_summary(ticker_to_analyze, fmp_api_key=FMP_API_KEY, gemini_api_key=GEMINI_API_KEY)

        with tab_financials:
            if FMP_API_KEY == "YOUR_FMP_KEY":
                st.error("❌ FMP_API_KEY is not set. Financial statements cannot be loaded. Please set your FMP_API_KEY in app.py.")
            else:
                financials.display_financials(ticker_to_analyze, fmp_api_key=FMP_API_KEY)

        with tab_probabilistic:
            probabilistic_stock_model.display_probabilistic_models(hist_data_for_tabs)

        with tab_news:
            if NEWS_API_KEY == "YOUR_NEWSAPI_KEY" or FMP_API_KEY == "YOUR_FMP_KEY":
                st.error("❌ NewsAPI_KEY or FMP_API_KEY is not set. News sentiment analysis will not work. Please set your API keys in app.py.")
            else:
                news_sentiment.display_news_sentiment(ticker_to_analyze, news_api_key=NEWS_API_KEY, fmp_api_key=FMP_API_KEY)

        with tab_forecast:
            forecast_module.display_forecasting(hist_data_for_tabs)


if __name__ == "__main__":
    main()

