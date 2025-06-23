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
FMP_API_KEY = "5C9DnMCAzYam2ZPjNpOxKLFxUiGhrJDD"     # Your provided FMP key
GEMINI_API_KEY = "AIzaSyAK8BevJ1wIrwMoYDsnCLQXdZlFglF92WE" # Your provided Gemini key
ALPHA_VANTAGE_API_KEY = "WLVUE35CQ906QK3K" # IMPORTANT: Get your free key from www.alphavantage.co


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


# --- Historical Data Loading (yfinance and Alpha Vantage Fallback) ---
@st.cache_data(ttl=3600, show_spinner=False)  # Cache historical data for 1 hour
def load_historical_data(ticker_symbol, alpha_vantage_api_key):
    """
    Loads historical stock data, first attempting yfinance, then falling back to Alpha Vantage.
    """
    if not ticker_symbol:
        return pd.DataFrame()

    hist_df = pd.DataFrame()

    # --- Attempt 1: Try yfinance with multiple periods ---
    st.info(f"Trying to load historical data for {ticker_symbol} using yfinance (primary source)...")
    periods_to_try_yf = ["max", "5y", "2y", "1y", "6mo", "3mo", "1mo"] # Ordered from longest to shortest

    for period in periods_to_try_yf:
        try:
            with st.spinner(f"Attempting yfinance for {ticker_symbol} (period: {period})..."):
                ticker = yf.Ticker(ticker_symbol)
                hist_df_yf = ticker.history(period=period, auto_adjust=True, timeout=15)

            if not hist_df_yf.empty:
                hist_df_yf.reset_index(inplace=True)
                hist_df_yf.rename(columns={
                    'Date': 'Date',
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close', # 'Close' will be adjusted due to auto_adjust=True
                    'Volume': 'Volume'
                }, inplace=True)
                hist_df_yf['Date'] = pd.to_datetime(hist_df_yf['Date']).dt.date
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                hist_df = hist_df_yf[required_cols]
                hist_df.sort_values(by='Date', ascending=True, inplace=True)
                hist_df.reset_index(drop=True, inplace=True)
                st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using yfinance (period: '{period}').")
                return hist_df # Return on first successful yfinance load
            else:
                print(f"yfinance returned empty data for {ticker_symbol} with period '{period}'. Trying next yfinance period.")
                continue

        except requests.exceptions.RequestException as req_err:
            print(f"DEBUG: yfinance network error for {ticker_symbol} ({period}): {req_err}")
            st.warning(f"⚠️ YFinance network error for {ticker_symbol} (period: {period}). Trying next yfinance period.")
            time.sleep(1) # Small delay
            continue
        except Exception as e:
            print(f"DEBUG: Generic yfinance error for {ticker_symbol} ({period}): {e}")
            st.warning(f"⚠️ YFinance data issue for {ticker_symbol} (period: {period}): {e}. This often indicates an issue with the data source's response, an invalid ticker, or temporary data unavailability. Trying next period.")
            time.sleep(1) # Small delay
            continue

    # --- Attempt 2: Fallback to Alpha Vantage if yfinance completely failed ---
    st.info(f"YFinance failed for {ticker_symbol}. Falling back to Alpha Vantage...")
    if not alpha_vantage_api_key or alpha_vantage_api_key == "YOUR_ALPHA_VANTAGE_API_KEY":
        st.error("❌ Alpha Vantage API key is not set. Cannot use Alpha Vantage as a fallback. Please update `app.py`.")
        return pd.DataFrame()

    alpha_vantage_url = "https://www.alphavantage.co/query"
    params_av = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker_symbol,
        "outputsize": "full", # or "compact" for last 100 days
        "apikey": alpha_vantage_api_key
    }

    try:
        with st.spinner(f"Attempting to load historical data for {ticker_symbol} from Alpha Vantage... (This may take a moment due to API limits)"):
            # Alpha Vantage free tier has a limit of 5 calls per minute.
            # We add a sleep here before the call, assuming it's the only AV call in sequence,
            # or it's the first call after a period of no AV calls.
            # If multiple AV calls are made in quick succession, this might still hit limits.
            time.sleep(15) # Wait 15 seconds to respect the rate limit (5 calls/min means 12s per call average)

            response_av = requests.get(alpha_vantage_url, params=params_av, timeout=20)
            response_av.raise_for_status()
            data_av = response_av.json()

            if "Time Series (Daily)" in data_av:
                raw_data = data_av["Time Series (Daily)"]
                df_av = pd.DataFrame.from_dict(raw_data, orient="index")
                df_av.index = pd.to_datetime(df_av.index)
                df_av.sort_index(inplace=True)

                # --- IMPROVED ALPHA VANTAGE COLUMN MAPPING ---
                # Explicitly map Alpha Vantage column names to desired DataFrame column names
                column_mapping = {
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close_Unadjusted', # Keep original if needed, but we'll use adjusted
                    '5. adjusted close': 'Close', # This is what we primarily need for 'Close'
                    '6. volume': 'Volume'
                    # '7. dividend amount': 'Dividend_Amount',
                    # '8. split coefficient': 'Split_Coefficient'
                }
                
                # Filter df_av to only include columns we need and rename them
                df_av = df_av[[col for col in column_mapping.keys() if col in df_av.columns]]
                df_av = df_av.rename(columns=column_mapping)
                
                # Convert relevant columns to numeric, coercing errors
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df_av.columns:
                        df_av[col] = pd.to_numeric(df_av[col], errors='coerce')
                
                df_av.reset_index(inplace=True)
                df_av.rename(columns={'index': 'Date'}, inplace=True)
                df_av['Date'] = df_av['Date'].dt.date

                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                # Ensure all required columns exist and then drop NaNs
                # Fill missing required columns with NaN if they don't exist, then drop rows with any NaNs in required_cols
                for col in required_cols:
                    if col not in df_av.columns:
                        df_av[col] = np.nan
                
                hist_df = df_av[required_cols].dropna().reset_index(drop=True)

                if not hist_df.empty:
                    st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using Alpha Vantage.")
                    return hist_df
                else:
                    st.warning(f"⚠️ Alpha Vantage returned empty or malformed data for {ticker_symbol} after processing. No historical data available.")

            elif "Error Message" in data_av:
                st.error(f"❌ Alpha Vantage API Error for {ticker_symbol}: {data_av['Error Message']}. Please check your API key or usage limits.")
            else:
                st.error(f"❌ Alpha Vantage returned unexpected data format for {ticker_symbol}. Raw response keys: {list(data_av.keys()) if isinstance(data_av, dict) else 'Not a dict'}")

    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 429: # Alpha Vantage specific rate limit
            st.error(f"❌ Alpha Vantage API rate limit hit for {ticker_symbol}. Please wait at least 1 minute before trying another stock.")
        elif http_err.response.status_code in [401, 403]:
            st.error("❌ Alpha Vantage API key is invalid or unauthorized. Please check your ALPHA_VANTAGE_API_KEY in app.py.")
        else:
            st.error(f"❌ Alpha Vantage HTTP error occurred: {http_err}. Status: {http_err.response.status_code}")
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"❌ Alpha Vantage Connection error: {conn_err}. Please check your internet connection.")
    except requests.exceptions.Timeout as timeout_err:
        st.error(f"❌ Alpha Vantage Request timed out. The server might be slow or unresponsive. Please try again.")
    except json.JSONDecodeError as json_err:
        st.error(f"❌ Alpha Vantage: Received invalid JSON data from API. Please try again later. Error: {json_err}")
    except KeyError as ke: # Catch potential KeyError if expected columns are missing
        st.error(f"❌ Alpha Vantage: Data parsing error - expected column not found. Error: {ke}. This may indicate a change in API response format.")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while fetching from Alpha Vantage: {e}")

    # If both yfinance and Alpha Vantage fail
    st.error(f"❌ Historical data for {ticker_symbol} could not be retrieved from any source. Please double-check the ticker symbol (e.g., AAPL for US, RELIANCE.NS for NSE, TCS.BO for BSE for Indian stocks) or try again later.")
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
        if FMP_API_KEY == "YOUR_FMP_KEY": # This check remains, but FMP_API_KEY is now set above
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
        # This block now uses the enhanced load_historical_data with Alpha Vantage fallback
        if 'historical_data' not in st.session_state or st.session_state.historical_data is None or \
           st.session_state.historical_data.empty or \
           (hasattr(st.session_state.historical_data, 'name') and st.session_state.historical_data.name != ticker_to_analyze):
            st.session_state.historical_data = load_historical_data(ticker_to_analyze, ALPHA_VANTAGE_API_KEY)
            if not st.session_state.historical_data.empty:
                st.session_state.historical_data.name = ticker_to_analyze # Store ticker with data

        hist_data_for_tabs = st.session_state.historical_data

        if hist_data_for_tabs.empty:
            st.error(f"❌ Analysis cannot proceed for {ticker_to_analyze}: Historical data could not be retrieved from any source. Please verify the ticker or try again later.")
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
