import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
import requests
import json
import yfinance as yf
from ratelimit import limits, sleep_and_retry
import logging

# Import functions from your separate modules
# Assuming these modules exist in a 'pages' directory
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
    menu_items={'Get help': None, 'Report a bug': None, 'About': None},
    page_icon="📈"
)

# Custom CSS to hide Streamlit hamburger menu and header
st.markdown("""
    <style>
        button[data-testid="stSidebarToggle"] { display: none !important; }
        header { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# API Keys
NEWS_API_KEY = "874ba654bdcd4aa7b68f7367a907cc2f"
FMP_API_KEY = "5C9DnMCAzYam2ZPjNpOxKLFxUiGhrJDD"
GEMINI_API_KEY = "AIzaSyAK8BevJ1wIrwMoYDsnCLQXdZlFglF92WE"
ALPHA_VANTAGE_API_KEY = "WLVUE35CQ906QK3K"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom CSS and Font Loading ---
def load_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: CSS file not found at {file_path}. Please ensure 'assets/style.css' exists.")

st.markdown(
    "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap' rel='stylesheet'>",
    unsafe_allow_html=True)
load_css("assets/style.css")

# --- Validate API Keys ---
def validate_api_key(api_name, url, params):
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        st.warning(f"⚠️ {api_name} API key validation failed: {e}. This may affect data retrieval.")
        logging.error(f"{api_name} API key validation failed: {e}")
        return False

# Validate API keys at startup
fmp_valid = validate_api_key("FMP", "https://financialmodelingprep.com/api/v3/stock/list",
                             {"apikey": FMP_API_KEY, "limit": 1})
alpha_vantage_valid = validate_api_key("Alpha Vantage", "https://www.alphavantage.co/query",
                                      {"function": "GLOBAL_QUOTE", "symbol": "AAPL", "apikey": ALPHA_VANTAGE_API_KEY})

# --- Historical Data Loading (Online APIs Only) ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_data(ticker_symbol, alpha_vantage_api_key, fmp_api_key):
    """
    Loads historical stock data from online APIs (yfinance, Alpha Vantage, FMP).
    No local CSV fallback.
    """
    if not ticker_symbol:
        st.error("❌ No ticker provided.")
        return pd.DataFrame()

    hist_df = pd.DataFrame()
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    # --- Attempt 1: yfinance ---
    st.info(f"Trying to load historical data for {ticker_symbol} using yfinance...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist_df_yf = ticker.history(period="max", auto_adjust=True, timeout=10)
        if not hist_df_yf.empty:
            hist_df_yf.reset_index(inplace=True)
            hist_df_yf['Date'] = pd.to_datetime(hist_df_yf['Date']).dt.date
            hist_df_yf = hist_df_yf.rename(columns={
                'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
            })
            hist_df = hist_df_yf[required_cols].dropna().reset_index(drop=True)
            if not hist_df.empty:
                st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using yfinance.")
                logging.info(f"yfinance success for {ticker_symbol}: {len(hist_df)} rows")
                return hist_df
        else:
            st.warning(f"⚠️ yfinance returned empty data for {ticker_symbol}.")
            logging.warning(f"yfinance empty data for {ticker_symbol}")
    except Exception as e:
        st.warning(f"⚠️ yfinance failed for {ticker_symbol}: {e}")
        logging.warning(f"yfinance failed for {ticker_symbol}: {e}")

    # --- Attempt 2: Alpha Vantage ---
    if alpha_vantage_api_key and alpha_vantage_api_key != "YOUR_ALPHA_VANTAGE_API_KEY" and alpha_vantage_valid:
        @sleep_and_retry
        @limits(calls=5, period=60)  # 5 calls per minute
        def fetch_alpha_vantage():
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker_symbol,
                "outputsize": "full",
                "apikey": alpha_vantage_api_key
            }
            with st.spinner(f"Loading {ticker_symbol} from Alpha Vantage..."):
                try:
                    response = requests.get(url, params=params, timeout=20)
                    response.raise_for_status()
                    data = response.json()
                    if "Time Series (Daily)" in data:
                        df_av = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
                        df_av.index = pd.to_datetime(df_av.index)
                        df_av.sort_index(inplace=True)
                        df_av = df_av.rename(columns={
                            '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
                            '5. adjusted close': 'Close', '6. volume': 'Volume'
                        })
                        df_av.reset_index(inplace=True)
                        df_av['Date'] = df_av['index'].dt.date
                        df_av = df_av[required_cols].dropna().reset_index(drop=True)
                        if not df_av.empty:
                            st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using Alpha Vantage.")
                            logging.info(f"Alpha Vantage success for {ticker_symbol}: {len(df_av)} rows")
                            return df_av
                    else:
                        st.warning(f"⚠️ Alpha Vantage returned no data for {ticker_symbol}: {data.get('Error Message', 'No data')}")
                        logging.warning(f"Alpha Vantage no data for {ticker_symbol}: {data}")
                except requests.exceptions.HTTPError as http_err:
                    st.warning(f"⚠️ Alpha Vantage HTTP error for {ticker_symbol}: {http_err}")
                    logging.warning(f"Alpha Vantage HTTP error for {ticker_symbol}: {http_err}")
                except Exception as e:
                    st.warning(f"⚠️ Alpha Vantage failed for {ticker_symbol}: {e}")
                    logging.warning(f"Alpha Vantage failed for {ticker_symbol}: {e}")
                return pd.DataFrame()

        hist_df = fetch_alpha_vantage()
        if not hist_df.empty:
            return hist_df

    # --- Attempt 3: FMP (historical-price-full) ---
    if fmp_api_key and fmp_api_key != "YOUR_FMP_KEY" and fmp_valid:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_symbol}"
        params = {"apikey": fmp_api_key, "timeseries": 1260}  # Approx 5 years
        with st.spinner(f"Loading {ticker_symbol} from FMP..."):
            try:
                response = requests.get(url, params=params, timeout=20)
                response.raise_for_status()
                data = response.json()
                if "historical" in data and data["historical"]:
                    df_fmp = pd.DataFrame(data["historical"])
                    df_fmp['date'] = pd.to_datetime(df_fmp['date']).dt.date
                    df_fmp = df_fmp.rename(columns={
                        'date': 'Date', 'open': 'Open', 'high': 'High',
                        'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                    })
                    hist_df = df_fmp[required_cols].dropna().reset_index(drop=True)
                    if not hist_df.empty:
                        st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using FMP.")
                        logging.info(f"FMP success for {ticker_symbol}: {len(hist_df)} rows")
                        return hist_df
                else:
                    st.warning(f"⚠️ FMP returned no data for {ticker_symbol}: {data.get('Error Message', 'No data')}")
                    logging.warning(f"FMP no data for {ticker_symbol}: {data}")
            except requests.exceptions.HTTPError as http_err:
                st.warning(f"⚠️ FMP HTTP error for {ticker_symbol}: {http_err}")
                logging.warning(f"FMP HTTP error for {ticker_symbol}: {http_err}")
            except Exception as e:
                st.warning(f"⚠️ FMP failed for {ticker_symbol}: {e}")
                logging.warning(f"FMP failed for {ticker_symbol}: {e}")

    st.error(f"""
    ❌ Failed to load historical data for {ticker_symbol} from all online sources.
    Possible reasons:
    - Invalid ticker symbol (check spelling, e.g., 'AAPL' or 'RELIANCE.NS').
    - API rate limits exceeded (wait 1 minute for Alpha Vantage or 24 hours for FMP).
    - Invalid or unauthorized API keys (verify at https://www.alphavantage.co or https://financialmodelingprep.com).
    - Network issues (check your internet connection).
    Please try again or use a different ticker.
    """)
    logging.error(f"All APIs failed for {ticker_symbol}")
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
        help="Type a few letters to see suggestions. Press Enter to analyze. For Indian stocks, use .NS for NSE (e.g., RELIANCE.NS) or .BO for BSE (e.g., TCS.BO)."
    )

    suggestions = []
    if ticker_input and fmp_valid:
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

        # Load historical data
        if 'historical_data' not in st.session_state or st.session_state.historical_data is None or \
           st.session_state.historical_data.empty or \
           (hasattr(st.session_state.historical_data, 'name') and st.session_state.historical_data.name != ticker_to_analyze):
            st.session_state.historical_data = load_historical_data(ticker_to_analyze, ALPHA_VANTAGE_API_KEY, FMP_API_KEY)
            if not st.session_state.historical_data.empty:
                st.session_state.historical_data.name = ticker_to_analyze

        hist_data_for_tabs = st.session_state.historical_data

        if hist_data_for_tabs.empty:
            st.error(f"❌ Analysis cannot proceed for {ticker_to_analyze}: Historical data could not be retrieved. Please verify the ticker or try again later.")
            st.session_state.analyze_triggered = False
            return

        tab_summary, tab_financials, tab_probabilistic, tab_forecast, tab_news = st.tabs([
            "Company Overview", "Financials", "Probabilistic Models", "Forecasting", "News Sentiment"
        ])

        with tab_summary:
            if not fmp_valid:
                st.warning("⚠️ FMP API key invalid. Company overview may be incomplete.")
            if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
                st.warning("⚠️ Gemini API key not set. AI-powered insights unavailable.")
            stock_summary.display_stock_summary(ticker_to_analyze, fmp_api_key=FMP_API_KEY, gemini_api_key=GEMINI_API_KEY)

        with tab_financials:
            if not fmp_valid:
                st.error("❌ FMP API key invalid. Financial statements cannot be loaded.")
            else:
                financials.display_financials(ticker_to_analyze, fmp_api_key=FMP_API_KEY)

        with tab_probabilistic:
            probabilistic_stock_model.display_probabilistic_models(hist_data_for_tabs)

        with tab_news:
            if NEWS_API_KEY == "YOUR_NEWSAPI_KEY" or not fmp_valid:
                st.error("❌ NewsAPI or FMP API key invalid. News sentiment analysis unavailable.")
            else:
                news_sentiment.display_news_sentiment(ticker_to_analyze, news_api_key=NEWS_API_KEY, fmp_api_key=FMP_API_KEY)

        with tab_forecast:
            forecast_module.display_forecasting(hist_data_for_tabs)

if __name__ == "__main__":
    main()
