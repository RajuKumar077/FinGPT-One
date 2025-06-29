import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import time
from retrying import retry
from pages.yahoo_autocomplete import fetch_ticker_suggestions
from pages.stock_summary import fetch_stock_data, display_stock_summary
from pages.probabilistic_stock_model import display_probabilistic_models
from pages.news_sentiment import display_news_sentiment
from pages.forecast_module import display_forecasting
from pages.financials import display_financials
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# API Keys (use environment variables for security)
FMP_API_KEY = os.getenv("FMP_API_KEY", "5C9DnMCAzYam2ZPjNpOxKLFxUiGhrJDD")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "874ba654bdcd4aa7b68f7367a907cc2f")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "8UU32LX81NSED6CM")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAK8BevJ1wIrwMoYDsnCLQXdZlFglF92WE")

# Streamlit configuration
st.set_page_config(
    page_title="FinGPT One - Stock Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Custom CSS for consistent styling
st.markdown("""
<style>
    .main { background-color: #1E1E1E; }
    .section-title { color: #00ACC1; font-size: 24px; font-weight: bold; margin-bottom: 10px; }
    .section-subtitle { color: #B0BEC5; font-size: 18px; font-weight: bold; margin-top: 20px; }
    .metric-card { 
        background-color: #2D2D2D; 
        padding: 15px; 
        border-radius: 10px; 
        text-align: center; 
        margin-bottom: 10px;
    }
    .metric-card .card-icon { font-size: 24px; margin-bottom: 5px; }
    .metric-card .card-title { font-size: 14px; color: #B0BEC5; margin-bottom: 5px; }
    .metric-card .card-value { font-size: 20px; font-weight: bold; color: #00ACC1; }
    .news-card { 
        background-color: #2D2D2D; 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 15px;
    }
    .news-link { color: #00ACC1; text-decoration: none; }
    .news-link:hover { text-decoration: underline; }
    .sidebar .sidebar-content { background-color: #2D2D2D; }
    h1 { color: #00ACC1; font-family: 'Inter', sans-serif; }
    p { color: #B0BEC5; font-family: 'Inter', sans-serif; }
</style>
<link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap' rel='stylesheet'>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_data(ticker_symbol, fmp_api_key, retries=3, initial_delay=0.5):
    """
    Loads historical stock data using yfinance as primary source and FMP as fallback.

    Args:
        ticker_symbol (str): Stock ticker.
        fmp_api_key (str): FMP API key.
        retries (int): Number of retry attempts.
        initial_delay (float): Initial retry delay in seconds.

    Returns:
        pd.DataFrame: Historical data with Date (index), Open, High, Low, Close, Volume.
    """
    if not ticker_symbol or not isinstance(ticker_symbol, str):
        st.error("❌ Invalid ticker symbol.")
        logging.error("Invalid ticker symbol provided")
        return pd.DataFrame()

    # Track FMP API calls
    if 'fmp_calls' not in st.session_state:
        st.session_state.fmp_calls = 0

    # Attempt 1: yfinance
    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                time.sleep(initial_delay * (2 ** (attempt - 1)))
            with st.spinner(f"Fetching historical data for {ticker_symbol} via yfinance..."):
                ticker = yf.Ticker(ticker_symbol)
                hist_df = ticker.history(period="max", auto_adjust=True, timeout=15)
                if not hist_df.empty:
                    hist_df.reset_index(inplace=True)
                    hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.date
                    hist_df = hist_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    hist_df = hist_df.sort_values('Date').set_index('Date')
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
                    hist_df = hist_df.dropna()
                    if not hist_df.empty:
                        st.success(f"✅ Loaded historical data for {ticker_symbol} via yfinance.")
                        logging.info(f"yfinance success for {ticker_symbol}: {len(hist_df)} rows")
                        return hist_df
                break
        except Exception as e:
            if attempt == retries:
                st.warning(f"⚠️ yfinance failed for {ticker_symbol}: {e}. Trying FMP...")
                logging.warning(f"yfinance failed for {ticker_symbol}: {e}")

    # Attempt 2: FMP historical-price-full
    if not fmp_api_key or fmp_api_key == "YOUR_FMP_KEY":
        st.error("❌ FMP API key is missing. Cannot fetch historical data.")
        logging.error("FMP API key missing")
        return pd.DataFrame()

    if st.session_state.fmp_calls >= 240:
        st.warning("⚠️ Approaching FMP API daily limit (250 requests). Consider pausing or upgrading your plan.")
        logging.warning("Approaching FMP API daily limit")

    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_symbol}"
    params = {"apikey": fmp_api_key, "timeseries": 1260}  # Limit to ~5 years

    @retry(stop_max_attempt_number=retries, wait_exponential_multiplier=int(initial_delay * 1000))
    def fetch_fmp():
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()

    try:
        with st.spinner(f"Fetching historical data for {ticker_symbol} via FMP..."):
            st.session_state.fmp_calls += 1
            data = fetch_fmp()
            if data and "historical" in data and data["historical"]:
                df = pd.DataFrame(data["historical"])
                df['date'] = pd.to_datetime(df['date']).dt.date
                df = df.rename(columns={
                    'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low',
                    'close': 'Close', 'volume': 'Volume'
                })
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df = df.sort_values('Date').set_index('Date')
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna()
                if not df.empty:
                    st.success(f"✅ Loaded historical data for {ticker_symbol} via FMP.")
                    logging.info(f"FMP success for {ticker_symbol}: {len(df)} rows")
                    return df
            else:
                st.warning(f"⚠️ FMP returned no historical data for {ticker_symbol}.")
                logging.warning(f"FMP no data for {ticker_symbol}: {data}")
                return pd.DataFrame()
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 429:
            st.error(f"⚠️ FMP API rate limit reached (250 requests/day) for {ticker_symbol}. Please wait 24 hours or upgrade your plan.")
            logging.error(f"FMP rate limit reached for {ticker_symbol}")
        elif http_err.response.status_code in [401, 403]:
            st.error(f"❌ FMP API key unauthorized for {ticker_symbol}. Verify at https://financialmodelingprep.com.")
            logging.error(f"FMP API key unauthorized for {ticker_symbol}")
        else:
            st.error(f"⚠️ FMP HTTP error for {ticker_symbol}: {http_err} (Status: {http_err.response.status_code})")
            logging.error(f"FMP HTTP error for {ticker_symbol}: {http_err}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ FMP error for {ticker_symbol}: {e}")
        logging.error(f"FMP error for {ticker_symbol}: {e}")
        return pd.DataFrame()

    st.error(f"""
    ❌ Failed to load historical data for {ticker_symbol} from all online sources.
    Possible reasons:
    - Invalid ticker symbol (check spelling, e.g., 'AAPL' or 'RELIANCE.NS').
    - API rate limits exceeded (wait 24 hours for FMP).
    - Invalid or unauthorized FMP API key (verify at https://financialmodelingprep.com).
    - Network issues (check your internet connection).
    Please try again or use a different ticker.
    """)
    logging.error(f"All APIs failed for {ticker_symbol}")
    return pd.DataFrame()

def main():
    """Main function for the FinGPT One app."""
    st.markdown("""
        <h1>📈 FinGPT One - Stock Analysis Dashboard</h1>
        <p>Comprehensive AI-powered stock analysis tool</p>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("FinGPT One")
    st.sidebar.markdown("### Stock Selection")
    if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
        st.sidebar.warning("⚠️ Alpha Vantage API key is missing. Ticker suggestions will not work.")
    query = st.sidebar.text_input("Enter ticker or company name:", value="AAPL", key="ticker_input")
    suggestions = fetch_ticker_suggestions(query, ALPHA_VANTAGE_API_KEY) if query else []
    selected_suggestion = st.sidebar.selectbox("Select a stock:", [""] + suggestions, key="ticker_select")
    ticker = selected_suggestion.split(" - ")[0].strip().upper() if selected_suggestion else query.strip().upper()

    # Troubleshooting section
    with st.sidebar.expander("ℹ️ Troubleshooting"):
        st.markdown("""
        If data loading fails:
        - **Invalid Ticker**: Use correct format (e.g., 'AAPL', 'RELIANCE.NS').
        - **Rate Limits**: Wait 24 hours if FMP limit (250/day) is reached.
        - **API Keys**: Verify FMP key at https://financialmodelingprep.com.
        - **Network**: Check your internet connection.
        Try a different ticker or retry later.
        """)

    page = st.sidebar.radio("Go to:", [
        "Stock Summary",
        "Probabilistic Models",
        "News Sentiment",
        "Forecasting",
        "Financial Statements"
    ], key="page_select")

    # Validate ticker
    if not ticker:
        st.warning("⚠️ Please enter or select a valid ticker to proceed.")
        return

    # Initialize session state
    if 'current_ticker' not in st.session_state or st.session_state.current_ticker != ticker:
        st.session_state.current_ticker = ticker
        st.session_state.historical_data = None

    # Load historical data for relevant pages
    hist_data = None
    if page in ["Stock Summary", "Probabilistic Models", "Forecasting"]:
        if (st.session_state.get('historical_data') is None or
            st.session_state.historical_data.empty or
            st.session_state.get('historical_data_ticker') != ticker):
            hist_data = load_historical_data(ticker, FMP_API_KEY)
            st.session_state.historical_data = hist_data
            st.session_state.historical_data_ticker = ticker
        else:
            hist_data = st.session_state.historical_data

    # Display selected page
    st.markdown(f"<h2 class='section-title'>Analysis for {ticker}</h2>", unsafe_allow_html=True)
    if page == "Stock Summary":
        if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
            st.warning("⚠️ Alpha Vantage API key is missing. News and insights may be unavailable.")
        if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
            st.warning("⚠️ Gemini API key is missing. AI insights will be unavailable.")
        display_stock_summary(ticker, hist_data, FMP_API_KEY, ALPHA_VANTAGE_API_KEY, GEMINI_API_KEY)
    elif page == "Probabilistic Models":
        if hist_data.empty:
            st.error("❌ No historical data available for probabilistic models.")
        else:
            display_probabilistic_models(hist_data)
    elif page == "News Sentiment":
        if NEWS_API_KEY == "YOUR_NEWSAPI_KEY":
            st.error("❌ NewsAPI key is missing. News sentiment analysis is unavailable.")
        else:
            display_news_sentiment(ticker, NEWS_API_KEY, FMP_API_KEY)
    elif page == "Forecasting":
        if hist_data.empty:
            st.error("❌ No historical data available for forecasting.")
        else:
            display_forecasting(hist_data, ticker)
    elif page == "Financial Statements":
        if FMP_API_KEY == "YOUR_FMP_KEY":
            st.error("❌ FMP API key is missing. Financial statements are unavailable.")
        else:
            display_financials(ticker, FMP_API_KEY)

if __name__ == "__main__":
    main()
