from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import time
from retrying import retry
import logging

# Import Custom Components
from Components.yahoo_autocomplete import fetch_ticker_suggestions
from Components.stock_summary import fetch_stock_data, display_stock_summary
from Components.probabilistic_stock_model import display_probabilistic_models
from Components.news_sentiment import display_news_sentiment
from Components.forecast_module import display_forecasting
from Components.financials import display_financials # This now uses yahooquery

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Load API keys from .env file or fallback to defaults
load_dotenv()
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
    h1 { color: #00ACC1; font-family: 'Inter', sans-serif; }
    p { color: #B0BEC5; font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_data(ticker_symbol, fmp_api_key, retries=3, initial_delay=0.5):
    """Loads historical data. yfinance is primary, FMP is backup."""
    if not ticker_symbol:
        return pd.DataFrame()

    # Attempt 1: yfinance (Free & Reliable)
    for attempt in range(retries + 1):
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist_df = ticker.history(period="5y", auto_adjust=True, timeout=15)
            if not hist_df.empty:
                hist_df.reset_index(inplace=True)
                hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.date
                hist_df = hist_df.sort_values('Date').set_index('Date')
                return hist_df
        except Exception as e:
            logging.warning(f"yfinance attempt {attempt} failed for {ticker_symbol}: {e}")
            if attempt < retries: time.sleep(initial_delay * (2 ** attempt))

    # Attempt 2: FMP fallback (Only if key is valid)
    if fmp_api_key and fmp_api_key != "YOUR_FMP_KEY":
        try:
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_symbol}?apikey={fmp_api_key}"
            response = requests.get(url, timeout=15)
            data = response.json()
            if "historical" in data:
                df = pd.DataFrame(data["historical"])
                df['date'] = pd.to_datetime(df['date']).dt.date
                df = df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                return df.set_index('Date').sort_index()
        except Exception as e:
            logging.error(f"FMP fallback failed: {e}")
    
    return pd.DataFrame()

def main():
    st.sidebar.title("FinGPT One")
    
    # Ticker Selection
    query = st.sidebar.text_input("Enter Ticker (e.g. TSLA, RELIANCE.NS):", value="AAPL").strip().upper()
    
    # Page Selection
    page = st.sidebar.radio("Go to:", [
        "Stock Summary",
        "Probabilistic Models",
        "News Sentiment",
        "Forecasting",
        "Financial Statements"
    ])

    if not query:
        st.warning("Please enter a ticker symbol.")
        return

    # Cache handling for historical data
    if 'hist_data' not in st.session_state or st.session_state.get('last_ticker') != query:
        with st.spinner(f"Loading data for {query}..."):
            st.session_state.hist_data = load_historical_data(query, FMP_API_KEY)
            st.session_state.last_ticker = query

    hist_data = st.session_state.hist_data

    # Routing
    if page == "Stock Summary":
        display_stock_summary(query, hist_data, FMP_API_KEY, ALPHA_VANTAGE_API_KEY, GEMINI_API_KEY)
    
    elif page == "Probabilistic Models":
        if not hist_data.empty:
            display_probabilistic_models(hist_data)
        else:
            st.error("No data available for models.")

    elif page == "News Sentiment":
        display_news_sentiment(query, NEWS_API_KEY, FMP_API_KEY)

    elif page == "Forecasting":
        if not hist_data.empty:
            display_forecasting(hist_data, query)
        else:
            st.error("No data available for forecasting.")

    elif page == "Financial Statements":
        # ADVANCED FEATURE: Now uses yahooquery (Free/No Key)
        display_financials(query)

if __name__ == "__main__":
    main()
