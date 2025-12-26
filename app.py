from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
import yfinance as yf
import time
import logging

# Import Custom Components
# Note: Ensure these components are updated to their "Free" versions
from Components.stock_summary import display_stock_summary
from Components.probabilistic_stock_model import display_probabilistic_models
from Components.news_sentiment import display_news_sentiment
from Components.forecast_module import display_forecasting
from Components.financials import display_financials_tab  # Updated name from previous step

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load API keys (Keeping only what's necessary)
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "874ba654bdcd4aa7b68f7367a907cc2f")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAK8BevJ1wIrwMoYDsnCLQXdZlFglF92WE")
# FMP and Alpha Vantage are now optional/backup only

# Streamlit configuration
st.set_page_config(
    page_title="FinGPT One - Stock Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_data(ticker_symbol, retries=3):
    """Primary data loader using yfinance (Free/Lifetime)."""
    if not ticker_symbol:
        return pd.DataFrame()

    for attempt in range(retries + 1):
        try:
            # yfinance is robust for historical OHLCV data
            df = yf.download(ticker_symbol, period="5y", interval="1d")
            if not df.empty:
                # Handle yfinance multi-index columns if they exist
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed for {ticker_symbol}: {e}")
            time.sleep(1)
    return pd.DataFrame()

def main():
    st.sidebar.title("FinGPT One")
    
    # 🔍 Ticker Selection
    query = st.sidebar.text_input("Enter Ticker (e.g. TSLA, NVDA):", value="AAPL").strip().upper()
    
    # 🧭 Navigation
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

    # Data Orchestration
    if 'hist_data' not in st.session_state or st.session_state.get('last_ticker') != query:
        with st.spinner(f"Loading {query} data..."):
            st.session_state.hist_data = load_historical_data(query)
            st.session_state.last_ticker = query

    hist_data = st.session_state.hist_data

    # 🚀 Routing Logic (Updated to match Free Components)
    if page == "Stock Summary":
        # Removed FMP_API_KEY requirement
        display_stock_summary(query, hist_data)
    
    elif page == "Probabilistic Models":
        if not hist_data.empty:
            display_probabilistic_models(hist_data)
        else:
            st.error("Historical data required for ML models.")

    elif page == "News Sentiment":
        # News still benefits from an API key, but uses free fallbacks if needed
        display_news_sentiment(query, NEWS_API_KEY)

    elif page == "Forecasting":
        if not hist_data.empty:
            display_forecasting(hist_data, query)

    elif page == "Financial Statements":
        # This now triggers the yahooquery logic (Free/No Key)
        display_financials_tab(query)

if __name__ == "__main__":
    main()
