from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import time
from retrying import retry

# ⛔ Removed FMP import dependency for financials
from Components.yahoo_autocomplete import fetch_ticker_suggestions
from Components.stock_summary import fetch_stock_data , display_stock_summary
from Components.probabilistic_stock_model import display_probabilistic_models
from Components.news_sentiment import display_news_sentiment
from Components.forecast_module import display_forecasting

# ⬇️ NEW UPDATED FINANCIALS (Yahoo only)
from Components.financials import display_financials

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Load API keys
load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY") 
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Streamlit configuration
st.set_page_config(
    page_title="FinGPT One - Stock Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Sidebar UI
st.sidebar.title("FinGPT One")
st.sidebar.markdown("### Stock Selection")
query = st.sidebar.text_input("Enter ticker or company name:", value="AAPL", key="ticker_input")
suggestions = fetch_ticker_suggestions(query, ALPHA_VANTAGE_API_KEY) if query else []
selected_suggestion = st.sidebar.selectbox("Select a stock:", [""] + suggestions, key="ticker_select")
ticker = selected_suggestion.split(" - ")[0].strip().upper() if selected_suggestion else query.strip().upper()

page = st.sidebar.radio("Go to:", [
    "Stock Summary",
    "Probabilistic Models",
    "News Sentiment",
    "Forecasting",
    "Financial Statements"
], key="page_select")

# Historical loader (same as before)
@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_data(ticker_symbol, fmp_api_key, retries=3, initial_delay=0.5):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="max", auto_adjust=True)
        df = df[["Open","High","Low","Close","Volume"]]
        df.index = pd.to_datetime(df.index)
        return df
    except:
        st.error("❌ Failed to load data. Check ticker.")
        return pd.DataFrame()

# Main App
def main():
    if not ticker:
        st.warning("⚠️ Enter a ticker to continue.")
        return

    hist_data = load_historical_data(ticker, FMP_API_KEY)

    if page == "Stock Summary":
        display_stock_summary(ticker, hist_data, FMP_API_KEY, ALPHA_VANTAGE_API_KEY, GEMINI_API_KEY)

    elif page == "Probabilistic Models":
        display_probabilistic_models(hist_data)

    elif page == "News Sentiment":
        display_news_sentiment(ticker, NEWS_API_KEY, FMP_API_KEY)

    elif page == "Forecasting":
        display_forecasting(hist_data, ticker)

    # ✅ Updated: NO FMP required here
    elif page == "Financial Statements":
        display_financials(ticker)  # <-- FINAL FIX

if __name__ == "__main__":
    main()
