import os
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv

# --- 1. HYBRID SECRETS LOADING ---
load_dotenv()

def get_secret(key_name):
    """Safely fetch secrets from Streamlit Cloud or local .env without crashing."""
    try:
        # Check Streamlit's native secrets first
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    # Fallback to local environment variables
    return os.getenv(key_name)

# Assign keys
TIINGO_API_KEY = get_secret("TIINGO_API_KEY")
FMP_API_KEY = get_secret("FMP_API_KEY")
ALPHA_VANTAGE_API_KEY = get_secret("ALPHA_VANTAGE_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

# --- 2. COMPONENT IMPORTS ---
try:
    from Components.stock_summary import display_stock_summary
    from Components.probabilistic_stock_model import display_probabilistic_models
    from Components.forecast_module import display_forecasting
    from Components.financials import display_financials
except ImportError as e:
    st.error(f"⚠️ Component Import Error: {e}")

# --- 3. APP CONFIGURATION ---
st.set_page_config(page_title="FinGPT One - Pro", layout="wide", page_icon="📈")

# --- 4. THE DATA ENGINE ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(symbol):
    """Multi-layer failover data engine."""
    # LAYER 1: TIINGO
    if TIINGO_API_KEY and TIINGO_API_KEY != "your_tiingo_code_here":
        try:
            url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?token={TIINGO_API_KEY}"
            res = requests.get(url, timeout=5).json()
            if isinstance(res, list) and len(res) > 0:
                df = pd.DataFrame(res)
                df['date'] = pd.to_datetime(df['date']).dt.date
                df = df.rename(columns={'adjClose': 'Close', 'adjHigh': 'High', 'adjLow': 'Low', 'adjOpen': 'Open', 'adjVolume': 'Volume'})
                return df.set_index('date').sort_index(), "Tiingo Institutional"
        except: pass

    # LAYER 2: ALPHA VANTAGE
    if ALPHA_VANTAGE_API_KEY:
        try:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
            data = requests.get(url, timeout=5).json()
            if "Time Series (Daily)" in data:
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                df.index = pd.to_datetime(df.index).date
                df = df.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "6. volume": "Volume"}).astype(float)
                return df.sort_index(), "Alpha Vantage"
        except: pass

    # LAYER 3: YFINANCE (The most reliable backup)
    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period="2y")
        if not df.empty:
            df.index = df.index.date
            return df, "Yahoo Finance"
    except: pass

    return pd.DataFrame(), None

# --- 5. MAIN INTERFACE ---
def main():
    st.sidebar.title("💎 FinGPT One")
    query = st.sidebar.text_input("Enter Ticker:", value="AAPL").strip().upper()
    
    page = st.sidebar.radio("Dashboard Navigation", [
        "Stock Summary", "Forecasting", "Probabilistic Models", "Financials"
    ])

    if query:
        if 'data' not in st.session_state or st.session_state.get('last_ticker') != query:
            with st.spinner(f"Connecting to Markets for {query}..."):
                df, src = fetch_stock_data(query)
                st.session_state.data = df
                st.session_state.source = src
                st.session_state.last_ticker = query

        hist_data = st.session_state.data
        source = st.session_state.source

        if not hist_data.empty:
            st.sidebar.info(f"Connected via {source}")
            
            if page == "Stock Summary":
                display_stock_summary(query, hist_data, FMP_API_KEY, ALPHA_VANTAGE_API_KEY, GEMINI_API_KEY)
            elif page == "Forecasting":
                display_forecasting(hist_data, query)
            elif page == "Probabilistic Models":
                display_probabilistic_models(hist_data)
            elif page == "Financials":
                display_financials(query)
        else:
            st.error(f"Could not find data for {query}.")

if __name__ == "__main__":
    main()