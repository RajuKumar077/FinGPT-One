import streamlit as st
import pandas as pd
import yfinance as yf
import os
from dotenv import load_dotenv

# Import Components
from Components.stock_summary import display_stock_summary
from Components.financials import display_financials_tab
from Components.probabilistic_stock_model import display_probabilistic_models

# Initial Config
load_dotenv()
st.set_page_config(page_title="FinGPT One", layout="wide", page_icon="📈")

# --- Optimized Data Loader ---
@st.cache_data(ttl=3600)
def get_historical_data(ticker):
    """Fetches 5 years of daily data via yfinance."""
    try:
        # download returns a cleaner format than Ticker.history for multiple indicators
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if df.empty:
            return None
        
        # Flatten MultiIndex Columns (Crucial for newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df
    except Exception:
        return None

def main():
    # Sidebar
    st.sidebar.title("🚀 FinGPT One")
    symbol = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").strip().upper()
    
    menu = st.sidebar.radio("Navigation", [
        "Stock Summary", 
        "Financial Statements", 
        "AI Analysis"
    ])

    if not symbol:
        st.sidebar.error("Please enter a symbol.")
        return

    # Global Data Fetch
    hist_data = get_historical_data(symbol)

    if hist_data is not None:
        # Route to Pages
        if menu == "Stock Summary":
            # Matching the 2-argument signature in stock_summary.py
            display_stock_summary(symbol, hist_data)
            
        elif menu == "Financial Statements":
            display_financials_tab(symbol)
            
        elif menu == "AI Analysis":
            display_probabilistic_models(hist_data)
    else:
        st.error(f"Could not retrieve data for {symbol}. It may be delisted or the ticker is incorrect.")

if __name__ == "__main__":
    main()
