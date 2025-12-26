# Components/fmp_autocomplete.py
import yfinance as yf
import streamlit as st

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_suggestions(query: str) -> list:
    """
    Fetch stock ticker suggestions using yfinance.
    
    Args:
        query (str): Search query (e.g., 'Apple' or 'AAPL').
        
    Returns:
        list: List of formatted ticker suggestions (e.g., "AAPL - Apple Inc.").
    """
    if not query:
        st.error("❌ Please enter a search query for ticker suggestions.")
        return []

    try:
        # yfinance doesn't have a direct autocomplete, we can use tickers from Ticker module
        # We'll use yf.Tickers for multiple symbols
        # Note: limited by what yfinance can fetch
        tickers = yf.Tickers(query)
        suggestions = []
        for symbol, ticker in tickers.tickers.items():
            info = ticker.info
            name = info.get('shortName') or info.get('longName')
            if name:
                suggestions.append(f"{symbol} - {name}")
        if not suggestions:
            st.info(f"No ticker suggestions found for '{query}'.")
        return suggestions
    except Exception as e:
        st.error(f"⚠️ Error fetching ticker suggestions: {e}")
        return []
