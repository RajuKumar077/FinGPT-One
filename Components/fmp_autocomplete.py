import streamlit as st
import yfinance as yf
from typing import List

# ---------------------------
# 🔍 Ticker Search with yfinance
# ---------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_suggestions(query: str) -> List[str]:
    """
    Fetches stock ticker suggestions using yfinance (FREE, NO API KEY NEEDED).
    
    This is a lightweight replacement for FMP API that:
    - Works globally (US, India NSE/BSE, Canada, etc.)
    - Has no rate limits
    - Requires no API key
    - Uses cached data
    
    Args:
        query (str): Partial ticker symbol or company name
        
    Returns:
        List[str]: List of formatted suggestions (e.g., "AAPL - Apple Inc.")
    """
    
    if not query or not isinstance(query, str):
        st.error("❌ Invalid search query. Please enter a valid ticker.")
        return []
    
    query = query.strip().upper()
    
    if len(query) < 1:
        return []
    
    # Try to fetch ticker info
    try:
        ticker = yf.Ticker(query)
        info = ticker.info
        
        # Check if valid ticker
        if info and 'longName' in info:
            name = info.get('longName', 'N/A')
            exchange = info.get('exchange', 'Unknown')
            suggestions = [f"{query} - {name} ({exchange})"]
            return suggestions
        
        # If direct lookup fails, try partial match search
        suggestions = try_partial_match(query)
        return suggestions if suggestions else []
        
    except Exception as e:
        st.warning(f"⚠️ Ticker search error: {e}")
        return []


def try_partial_match(query: str) -> List[str]:
    """
    Fallback: Try to find matching tickers from common symbols.
    Useful when exact ticker isn't found.
    """
    # Popular global tickers for quick search
    COMMON_TICKERS = {
        # US Tech Giants
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corp.',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'META': 'Meta Platforms Inc.',
        'TSLA': 'Tesla Inc.',
        'NVDA': 'NVIDIA Corp.',
        'AMD': 'Advanced Micro Devices',
        'NFLX': 'Netflix Inc.',
        'PYPL': 'PayPal Holdings',
        
        # Indian Stocks (NSE)
        'TCS': 'Tata Consultancy Services',
        'INFY': 'Infosys Limited',
        'WIPRO': 'Wipro Limited',
        'HCLTECH': 'HCL Technologies',
        'BAJAJFINSV': 'Bajaj Finserv',
        'RELIANCE': 'Reliance Industries',
        'SBIN': 'State Bank of India',
        'ICICIBANK': 'ICICI Bank',
        'AXISBANK': 'Axis Bank',
        'MARUTI': 'Maruti Suzuki India',
        
        # Global Banks
        'JPM': 'JPMorgan Chase',
        'BAC': 'Bank of America',
        'WFC': 'Wells Fargo',
        'GS': 'Goldman Sachs',
    }
    
    suggestions = []
    for symbol, name in COMMON_TICKERS.items():
        if query in symbol or query in name.upper():
            suggestions.append(f"{symbol} - {name}")
    
    return suggestions[:20]  # Limit to 20 suggestions


def display_autocomplete_widget(session_key: str = "selected_ticker"):
    """
    Streamlit widget for ticker autocomplete search.
    Returns the selected ticker symbol.
    """
    st.subheader("🔍 Stock Search")
    
    search_query = st.text_input(
        "Enter ticker symbol or company name:",
        placeholder="e.g., AAPL, TCS, MSFT...",
        key="ticker_search_input"
    )
    
    if search_query:
        suggestions = fetch_ticker_suggestions(search_query)
        
        if suggestions:
            selected = st.selectbox(
                "Available options:",
                suggestions,
                key=session_key
            )
            # Extract ticker from selected option (format: "AAPL - Apple Inc.")
            ticker_symbol = selected.split(' - ')[0]
            return ticker_symbol
        else:
            st.info("😕 No matches found. Try another symbol.")
            return None
    
    return None
