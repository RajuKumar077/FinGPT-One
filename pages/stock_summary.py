import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
import time
from textblob import TextBlob
from dotenv import load_dotenv
import os

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY", "5C9DnMCAzYam2ZPjNpOxKLFxUiGhrJDD")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "8UU32LX81NSED6CM")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAK8BevJ1wIrwMoYDsnCLQXdZlFglF92WE")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker_symbol, alpha_vantage_api_key, fmp_api_key, retries=3, initial_delay=0.5):
    """
    Fetches company profile using FMP API (historical data is handled in app.py).

    Args:
        ticker_symbol (str): Stock ticker.
        alpha_vantage_api_key (str): Alpha Vantage API key.
        fmp_api_key (str): FMP API key.
        retries (int): Number of retry attempts.
        initial_delay (float): Initial retry delay in seconds.

    Returns:
        dict: Company profile or None.
    """
    profile = None
    if fmp_api_key and fmp_api_key != "YOUR_FMP_KEY":
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker_symbol}"
        params = {"apikey": fmp_api_key}
        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    time.sleep(initial_delay * (2 ** (attempt - 1)))
                response = requests.get(url, params=params, timeout=20)
                response.raise_for_status()
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    profile = data[0]
                    break
            except requests.exceptions.HTTPError as http_err:
                if attempt == retries:
                    if http_err.response.status_code == 429:
                        st.error("⚠️ FMP API rate limit reached (250 requests/day).")
                    elif http_err.response.status_code in [401, 403]:
                        st.error("❌ Invalid FMP API key.")
                    else:
                        st.error(f"⚠️ FMP HTTP error: {http_err} (Status: {http_err.response.status_code})")
            except Exception as e:
                if attempt == retries:
                    st.error(f"⚠️ FMP profile error: {e}")
    return profile

def display_stock_summary(ticker_symbol, hist_data, fmp_api_key, alpha_vantage_api_key, gemini_api_key):
    """Displays the stock summary page with price data and company profile."""
    st.subheader(f"Stock Summary for {ticker_symbol}")

    # Fetch company profile
    profile = fetch_stock_data(ticker_symbol, alpha_vantage_api_key, fmp_api_key)

    # Display company profile
    if profile:
        st.markdown("##### Company Profile")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Name**: {profile.get('companyName', 'N/A')}")
            st.markdown(f"**Sector**: {profile.get('sector', 'N/A')}")
            st.markdown(f"**Industry**: {profile.get('industry', 'N/A')}")
        with cols[1]:
            st.markdown(f"**Market Cap**: ${profile.get('mktCap', 'N/A'):,.0f}")
            st.markdown(f"**Exchange**: {profile.get('exchangeShortName', 'N/A')}")
            st.markdown(f"**Website**: [{profile.get('website', 'N/A')}]({profile.get('website', '#')})")
        st.markdown(f"**Description**: {profile.get('description', 'No description available.')}")
        
        # AI Insights (mocked if no Gemini key)
        if profile.get('description'):
            st.markdown("##### AI Insights")
            if gemini_api_key and gemini_api_key != "YOUR_GEMINI_API_KEY":
                st.write("AI insights not implemented in this version.")
            else:
                st.write(f"Mock analysis: {profile.get('description')[:100]}... (AI insights unavailable without Gemini API key).")

    # Display price data
    if hist_data is None or hist_data.empty:
        st.error("❌ No historical data to display.")
        return

    st.markdown("##### Price Data")
    latest_data = hist_data.iloc[-1]
    prev_data = hist_data.iloc[-2] if len(hist_data) > 1 else latest_data
    price_change = latest_data['Close'] - prev_data['Close']
    price_change_pct = (price_change / prev_data['Close'] * 100) if prev_data['Close'] != 0 else 0

    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"**Current Price**: ${latest_data['Close']:.2f}")
    with cols[1]:
        st.markdown(f"**Price Change**: {price_change:.2f} ({price_change_pct:.2f}%)")
    with cols[2]:
        st.markdown(f"**52-Week High**: ${hist_data['High'].max():.2f}")
    with cols[3]:
        st.markdown(f"**52-Week Low**: ${hist_data['Low'].min():.2f}")

    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#00ACC1')
    ))
    fig.update_layout(
        title=f"Price History for {ticker_symbol}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template='plotly_dark',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
