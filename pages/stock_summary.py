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
    Fetches historical stock data and company profile using yfinance and FMP APIs.
    
    Args:
        ticker_symbol (str): Stock ticker.
        alpha_vantage_api_key (str): Alpha Vantage API key.
        fmp_api_key (str): FMP API key.
        retries (int): Number of retry attempts.
        initial_delay (float): Initial retry delay in seconds.
    
    Returns:
        tuple: (pd.DataFrame: historical data, dict: company profile or None).
    """
    if not ticker_symbol or not isinstance(ticker_symbol, str):
        st.error("❌ Invalid ticker symbol.")
        return pd.DataFrame(), None

    # Attempt 1: yfinance for historical data
    hist_df = pd.DataFrame()
    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                time.sleep(initial_delay * (2 ** (attempt - 1)))
            with st.spinner(f"Fetching historical data for {ticker_symbol} via yfinance..."):
                ticker = yf.Ticker(ticker_symbol)
                hist_df = ticker.history(period="5y", auto_adjust=True, timeout=15)
                if not hist_df.empty:
                    hist_df.reset_index(inplace=True)
                    hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.date
                    hist_df = hist_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    hist_df = hist_df.sort_values('Date').set_index('Date')
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
                    hist_df = hist_df.dropna()
                    if not hist_df.empty:
                        break
        except Exception as e:
            if attempt == retries:
                st.warning(f"⚠️ yfinance failed for {ticker_symbol}: {e}. Trying FMP...")

    # Fallback to FMP if yfinance fails
    if hist_df.empty and fmp_api_key and fmp_api_key != "YOUR_FMP_KEY":
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_symbol}"
        params = {"apikey": fmp_api_key}
        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    time.sleep(initial_delay * (2 ** (attempt - 1)))
                with st.spinner(f"Fetching historical data for {ticker_symbol} via FMP..."):
                    response = requests.get(url, params=params, timeout=20)
                    response.raise_for_status()
                    data = response.json()
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
                        hist_df = df.dropna()
                        if not hist_df.empty:
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
                    st.error(f"⚠️ FMP error: {e}")

    # Fetch company profile from FMP
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

    if hist_df.empty:
        st.error(f"❌ No historical data available for {ticker_symbol}.")
    else:
        st.success(f"✅ Loaded {len(hist_df)} days of historical data for {ticker_symbol}.")

    return hist_df, profile

def display_stock_summary():
    """Displays the stock summary page with price data and company profile."""
    ticker = st.session_state.get('current_ticker', 'AAPL')
    st.subheader(f"Stock Summary for {ticker}")

    if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
        st.warning("⚠️ Alpha Vantage API key is missing. News and insights may be unavailable.")
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        st.warning("⚠️ Gemini API key is missing. AI insights will be unavailable.")

    hist_data, profile = fetch_stock_data(ticker, ALPHA_VANTAGE_API_KEY, FMP_API_KEY)

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
            if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
                st.write("AI insights not implemented in this version.")
            else:
                st.write(f"Mock analysis: {profile.get('description')[:100]}... (AI insights unavailable without Gemini API key).")

    # Display price data
    if not hist_data.empty:
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
            title=f"Price History for {ticker}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ No historical data to display.")
