# Components/stock_summary.py

import yfinance as yf
import pandas as pd
import streamlit as st

def fetch_stock_data(ticker, period="1y", interval="1d"):
    """
    Fetch historical stock data using yfinance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        period (str): Time period (default '1y', other options: '5d', '1mo', '5y', etc.).
        interval (str): Data interval (default '1d', other options: '1h', '1wk', '1mo').
    
    Returns:
        pd.DataFrame: Historical stock data with Date as index.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            st.warning(f"No data found for ticker '{ticker}'.")
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()


def display_stock_summary(df, ticker):
    """
    Display stock summary metrics and charts in Streamlit.
    
    Args:
        df (pd.DataFrame): Historical stock data from fetch_stock_data.
        ticker (str): Stock ticker symbol.
    """
    if df.empty:
        st.info(f"No stock data available to display for {ticker}.")
        return

    st.markdown(f"### 📊 Stock Summary for {ticker}")

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Latest Close", f"${df['Close'].iloc[-1]:.2f}")
    with col2:
        st.metric("📉 Lowest Close (period)", f"${df['Close'].min():.2f}")
    with col3:
        st.metric("📈 Highest Close (period)", f"${df['Close'].max():.2f}")

    # Price Chart
    st.markdown("#### Closing Price Over Time")
    st.line_chart(df.set_index('Date')['Close'])

    # Volume Chart
    st.markdown("#### Trading Volume Over Time")
    st.bar_chart(df.set_index('Date')['Volume'])
