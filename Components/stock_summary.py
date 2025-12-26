# Components/stock_summary.py
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

@st.cache_data(ttl=3600, show_spinner="Fetching stock data...")
def fetch_stock_data(ticker_symbol: str) -> dict:
    """
    Fetch company info and historical price data using yfinance.
    
    Returns:
        dict: {'info': company info dict, 'history': DataFrame of historical prices}
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        history = ticker.history(period="1y")  # 1 year historical data
        history.index = pd.to_datetime(history.index)
        return {'info': info, 'history': history}
    except Exception as e:
        st.error(f"❌ Error fetching data for {ticker_symbol}: {e}")
        return {'info': {}, 'history': pd.DataFrame()}


def display_stock_summary(ticker_symbol: str):
    """Display company profile and stock price chart."""
    st.subheader(f"Stock Summary: {ticker_symbol}")
    
    data = fetch_stock_data(ticker_symbol)
    info = data['info']
    history = data['history']

    # Company profile
    if info:
        st.markdown("##### Company Profile")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Name:** {info.get('longName', 'N/A')}")
            st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
        with cols[1]:
            market_cap = info.get('marketCap', 'N/A')
            try:
                market_cap = f"${int(market_cap):,}" if market_cap != 'N/A' else 'N/A'
            except:
                market_cap = 'N/A'
            st.markdown(f"**Market Cap:** {market_cap}")
            st.markdown(f"**Exchange:** {info.get('exchange', 'N/A')}")
            st.markdown(f"**Website:** [{info.get('website','N/A')}]({info.get('website','')})")
        st.markdown(f"**Description:** {info.get('longBusinessSummary', 'No description available.')}")
    else:
        st.warning("⚠️ Company profile unavailable.")

    # Historical price chart
    if not history.empty:
        st.markdown("##### Price History (1 Year)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history.index,
            y=history['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#00ACC1')
        ))
        fig.update_layout(
            title=f"{ticker_symbol} Closing Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        latest = history['Close'].iloc[-1]
        prev = history['Close'].iloc[-2] if len(history) > 1 else latest
        change = latest - prev
        change_pct = (change / prev * 100) if prev != 0 else 0
        cols = st.columns(3)
        with cols[0]:
            st.markdown(f"**Latest Close:** ${latest:.2f}")
        with cols[1]:
            color = "green" if change >= 0 else "red"
            st.markdown(f"**Change:** <span style='color:{color}'>{change:.2f} ({change_pct:.2f}%)</span>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"**52-Week High/Low:** ${history['High'].max():.2f} / ${history['Low'].min():.2f}")
    else:
        st.warning("⚠️ Historical price data unavailable.")
