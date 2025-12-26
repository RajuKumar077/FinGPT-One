# Components/stock_summary.py
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

def fetch_stock_info(ticker_symbol):
    """Fetch company info and historical data using yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        hist = ticker.history(period="1y")  # 1-year daily data
        return info, hist
    except Exception as e:
        st.error(f"❌ Error fetching data for {ticker_symbol}: {e}")
        return {}, None

def display_stock_summary(ticker_symbol):
    st.subheader(f"Stock Summary for {ticker_symbol}")
    
    info, hist_data = fetch_stock_info(ticker_symbol)
    
    if info:
        st.markdown("##### Company Profile")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Name**: {info.get('longName', 'N/A')}")
            st.markdown(f"**Sector**: {info.get('sector', 'N/A')}")
            st.markdown(f"**Industry**: {info.get('industry', 'N/A')}")
        with cols[1]:
            market_cap = info.get('marketCap', 'N/A')
            if market_cap != 'N/A':
                market_cap = f"${market_cap:,}"
            st.markdown(f"**Market Cap**: {market_cap}")
            st.markdown(f"**Exchange**: {info.get('exchange', 'N/A')}")
            st.markdown(f"**Website**: [{info.get('website', '#')}]({info.get('website', '#')})")
        st.markdown(f"**Description**: {info.get('longBusinessSummary', 'No description available.')}")
    
    if hist_data is not None and not hist_data.empty:
        st.markdown("##### Price Data")
        latest = hist_data.iloc[-1]
        prev = hist_data.iloc[-2] if len(hist_data) > 1 else latest
        price_change = latest['Close'] - prev['Close']
        price_pct = (price_change / prev['Close'] * 100) if prev['Close'] != 0 else 0
        
        cols = st.columns(4)
        cols[0].markdown(f"**Current Price**: ${latest['Close']:.2f}")
        cols[1].markdown(f"**Price Change**: {'+' if price_change>=0 else ''}{price_change:.2f} ({price_pct:.2f}%)")
        cols[2].markdown(f"**52-Week High**: ${hist_data['High'].max():.2f}")
        cols[3].markdown(f"**52-Week Low**: ${hist_data['Low'].min():.2f}")
        
        # Plot price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'],
                                 mode='lines', name='Close Price', line=dict(color='#00ACC1')))
        fig.update_layout(title=f"Price History for {ticker_symbol}", xaxis_title="Date",
                          yaxis_title="Price (USD)", template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No historical price data available.")
