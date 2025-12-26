import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from yahooquery import Ticker

def display_stock_summary(ticker_symbol, hist_data):
    """
    Displays the company overview and price action.
    Uses yahooquery for profile and yfinance (hist_data) for pricing.
    """
    # 1. Safety Check: Ensure data exists
    if hist_data is None or hist_data.empty:
        st.error(f"❌ No historical data found for {ticker_symbol}.")
        return

    st.title(f"🔍 {ticker_symbol} - Market Summary")

    try:
        # 2. Fetch Profile via yahooquery (Free)
        t = Ticker(ticker_symbol)
        all_profile = t.asset_profile
        
        # Handle cases where yahooquery returns an error string instead of a dict
        if isinstance(all_profile, dict) and ticker_symbol in all_profile:
            profile = all_profile[ticker_symbol]
        else:
            profile = {}

        # 3. Key Metrics Header
        latest_close = float(hist_data['Close'].iloc[-1])
        prev_close = float(hist_data['Close'].iloc[-2])
        change = latest_close - prev_close
        pct_change = (change / prev_close) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"${latest_close:.2f}", f"{pct_change:.2f}%")
        m2.metric("52W High", f"${hist_data['High'].max():.2f}")
        m3.metric("52W Low", f"${hist_data['Low'].min():.2f}")
        m4.metric("Avg Volume", f"{int(hist_data['Volume'].mean()):,}")

        st.divider()

        # 4. Company Info Section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Business Summary")
            description = profile.get('longBusinessSummary', "Description not available.")
            st.write(description)

        with col2:
            st.markdown("### Details")
            st.write(f"**Sector:** {profile.get('sector', 'N/A')}")
            st.write(f"**Industry:** {profile.get('industry', 'N/A')}")
            st.write(f"**Employees:** {profile.get('fullTimeEmployees', 'N/A'):,}" if isinstance(profile.get('fullTimeEmployees'), (int, float)) else "**Employees:** N/A")
            
            website = profile.get('website', '#')
            st.markdown(f"**Website:** [Visit Site]({website})")

        # 5. Visual Chart
        st.markdown("### Price Action (Last 5 Years)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_data.index, 
            y=hist_data['Close'], 
            fill='tozeroy',
            line=dict(color='#00ACC1', width=2),
            name="Close Price"
        ))
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ An error occurred while rendering the summary: {e}")
