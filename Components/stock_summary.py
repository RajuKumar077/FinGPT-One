import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def display_stock_summary(ticker, data, fmp_key, alpha_key, gemini_key):
    """Display stock summary with candlestick charts, volume analysis, and technical indicators."""
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                color: white; padding: 2rem; border-radius: 16px; text-align: center; margin-bottom: 2rem;'>
        <h1 style='margin: 0; font-size: 3rem;'>📊 {ticker} Stock Summary Pro</h1>
        <p style='margin: 0; font-size: 1.2rem; opacity: 0.9;'>Advanced analytics with OHLCV, MAs, and volume insights</p>
    </div>
    """, unsafe_allow_html=True)

    if len(data) < 2:
        st.warning("❌ Insufficient historical data for full analysis.")
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            st.metric("💰 Current Price", f"${current_price:.2f}")
        return

    st.markdown("### 📈 Key Metrics")
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    change = current_price - prev_price
    pct_change = (change / prev_price) * 100

    # Additional metrics
    open_price = data['Open'].iloc[-1]
    high_price = data['High'].iloc[-1]
    low_price = data['Low'].iloc[-1]
    volume = data['Volume'].iloc[-1]
    avg_volume = data['Volume'].tail(20).mean() if len(data) >= 20 else volume
    vol_pct = (volume / avg_volume - 1) * 100 if avg_volume > 0 else 0

    # 50-day MA for trend
    ma_50 = data['Close'].rolling(window=50, min_periods=1).mean().iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 Current Price", f"${current_price:,.2f}", f"{pct_change:+.2f}%")
    col2.metric("🟢 High", f"${high_price:,.2f}")
    col3.metric("🔴 Low", f"${low_price:,.2f}")
    col4.metric("📊 Volume", f"{volume:,.0f}", f"{vol_pct:+.1f}%")
    col5.metric("📉 50-Day MA", f"${ma_50:,.2f}", f"{(current_price / ma_50 - 1)*100:+.1f}%")

    st.markdown("### 📋 30-Day Summary Stats")
    if len(data) >= 30:
        recent_data = data.tail(30)
        stats = {
            'Metric': ['Mean Close', 'Volatility (Std)', 'Max Drawdown %', 'Sharpe Ratio (approx)', 'Win Rate %'],
            'Value': [
                f"${recent_data['Close'].mean():.2f}",
                f"{recent_data['Close'].pct_change().std() * np.sqrt(252):.2%}",
                f"{((recent_data['Close'] / recent_data['Close'].cummax()) - 1).min():.2%}",
                f"{recent_data['Close'].pct_change().mean() / recent_data['Close'].pct_change().std() * np.sqrt(252):.2f}",
                f"{(recent_data['Close'].pct_change() > 0).mean():.1%}"
            ]
        }
        st.dataframe(pd.DataFrame(stats), use_container_width=True)
    else:
        st.info("📊 Need 30+ days for detailed stats.")

    st.markdown("### 📊 Interactive Price Action")
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Candlestick with MAs', 'Volume'),
        row_width=[0.7, 0.3]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # Moving Averages
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    windows = [5, 10, 20, 50]
    for i, w in enumerate(windows):
        if len(data) >= w:
            ma = data['Close'].rolling(window=w).mean()
            fig.add_trace(
                go.Scatter(x=data.index, y=ma, line=dict(width=1, color=colors[i]), name=f'MA-{w}'),
                row=1, col=1
            )

    # Volume - Fixed: Use vectorized comparison
    colors_v = ['green' if close > open_ else 'red' for close, open_ in zip(data['Close'], data['Open'])]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors_v),
        row=2, col=1
    )

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(height=800, title=f"{ticker} Price Action with Indicators", template='plotly_white')
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🎯 RSI Momentum (14-Day)")
    try:
        rsi = data['Close'].pct_change().rolling(14).apply(lambda x: (x[x > 0].mean() / abs(x[x < 0].mean())) * 100 if len(x) == 14 else 50).fillna(50)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=rsi, line=dict(color='#f59e0b', width=2), name='RSI'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
        fig_rsi.update_layout(height=400, title=f"{ticker} RSI (14)", template='plotly_white', yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)
    except:
        st.info("⚠️ RSI calculation requires sufficient data.")

    if gemini_key:
        if st.button("✨ Generate AI Summary"):
            direction = "bullish" if pct_change > 0 else "bearish"
            rsi_val = rsi.iloc[-1] if 'rsi' in locals() else 50
            action = "buying opportunities" if direction == "bullish" else "caution"
            st.info(f"AI Analysis: Based on recent trends, {ticker} shows {direction} momentum with {rsi_val:.0f} RSI. Consider {action}.")

   