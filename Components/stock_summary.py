import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

CHART_PAPER = "rgba(255,255,255,0.02)"
CHART_PLOT = "rgba(255,255,255,0.01)"
TEXT_COLOR = "#E8EEF9"
GRID_COLOR = "rgba(255,255,255,0.08)"
FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif'


def apply_chart_theme(fig, title, height):
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_PLOT,
        font=dict(color=TEXT_COLOR, family=FONT_FAMILY),
        margin=dict(l=24, r=24, t=64, b=24),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def render_metric_card(label, value, detail, accent):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta" style="color:{accent};">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_rsi(close, window=14):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def create_price_chart(ticker, data):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.2, 0.22],
        subplot_titles=("Price Action", "Volume", "RSI Momentum"),
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color="#8ED8B3",
            decreasing_line_color="#F3A6B3",
        ),
        row=1,
        col=1,
    )

    ma_colors = {10: "#9DC6FF", 20: "#7EE0C3", 50: "#F8C471"}
    for window, color in ma_colors.items():
        if len(data) >= window:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Close"].rolling(window=window).mean(),
                    name=f"MA {window}",
                    line=dict(color=color, width=2),
                ),
                row=1,
                col=1,
            )

    volume_colors = np.where(data["Close"] >= data["Open"], "#8ED8B3", "#F3A6B3")
    fig.add_trace(
        go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color=volume_colors, opacity=0.7),
        row=2,
        col=1,
    )

    rsi = compute_rsi(data["Close"])
    fig.add_trace(go.Scatter(x=data.index, y=rsi, name="RSI", line=dict(color="#9DC6FF", width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(243,166,179,0.7)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(142,216,179,0.7)", row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=False)
    return apply_chart_theme(fig, f"{ticker} market overview", 880)


def display_stock_summary(ticker, data, fmp_key, alpha_key, gemini_key):
    st.markdown(
        f"""
        <section class="hero-panel">
            <div class="hero-kicker">Market Overview</div>
            <h1>{ticker} stock summary</h1>
            <p>Cleaner technicals, richer momentum signals, and a softer Apple-inspired glass layout for fast scanning.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if len(data) < 2:
        st.warning("Not enough historical data to build the full dashboard.")
        if not data.empty:
            render_metric_card("Current Price", f"${data['Close'].iloc[-1]:,.2f}", "Latest close", "#9DC6FF")
        return

    current_price = float(data["Close"].iloc[-1])
    prev_price = float(data["Close"].iloc[-2])
    pct_change = ((current_price - prev_price) / prev_price) * 100
    high_price = float(data["High"].iloc[-1])
    low_price = float(data["Low"].iloc[-1])
    volume = float(data["Volume"].iloc[-1])
    avg_volume = float(data["Volume"].tail(20).mean())
    ma_20 = float(data["Close"].rolling(window=20, min_periods=1).mean().iloc[-1])
    ma_50 = float(data["Close"].rolling(window=50, min_periods=1).mean().iloc[-1])
    annualized_vol = float(data["Close"].pct_change().tail(60).std() * np.sqrt(252))
    rsi = compute_rsi(data["Close"])

    st.markdown("### Snapshot")
    cols = st.columns(5)
    with cols[0]:
        render_metric_card("Current Price", f"${current_price:,.2f}", f"{pct_change:+.2f}% today", "#7EE0C3" if pct_change >= 0 else "#F3A6B3")
    with cols[1]:
        render_metric_card("Session Range", f"${low_price:,.2f} - ${high_price:,.2f}", "Day low/high", "#9DC6FF")
    with cols[2]:
        render_metric_card("Volume", f"{volume:,.0f}", f"{((volume / avg_volume) - 1) * 100:+.1f}% vs 20D avg", "#F8C471")
    with cols[3]:
        render_metric_card("Trend", f"${ma_20:,.2f}", f"20D MA | 50D MA ${ma_50:,.2f}", "#9DC6FF")
    with cols[4]:
        render_metric_card("Momentum", f"{rsi.iloc[-1]:.1f} RSI", f"Volatility {annualized_vol:.1%}", "#7EE0C3")

    if len(data) >= 30:
        recent = data.tail(30).copy()
        drawdown = ((recent["Close"] / recent["Close"].cummax()) - 1).min()
        returns = recent["Close"].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() not in (0, np.nan) else np.nan
        stat_frame = pd.DataFrame(
            {
                "Metric": ["30D Mean Close", "30D Return", "Max Drawdown", "Sharpe Ratio", "Positive Days"],
                "Value": [
                    f"${recent['Close'].mean():,.2f}",
                    f"{((recent['Close'].iloc[-1] / recent['Close'].iloc[0]) - 1):+.2%}",
                    f"{drawdown:.2%}",
                    f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A",
                    f"{(returns > 0).mean():.1%}",
                ],
            }
        )
        st.markdown("### Performance Notes")
        st.dataframe(stat_frame, use_container_width=True, hide_index=True)

    st.markdown("### Price Action")
    st.plotly_chart(create_price_chart(ticker, data.tail(180)), use_container_width=True)

    if gemini_key:
        if st.button("Generate AI Summary"):
            trend = "constructive" if current_price >= ma_20 >= ma_50 else "mixed"
            st.info(
                f"{ticker} looks {trend} right now. Price is {pct_change:+.2f}% on the day, RSI sits at {rsi.iloc[-1]:.1f}, and the 20-day average is ${ma_20:,.2f}."
            )
