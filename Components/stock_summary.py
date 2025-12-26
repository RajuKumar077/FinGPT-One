import streamlit as st

def display_stock_summary(ticker, data, fmp_key, alpha_key, gemini_key):
    """Accepts 5 arguments and includes safety checks for data length."""
    st.header(f"📊 {ticker} Summary")

    if len(data) < 2:
        st.warning("Insufficient historical data for a full comparison.")
        current_price = data['Close'].iloc[-1]
        st.metric("Price", f"${current_price:.2f}")
    else:
        # Prevent "Index out of bounds" by ensuring iloc[-2] exists
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        change = current_price - prev_price
        pct_change = (change / prev_price) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"${current_price:,.2f}", f"{pct_change:+.2f}%")
        c2.metric("High", f"${data['High'].iloc[-1]:,.2f}")
        c3.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")

    st.subheader("Price History")
    st.line_chart(data['Close'])

    if gemini_key:
        st.success("🤖 AI Insights: Ready")
    else:
        st.info("💡 Connect Gemini API for AI Analysis")