import streamlit as st
import pandas as pd
import yfinance as yf
from yahooquery import Ticker
import plotly.graph_objects as go

# ---------------------------
# 1️⃣  Load Financials (Yahoo)
# ---------------------------
@st.cache_data(ttl=3600)
def load_financial_data(ticker):
    try:
        t = Ticker(ticker)
        financials = {
            "income": t.income_statement(frequency="annual"),
            "balance": t.balance_sheet(frequency="annual"),
            "cash": t.cash_flow(frequency="annual"),
            "income_q": t.income_statement(frequency="quarterly"),
            "balance_q": t.balance_sheet(frequency="quarterly"),
            "cash_q": t.cash_flow(frequency="quarterly"),
            "profile": t.asset_profile,
            "price": t.price
        }
        return financials
    except Exception as e:
        st.error(f"❌ Error fetching Yahoo Finance data: {e}")
        return None

# ---------------------------
# 2️⃣  Display DataFrame Clean
# ---------------------------
def clean_and_show(df, title):
    if df is None or df.empty:
        st.info(f"⚠️ No {title} data available.")
        return
    df = df.reset_index().rename(columns={"asOfDate":"date"}).set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.T.sort_index(ascending=False)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.round(2)
    st.dataframe(df, use_container_width=True)
    return df

# ---------------------------
# 3️⃣  Plot Key Metrics
# ---------------------------
def plot_metric(df, fields, title):
    fig = go.Figure()
    for col in fields:
        if col in df.index:
            fig.add_trace(go.Scatter(
                x=df.columns,
                y=df.loc[col] / 1e9,  # billions scale
                mode="lines+markers",
                name=col
            ))
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Billions (USD)",
        template="plotly_dark",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 4️⃣  MAIN DISPLAY FUNCTION
# ---------------------------
def display_financials(ticker):
    ticker = ticker.upper().strip()
    st.subheader(f"📉 Financial Statements: {ticker}")

    data = load_financial_data(ticker)
    if not data or "price" not in data or data["price"] is None:
        st.error("❌ Invalid ticker or no data available. Try MSFT, AAPL, TSLA, INFY, TCS.")
        return

    # ---- Company Summary ----
    price = data.get("price", {}).get(ticker, {})
    profile = data.get("profile", {}).get(ticker, {})
    st.markdown(f"""
    ### 🏢 {price.get('longName','N/A')}  
    **Sector:** {profile.get('sector','N/A')} | **Industry:** {profile.get('industry','N/A')}  
    **Market Cap:** {price.get('marketCap','N/A'):,}  
    **52 Week Range:** {price.get('fiftyTwoWeekLow',0)} – {price.get('fiftyTwoWeekHigh',0)}
    """)

    st.markdown("---")

    # ---- Income Statement ----
    st.header("💰 Income Statement")
    inc = clean_and_show(data["income"], "Income Statement")
    if inc is not None:
        plot_metric(inc, ["TotalRevenue", "NetIncome"], "Revenue vs Net Income")

    st.markdown("---")

    # ---- Balance Sheet ----
    st.header("🏦 Balance Sheet")
    bal = clean_and_show(data["balance"], "Balance Sheet")
    if bal is not None:
        plot_metric(bal, ["TotalAssets", "TotalLiabilitiesNetMinorityInterest"], "Assets vs Liabilities")

    st.markdown("---")

    # ---- Cash Flow ----
    st.header("📌 Cash Flow")
    cash = clean_and_show(data["cash"], "Cash Flow Statement")
    if cash is not None:
        plot_metric(cash, ["OperatingCashFlow", "FreeCashFlow"], "Operating vs Free Cash Flow")

    st.success("✨ Financial data loaded successfully!")

