# Components/financials.py
import streamlit as st
import yfinance as yf
import pandas as pd

@st.cache_data(ttl=3600, show_spinner="Fetching financial statements...")
def fetch_financials(ticker_symbol: str) -> dict:
    """
    Fetches financial statements (Income Statement, Balance Sheet, Cash Flow) using yfinance.

    Args:
        ticker_symbol (str): Stock ticker symbol

    Returns:
        dict: Dictionary with keys 'income', 'balance', 'cashflow', each a pandas DataFrame
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        financials = {
            'income': ticker.financials.T if ticker.financials is not None else pd.DataFrame(),
            'balance': ticker.balance_sheet.T if ticker.balance_sheet is not None else pd.DataFrame(),
            'cashflow': ticker.cashflow.T if ticker.cashflow is not None else pd.DataFrame()
        }

        # Convert numeric columns to formatted strings
        for key, df in financials.items():
            if not df.empty:
                df.index = pd.to_datetime(df.index).date
                df = df.applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
                financials[key] = df

        return financials
    except Exception as e:
        st.error(f"❌ Error fetching financial statements for {ticker_symbol}: {e}")
        return {'income': pd.DataFrame(), 'balance': pd.DataFrame(), 'cashflow': pd.DataFrame()}


def display_financials(ticker_symbol: str):
    """Display financial statements in Streamlit app in your style."""
    st.markdown(f"## 💰 Financial Statements for {ticker_symbol}")

    statements = fetch_financials(ticker_symbol)

    # Income Statement
    st.markdown("### 📄 Income Statement")
    if not statements['income'].empty:
        st.dataframe(statements['income'].head(5), use_container_width=True)
    else:
        st.warning("⚠️ No Income Statement data available.")

    # Balance Sheet
    st.markdown("### 🏦 Balance Sheet")
    if not statements['balance'].empty:
        st.dataframe(statements['balance'].head(5), use_container_width=True)
    else:
        st.warning("⚠️ No Balance Sheet data available.")

    # Cash Flow
    st.markdown("### 💵 Cash Flow")
    if not statements['cashflow'].empty:
        st.dataframe(statements['cashflow'].head(5), use_container_width=True)
    else:
        st.warning("⚠️ No Cash Flow data available.")
