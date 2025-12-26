# Components/financials.py
import yfinance as yf
import pandas as pd
import streamlit as st

def fetch_financials(ticker_symbol):
    """Fetch income statement, balance sheet, and cash flow for a ticker using yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        income = ticker.financials.transpose()
        balance = ticker.balance_sheet.transpose()
        cashflow = ticker.cashflow.transpose()
        return income, balance, cashflow
    except Exception as e:
        st.error(f"❌ Error fetching financials for {ticker_symbol}: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def display_financials(ticker_symbol):
    st.subheader(f"Financial Statements for {ticker_symbol}")
    
    income, balance, cashflow = fetch_financials(ticker_symbol)
    
    if not income.empty:
        st.markdown("### Income Statement")
        st.dataframe(income)
    else:
        st.warning("⚠️ No Income Statement data available.")

    if not balance.empty:
        st.markdown("### Balance Sheet")
        st.dataframe(balance)
    else:
        st.warning("⚠️ No Balance Sheet data available.")

    if not cashflow.empty:
        st.markdown("### Cash Flow Statement")
        st.dataframe(cashflow)
    else:
        st.warning("⚠️ No Cash Flow data available.")
